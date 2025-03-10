"""
セーリング戦略分析システム - 最適VMG計算エンジン

推定された風向風速データを基に、各地点での理論上最適な
セーリング戦略（タック/ジャイブのタイミングや最適な進行方向）を計算します。
"""

import pandas as pd
import numpy as np
import math
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import warnings

# 内部モジュールのインポート (sailing_data_processor パッケージ内)
try:
    from .utilities.math_utils import normalize_angle, angle_difference
except ImportError:
    # スタンドアロン実行の場合はこちらを使用
    def normalize_angle(angle):
        """角度を0-360度の範囲に正規化"""
        return angle % 360
        
    def angle_difference(angle1, angle2):
        """2つの角度間の最小差分を計算（-180〜180度の範囲）"""
        return ((angle1 - angle2 + 180) % 360) - 180


class OptimalVMGCalculator:
    """最適VMG計算エンジン - 風向風速データを基に最適セーリング戦略を計算"""
    
    def __init__(self):
        """初期化"""
        # 艇種データライブラリ
        self.boat_types = {}
        # 風向風速データ
        self.wind_field = None
        # 計算結果キャッシュ
        self.vmg_cache = {}
        # 標準艇種をロード
        self._load_standard_boat_types()
        # 計算設定
        self.config = {
            'use_parallel': True,  # 並列計算を使用するか
            'cache_results': True,  # 計算結果をキャッシュするか
            'max_workers': max(1, multiprocessing.cpu_count() - 1),  # 並列計算の最大ワーカー数
            'path_resolution': 200,  # パス計算の解像度（メートル）
            'min_distance': 50,  # 目標到達判定距離（メートル）
            'safety_margin': 50.0,  # 安全マージン（メートル）
            'use_vectorization': True  # ベクトル化計算を使用するか
        }
    
    def update_config(self, **kwargs):
        """
        計算設定を更新
        
        Parameters:
        -----------
        **kwargs
            更新する設定のキーと値
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            else:
                warnings.warn(f"未知の設定キー: {key}")
    
    def load_polar_data(self, boat_type: str, file_path: str) -> bool:
        """
        極座標データファイルを読み込む
        
        Parameters:
        -----------
        boat_type : str
            艇種の識別子
        file_path : str
            極座標データファイルのパス
            
        Returns:
        --------
        bool
            読み込みの成功/失敗
        """
        try:
            # CSVファイルから極座標データを読み込み
            polar_data = pd.read_csv(file_path)
            
            # 最初の列が 'twa/tws' であることを確認
            if polar_data.columns[0] != 'twa/tws':
                # 他の可能な形式もチェック
                if 'twa' in polar_data.columns and 'tws' in polar_data.columns:
                    # twa と tws の列がある場合の処理
                    # ピボットして標準形式に変換
                    polar_data = polar_data.pivot(index='twa', columns='tws', values='speed')
                    polar_data.index.name = 'twa/tws'
                else:
                    raise ValueError(f"サポートされていないポーラーデータ形式: {file_path}")
            
            # 風向角をインデックスに設定
            polar_data.set_index('twa/tws', inplace=True)
            
            # インデックスを数値型に変換
            polar_data.index = pd.to_numeric(polar_data.index, errors='coerce')
            
            # 列名も数値型に変換
            polar_data.columns = pd.to_numeric(polar_data.columns, errors='coerce')
            
            # 最適VMG値を計算（事前計算）
            upwind_optimal = self._calculate_optimal_vmg_angles(polar_data, upwind=True)
            downwind_optimal = self._calculate_optimal_vmg_angles(polar_data, upwind=False)
            
            # 艇種データを登録
            self.boat_types[boat_type] = {
                'display_name': boat_type,
                'polar_data': polar_data,
                'upwind_optimal': upwind_optimal,
                'downwind_optimal': downwind_optimal
            }
            
            # VMGキャッシュをクリア
            if boat_type in self.vmg_cache:
                del self.vmg_cache[boat_type]
            
            return True
            
        except Exception as e:
            print(f"ポーラーデータの読み込みエラー ({boat_type}): {e}")
            return False
    
    def set_wind_field(self, wind_field: Dict[str, Any]) -> None:
        """
        風向風速データを設定
        
        Parameters:
        -----------
        wind_field : Dict[str, Any]
            風向風速データ（WindFieldInterpolator から取得したもの）
        """
        self.wind_field = wind_field
        # キャッシュをクリア
        self.vmg_cache = {}
    
    def calculate_optimal_vmg(self, boat_type: str, lat: float, lon: float, 
                            target_lat: float, target_lon: float) -> Dict[str, Any]:
        """
        指定された地点からターゲットへの最適VMGを計算
        
        Parameters:
        -----------
        boat_type : str
            艇種の識別子
        lat, lon : float
            現在位置の緯度・経度
        target_lat, target_lon : float
            目標地点の緯度・経度
            
        Returns:
        --------
        Dict[str, Any]
            最適VMGの情報（最適進路、最適艇速、推定所要時間など）
        """
        if boat_type not in self.boat_types:
            raise ValueError(f"未知の艇種: {boat_type}")
            
        if self.wind_field is None:
            raise ValueError("風向風速データが設定されていません")
        
        # キャッシュキーを生成
        if self.config['cache_results']:
            cache_key = f"{boat_type}_{lat:.6f}_{lon:.6f}_{target_lat:.6f}_{target_lon:.6f}"
            if cache_key in self.vmg_cache:
                return self.vmg_cache[cache_key]
        
        # 地点の風向風速を取得
        wind_info = self._get_wind_at_position(lat, lon)
        if wind_info is None:
            return None
        
        # 目標地点への方位を計算
        target_bearing = self._calculate_bearing(lat, lon, target_lat, target_lon)
        
        # 風向と目標方位の相対角度
        relative_angle = self._angle_difference(target_bearing, wind_info['direction'])
        
        # 風向の絶対値が45度未満なら風上、135度以上なら風下と判断
        if abs(relative_angle) < 45:
            # ほぼ真っすぐ目標に向かえる（リーチング）
            is_upwind = False
            is_direct = True
            optimal_twa = abs(relative_angle)
            boat_speed = self.get_boat_performance(boat_type, wind_info['speed'], optimal_twa)
            vmg = boat_speed  # 直接目標に向かうので VMG = 艇速
            
            optimal_course = target_bearing
            tack_needed = False
            
        elif abs(relative_angle) > 135:
            # 風下、直接向かえないので最適風下角を使用
            is_upwind = False
            is_direct = False
            
            # 風下での最適TWA（風向角）を取得
            optimal_twa, max_vmg = self._get_optimal_twa(boat_type, wind_info['speed'], False)
            
            # 最適コースは風向から最適TWAを加減
            if relative_angle > 0:
                optimal_course = self._normalize_angle(wind_info['direction'] + optimal_twa)
            else:
                optimal_course = self._normalize_angle(wind_info['direction'] - optimal_twa)
            
            boat_speed = self.get_boat_performance(boat_type, wind_info['speed'], optimal_twa)
            
            # 目標方向へのVMGを計算
            course_diff = abs(self._angle_difference(optimal_course, target_bearing))
            vmg = boat_speed * math.cos(math.radians(course_diff))
            
            # 反対タックの方が有利かチェック
            opposite_course = self._normalize_angle(wind_info['direction'] + (180 - optimal_twa) if relative_angle > 0 
                                                else wind_info['direction'] - (180 - optimal_twa))
            opposite_diff = abs(self._angle_difference(opposite_course, target_bearing))
            opposite_vmg = boat_speed * math.cos(math.radians(opposite_diff))
            
            tack_needed = opposite_vmg > vmg
            
            # 反対タックの方が有利な場合はコースを更新
            if tack_needed:
                optimal_course = opposite_course
                vmg = opposite_vmg
            
        else:
            # 風上、直接向かえないので最適風上角を使用
            is_upwind = True
            is_direct = False
            
            # 風上での最適TWA（風向角）を取得
            optimal_twa, max_vmg = self._get_optimal_twa(boat_type, wind_info['speed'], True)
            
            # 最適コースは風向から最適TWAを加減
            if relative_angle > 0:
                optimal_course = self._normalize_angle(wind_info['direction'] + optimal_twa)
            else:
                optimal_course = self._normalize_angle(wind_info['direction'] - optimal_twa)
            
            boat_speed = self.get_boat_performance(boat_type, wind_info['speed'], optimal_twa)
            
            # 目標方向へのVMGを計算
            course_diff = abs(self._angle_difference(optimal_course, target_bearing))
            vmg = boat_speed * math.cos(math.radians(course_diff))
            
            # 反対タックの方が有利かチェック
            opposite_course = self._normalize_angle(wind_info['direction'] - optimal_twa if relative_angle > 0 
                                                else wind_info['direction'] + optimal_twa)
            opposite_diff = abs(self._angle_difference(opposite_course, target_bearing))
            opposite_vmg = boat_speed * math.cos(math.radians(opposite_diff))
            
            tack_needed = opposite_vmg > vmg
            
            # 反対タックの方が有利な場合はコースを更新
            if tack_needed:
                optimal_course = opposite_course
                vmg = opposite_vmg
        
        # 距離と到達時間の推定
        distance = geodesic((lat, lon), (target_lat, target_lon)).meters
        eta_seconds = 0 if vmg <= 0 else distance / (vmg * 0.51444)  # ノットをm/sに変換
        
        # 結果を整理
        result = {
            'optimal_course': optimal_course,
            'boat_speed': boat_speed,
            'vmg': vmg,
            'is_upwind': is_upwind,
            'is_direct': is_direct,
            'tack_needed': tack_needed,
            'eta_seconds': eta_seconds,
            'distance_meters': distance,
            'wind_info': wind_info
        }
        
        # キャッシュに保存
        if self.config['cache_results']:
            self.vmg_cache[cache_key] = result
        
        return result
    
    def find_optimal_path(self, boat_type: str, start_lat: float, start_lon: float,
                        target_lat: float, target_lon: float, 
                        max_tacks: int = 5) -> Dict[str, Any]:
        """
        出発地点から目標地点までの最適なパスを見つける
        
        Parameters:
        -----------
        boat_type : str
            艇種の識別子
        start_lat, start_lon : float
            出発地点の緯度・経度
        target_lat, target_lon : float
            目標地点の緯度・経度
        max_tacks : int
            最大タック/ジャイブ回数
            
        Returns:
        --------
        Dict[str, Any]
            最適パス情報（経路点、推定所要時間など）
        """
        if boat_type not in self.boat_types:
            raise ValueError(f"未知の艇種: {boat_type}")
            
        if self.wind_field is None:
            raise ValueError("風向風速データが設定されていません")
        
        # パス計算用のパラメータ
        max_iterations = 200  # 最大計算回数
        min_distance = self.config['min_distance']  # 目標到達判定距離（メートル）
        step_distance = self.config['path_resolution']  # 各ステップでの進行距離（メートル）
        
        # 結果格納用
        path_points = []
        tack_points = []
        total_time = 0
        total_distance = 0
        
        # 現在位置を初期化
        current_lat, current_lon = start_lat, start_lon
        
        # 前回のタック情報
        last_tack_time = 0
        tack_count = 0
        current_leg_is_upwind = None  # 最初はどちらでもない
        
        # パス計算ループ
        for i in range(max_iterations):
            # 目標地点までの距離をチェック
            current_distance = geodesic((current_lat, current_lon), (target_lat, target_lon)).meters
            if current_distance < min_distance:
                # 目標に到達
                break
                
            # 現在地点から最適VMGを計算
            vmg_info = self.calculate_optimal_vmg(
                boat_type, current_lat, current_lon, target_lat, target_lon
            )
            
            if vmg_info is None:
                # 計算失敗
                break
            
            # 風上/風下の変化を検出（レグの区切り）
            if current_leg_is_upwind is not None and current_leg_is_upwind != vmg_info['is_upwind']:
                # 新しいレグとして扱う
                current_leg_is_upwind = vmg_info['is_upwind']
            elif current_leg_is_upwind is None:
                # 最初のレグ
                current_leg_is_upwind = vmg_info['is_upwind']
            
            # タックの判断
            if vmg_info['tack_needed'] and tack_count < max_tacks:
                # タックが必要で、最大タック数以下の場合
                tack_count += 1
                
                # タックポイントを記録
                tack_points.append({
                    'lat': current_lat,
                    'lon': current_lon,
                    'time': total_time,
                    'is_upwind': vmg_info['is_upwind']
                })
            
            # パスポイントを記録
            path_points.append({
                'lat': current_lat,
                'lon': current_lon,
                'course': vmg_info['optimal_course'],
                'speed': vmg_info['boat_speed'],
                'time': total_time,
                'wind_direction': vmg_info['wind_info']['direction'],
                'wind_speed': vmg_info['wind_info']['speed'],
                'is_upwind': vmg_info['is_upwind']
            })
            
            # 次の位置を計算
            course_rad = math.radians(vmg_info['optimal_course'])
            boat_speed_ms = vmg_info['boat_speed'] * 0.51444  # ノットをm/sに変換
            time_step = step_distance / boat_speed_ms if boat_speed_ms > 0 else 0  # 移動に必要な時間（秒）
            
            # 緯度・経度の変化量を計算（メートルを度に変換）
            lat_change = math.cos(course_rad) * step_distance / 111000  # 1度 ≈ 111km
            lon_change = math.sin(course_rad) * step_distance / (111000 * math.cos(math.radians(current_lat)))
            
            # 位置と時間を更新
            current_lat += lat_change
            current_lon += lon_change
            total_time += time_step
            total_distance += step_distance
        
        # 平均速度を計算
        avg_speed = total_distance / (total_time * 0.51444) if total_time > 0 else 0
        
        # 最終地点が目標に到達していない場合は最後のパスポイントを目標地点に設定
        if path_points and current_distance >= min_distance:
            last_point = path_points[-1].copy()
            last_point['lat'] = target_lat
            last_point['lon'] = target_lon
            path_points.append(last_point)
        
        # 結果を整理
        return {
            'path_points': path_points,
            'tack_points': tack_points,
            'total_distance': total_distance,
            'total_time': total_time,
            'avg_speed': avg_speed,
            'tack_count': tack_count
        }
    
    def calculate_optimal_route_for_course(self, boat_type: str, 
                                         waypoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        指定されたコース（ウェイポイントリスト）に対する最適戦略を計算
        
        Parameters:
        -----------
        boat_type : str
            艇種の識別子
        waypoints : List[Dict[str, Any]]
            コースのウェイポイントリスト
            
        Returns:
        --------
        Dict[str, Any]
            最適戦略情報（レッグごとの戦略、推定所要時間など）
        """
        if boat_type not in self.boat_types:
            raise ValueError(f"未知の艇種: {boat_type}")
            
        if self.wind_field is None:
            raise ValueError("風向風速データが設定されていません")
            
        if len(waypoints) < 2:
            raise ValueError("コースには少なくとも2つのウェイポイントが必要です")
        
        # 並列処理を使用するかどうか
        use_parallel = self.config['use_parallel'] and len(waypoints) > 2
        
        if use_parallel:
            # 並列処理のためのデータを準備
            start_points = []
            end_points = []
            
            for i in range(len(waypoints) - 1):
                start = waypoints[i]
                end = waypoints[i + 1]
                
                start_points.append((start['lat'], start['lon']))
                end_points.append((end['lat'], end['lon']))
            
            # 並列処理で最適パスを計算
            path_results = self._parallelize_path_calculation(
                boat_type, start_points, end_points
            )
        else:
            # 逐次処理
            path_results = []
            
            for i in range(len(waypoints) - 1):
                start = waypoints[i]
                end = waypoints[i + 1]
                
                # レッグの最適パスを計算
                path = self.find_optimal_path(
                    boat_type, 
                    start['lat'], start['lon'],
                    end['lat'], end['lon']
                )
                
                path_results.append(path)
        
        # レッグごとの最適パスを計算
        legs = []
        total_time = 0
        total_distance = 0
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            path = path_results[i]
            
            # レッグ情報を作成
            leg_info = {
                'start_waypoint': start,
                'end_waypoint': end,
                'leg_number': i + 1,
                'path': path,
                'start_time': total_time,
                'end_time': total_time + path['total_time'],
                'is_upwind': self._is_upwind_leg(start, end)
            }
            
            # レッグタイプの決定
            if leg_info['is_upwind']:
                leg_info['leg_type'] = 'upwind'
            else:
                leg_info['leg_type'] = 'downwind'
            
            # 合計時間と距離を更新
            total_time += path['total_time']
            total_distance += path['total_distance']
            
            legs.append(leg_info)
        
        # コース全体の結果
        return {
            'boat_type': boat_type,
            'legs': legs,
            'total_time': total_time,
            'total_distance': total_distance,
            'waypoints': waypoints,
            'total_tack_count': sum(leg['path']['tack_count'] for leg in legs)
        }
    
    def get_boat_performance(self, boat_type: str, wind_speed: float, 
                           wind_angle: float) -> float:
        """
        特定の風速・風向に対する艇の性能（艇速）を取得
        
        Parameters:
        -----------
        boat_type : str
            艇種の識別子
        wind_speed : float
            風速（ノット）
        wind_angle : float
            風向角（度）
            
        Returns:
        --------
        float
            推定艇速（ノット）
        """
        if boat_type not in self.boat_types:
            raise ValueError(f"未知の艇種: {boat_type}")
        
        # 風向角を0-180度の範囲に正規化
        wind_angle = abs(wind_angle) % 360
        if wind_angle > 180:
            wind_angle = 360 - wind_angle
            
        # ポーラーデータから艇速を補間
        polar_data = self.boat_types[boat_type]['polar_data']
        
        return self._interpolate_boat_speed(polar_data, wind_speed, wind_angle)
    
    def find_optimal_twa(self, boat_type: str, wind_speed: float, 
                       upwind: bool = True) -> Tuple[float, float]:
        """
        指定された風速に対する最適な風向角（最大VMG）を見つける
        
        Parameters:
        -----------
        boat_type : str
            艇種の識別子
        wind_speed : float
            風速（ノット）
        upwind : bool
            True: 風上向き、False: 風下向き
            
        Returns:
        --------
        Tuple[float, float]
            (最適風向角, 最大VMG)
        """
        # _get_optimal_twa の別名（公開API用）
        return self._get_optimal_twa(boat_type, wind_speed, upwind)
    
    def visualize_optimal_path(self, path_data: Dict[str, Any], 
                             show_wind: bool = True,
                             save_path: str = None) -> plt.Figure:
        """
        最適パスを可視化
        
        Parameters:
        -----------
        path_data : Dict[str, Any]
            find_optimal_path の返り値
        show_wind : bool
            風向風速を表示するかどうか
        save_path : str, optional
            保存先パス
            
        Returns:
        --------
        plt.Figure
            プロット図
        """
        # パスポイントを取得
        path_points = path_data['path_points']
        tack_points = path_data['tack_points']
        
        if not path_points:
            raise ValueError("パスポイントがありません")
        
        # 緯度・経度の抽出
        lats = [p['lat'] for p in path_points]
        lons = [p['lon'] for p in path_points]
        
        # タックポイントの緯度・経度
        tack_lats = [p['lat'] for p in tack_points]
        tack_lons = [p['lon'] for p in tack_points]
        
        # プロット設定
        plt.figure(figsize=(10, 8))
        
        # パスをプロット
        plt.plot(lons, lats, 'b-', linewidth=2, label='Optimal Path')
        
        # 始点と終点を強調
        plt.plot(lons[0], lats[0], 'go', markersize=10, label='Start')
        plt.plot(lons[-1], lats[-1], 'ro', markersize=10, label='End')
        
        # タックポイントをプロット
        if tack_points:
            plt.plot(tack_lons, tack_lats, 'yx', markersize=10, label='Tack Points')
        
        # 風向風速を表示
        if show_wind and 'wind_direction' in path_points[0] and 'wind_speed' in path_points[0]:
            # 表示するポイントの間引き
            skip = max(1, len(path_points) // 20)
            
            for i in range(0, len(path_points), skip):
                p = path_points[i]
                
                # 風向を矢印で表示（風が吹いてくる方向から）
                wind_dir_rad = math.radians(p['wind_direction'])
                u = -math.sin(wind_dir_rad) * 0.0001 * p['wind_speed']
                v = -math.cos(wind_dir_rad) * 0.0001 * p['wind_speed']
                
                plt.arrow(p['lon'], p['lat'], u, v, head_width=0.0003, 
                        head_length=0.0006, fc='r', ec='r', alpha=0.6)
        
        # グリッドと軸ラベル
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Optimal Sailing Path')
        plt.legend()
        
        # アスペクト比を調整
        plt.axis('equal')
        
        # 保存処理
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
        return plt.gcf()  # 現在の図を返す
    
    def visualize_course_strategy(self, route_data: Dict[str, Any],
                                show_wind: bool = True,
                                save_path: str = None) -> plt.Figure:
        """
        コース全体の戦略を可視化
        
        Parameters:
        -----------
        route_data : Dict[str, Any]
            calculate_optimal_route_for_course の返り値
        show_wind : bool
            風向風速を表示するかどうか
        save_path : str, optional
            保存先パス
            
        Returns:
        --------
        plt.Figure
            プロット図
        """
        if not route_data or 'legs' not in route_data or not route_data['legs']:
            raise ValueError("有効なルートデータがありません")
        
        boat_type = route_data.get('boat_type', 'Unknown')
        waypoints = route_data.get('waypoints', [])
        
        # プロット設定
        plt.figure(figsize=(12, 10))
        
        # 各レッグのパスを描画
        for i, leg in enumerate(route_data['legs']):
            path = leg['path']
            if not path['path_points']:
                continue
                
            # パスポイントの緯度・経度を抽出
            lats = [p['lat'] for p in path['path_points']]
            lons = [p['lon'] for p in path['path_points']]
            
            # レッグの種類に基づいて色を選択
            color = 'red' if leg['is_upwind'] else 'blue'
            label = f"Leg {i+1}: {leg['start_waypoint']['name']} → {leg['end_waypoint']['name']}"
            
            # パスを描画
            plt.plot(lons, lats, color=color, linewidth=2, label=label)
            
            # タックポイントを描画
            tack_lats = [p['lat'] for p in path['tack_points']]
            tack_lons = [p['lon'] for p in path['tack_points']]
            
            if tack_lats:
                plt.plot(tack_lons, tack_lats, 'yx', markersize=8)
            
            # 風向風速を表示
            if show_wind and path['path_points'] and 'wind_direction' in path['path_points'][0]:
                # 表示するポイントの間引き
                skip = max(1, len(path['path_points']) // 5)
                
                for j in range(0, len(path['path_points']), skip):
                    p = path['path_points'][j]
                    
                    # 風向を矢印で表示（風が吹いてくる方向から）
                    if 'wind_direction' in p and 'wind_speed' in p:
                        wind_dir_rad = math.radians(p['wind_direction'])
                        u = -math.sin(wind_dir_rad) * 0.0001 * p['wind_speed']
                        v = -math.cos(wind_dir_rad) * 0.0001 * p['wind_speed']
                        
                        plt.arrow(p['lon'], p['lat'], u, v, head_width=0.0003, 
                                head_length=0.0006, fc='r', ec='r', alpha=0.6)
        
        # ウェイポイントを描画
        for wp in waypoints:
            plt.plot(wp['lon'], wp['lat'], 'ko', markersize=10)
            plt.text(wp['lon'] + 0.001, wp['lat'] + 0.001, wp['name'], fontsize=10)
        
        # グリッドと軸ラベル
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f"{boat_type} - Full Course Optimal Strategy")
        plt.legend()
        
        # アスペクト比を調整
        plt.axis('equal')
        
        # 保存処理
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
        return plt.gcf()  # 現在の図を返す
    
    # ----- パフォーマンス最適化メソッド -----
    
    def _vectorized_boat_performance(self, boat_type: str, wind_speeds: np.ndarray, 
                                   wind_angles: np.ndarray) -> np.ndarray:
        """
        風速と風向角の配列に対して一括で艇速を計算（ベクトル化処理）
        
        Parameters:
        -----------
        boat_type : str
            艇種の識別子
        wind_speeds : np.ndarray
            風速の配列（ノット）
        wind_angles : np.ndarray
            風向角の配列（度）
            
        Returns:
        --------
        np.ndarray
            艇速の配列（ノット）
        """
        if boat_type not in self.boat_types:
            raise ValueError(f"未知の艇種: {boat_type}")
        
        # 風向角を0-180度の範囲に正規化
        angles = np.abs(wind_angles) % 360
        angles = np.where(angles > 180, 360 - angles, angles)
        
        # ポーラーデータを取得
        polar_data = self.boat_types[boat_type]['polar_data']
        
        # インデックスとカラムをNumPy配列に変換
        twa_indices = np.array([float(twa) for twa in polar_data.index])
        tws_columns = np.array([float(tws) for tws in polar_data.columns])
        
        # 結果配列を初期化
        boat_speeds = np.zeros_like(wind_speeds, dtype=float)
        
        # 各ポイントに対して処理
        for i in range(len(wind_speeds)):
            ws = wind_speeds[i]
            wa = angles[i]
            
            # 範囲チェック
            ws = max(min(ws, tws_columns.max()), tws_columns.min())
            wa = max(min(wa, twa_indices.max()), twa_indices.min())
            
            # 最近接点のインデックスを見つける
            twa_idx = np.abs(twa_indices - wa).argmin()
            tws_idx = np.abs(tws_columns - ws).argmin()
            
            # 4つの最近接点のインデックスを取得
            twa_lower_idx = np.where(twa_indices <= wa)[0][-1] if np.any(twa_indices <= wa) else 0
            twa_upper_idx = np.where(twa_indices >= wa)[0][0] if np.any(twa_indices >= wa) else len(twa_indices)-1
            tws_lower_idx = np.where(tws_columns <= ws)[0][-1] if np.any(tws_columns <= ws) else 0
            tws_upper_idx = np.where(tws_columns >= ws)[0][0] if np.any(tws_columns >= ws) else len(tws_columns)-1
            
            # 同じ点の場合は直接値を使用
            if twa_lower_idx == twa_upper_idx and tws_lower_idx == tws_upper_idx:
                boat_speeds[i] = float(polar_data.iloc[twa_lower_idx, tws_lower_idx])
                continue
            
            # 双線形補間のための4点
            twa_lower = twa_indices[twa_lower_idx]
            twa_upper = twa_indices[twa_upper_idx]
            tws_lower = tws_columns[tws_lower_idx]
            tws_upper = tws_columns[tws_upper_idx]
            
            # 双線形補間の重み
            alpha = (wa - twa_lower) / (twa_upper - twa_lower) if twa_upper != twa_lower else 0
            beta = (ws - tws_lower) / (tws_upper - tws_lower) if tws_upper != tws_lower else 0
            
            # 4点の値
            v00 = float(polar_data.iloc[twa_lower_idx, tws_lower_idx])
            v01 = float(polar_data.iloc[twa_lower_idx, tws_upper_idx])
            v10 = float(polar_data.iloc[twa_upper_idx, tws_lower_idx])
            v11 = float(polar_data.iloc[twa_upper_idx, tws_upper_idx])
            
            # 双線形補間
            v0 = v00 * (1 - beta) + v01 * beta
            v1 = v10 * (1 - beta) + v11 * beta
            boat_speeds[i] = v0 * (1 - alpha) + v1 * alpha
        
        return boat_speeds
    
    def batch_calculate_optimal_vmg(self, boat_type: str, points: np.ndarray, 
                                 target_lat: float, target_lon: float) -> List[Dict[str, Any]]:
        """
        複数の地点からターゲットへの最適VMGを一括計算
        
        Parameters:
        -----------
        boat_type : str
            艇種の識別子
        points : np.ndarray
            位置の配列、形状は (n, 2)、各行は [lat, lon]
        target_lat, target_lon : float
            目標地点の緯度・経度
            
        Returns:
        --------
        List[Dict[str, Any]]
            各地点の最適VMG情報
        """
        if boat_type not in self.boat_types:
            raise ValueError(f"未知の艇種: {boat_type}")
            
        if self.wind_field is None:
            raise ValueError("風向風速データが設定されていません")
        
        # 結果を格納するリスト
        results = []
        
        # 各ポイントに対して計算
        for point in points:
            lat, lon = point
            
            # 通常のVMG計算を使用
            result = self.calculate_optimal_vmg(
                boat_type, lat, lon, target_lat, target_lon
            )
            
            results.append(result)
        
        return results
    
    def _parallelize_path_calculation(self, boat_type: str, start_points: List[Tuple[float, float]],
                                    target_points: List[Tuple[float, float]],
                                    max_tacks: int = 5) -> List[Dict[str, Any]]:
        """
        複数の経路に対する最適パス計算を並列処理で実行
        
        Parameters:
        -----------
        boat_type : str
            艇種の識別子
        start_points : List[Tuple[float, float]]
            開始点の配列、各要素は (lat, lon)
        target_points : List[Tuple[float, float]]
            目標点の配列、各要素は (lat, lon)
        max_tacks : int
            最大タック/ジャイブ回数
            
        Returns:
        --------
        List[Dict[str, Any]]
            各経路の最適パス情報
        """
        if boat_type not in self.boat_types:
            raise ValueError(f"未知の艇種: {boat_type}")
            
        if self.wind_field is None:
            raise ValueError("風向風速データが設定されていません")
        
        # 使用するCPUコア数
        num_cores = min(self.config['max_workers'], len(start_points))
        
        # 並列計算用の関数
        def _calculate_path(args):
            start_lat, start_lon, target_lat, target_lon, max_tacks = args
            try:
                return self.find_optimal_path(
                    boat_type=boat_type,
                    start_lat=start_lat,
                    start_lon=start_lon,
                    target_lat=target_lat,
                    target_lon=target_lon,
                    max_tacks=max_tacks
                )
            except Exception as e:
                print(f"パス計算エラー: {e}")
                return None
        
        # 計算用の引数リスト
        args_list = [
            (start[0], start[1], target[0], target[1], max_tacks)
            for start, target in zip(start_points, target_points)
        ]
        
        # 並列処理を実行
        results = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(_calculate_path, args_list))
        
        # None結果を処理
        results = [r if r is not None else {} for r in results]
        
        return results
    
    # ----- 安全チェックと評価メソッド -----
    
    def get_calculation_quality_report(self) -> Dict[str, Any]:
        """
        計算品質レポートを生成
        
        計算の信頼性、精度、性能に関する評価情報を提供
        
        Returns:
        --------
        Dict[str, Any]
            品質レポート情報
        """
        quality_report = {
            'polar_data_quality': {},
            'wind_field_quality': {},
            'general_quality': {}
        }
        
        # ポーラーデータの品質チェック
        if self.boat_types:
            for boat_id, boat_data in self.boat_types.items():
                if 'polar_data' in boat_data:
                    polar_df = boat_data['polar_data']
                    
                    quality_report['polar_data_quality'][boat_id] = {
                        'data_points': polar_df.size,
                        'wind_angle_range': [float(polar_df.index.min()), float(polar_df.index.max())],
                        'wind_speed_range': [float(min(polar_df.columns)), float(max(polar_df.columns))],
                        'completeness': not polar_df.isna().any().any(),
                        'has_upwind_optimal': bool(boat_data.get('upwind_optimal')),
                        'has_downwind_optimal': bool(boat_data.get('downwind_optimal'))
                    }
        
        # 風向風速データの品質チェック
        if self.wind_field:
            wind_conf = self.wind_field.get('confidence', np.ones_like(self.wind_field['wind_direction']))
            
            quality_report['wind_field_quality'] = {
                'grid_resolution': self.wind_field['lat_grid'].shape,
                'lat_range': [float(np.min(self.wind_field['lat_grid'])), float(np.max(self.wind_field['lat_grid']))],
                'lon_range': [float(np.min(self.wind_field['lon_grid'])), float(np.max(self.wind_field['lon_grid']))],
                'average_confidence': float(np.mean(wind_conf)),
                'min_confidence': float(np.min(wind_conf)),
                'wind_speed_range': [float(np.min(self.wind_field['wind_speed'])), 
                                    float(np.max(self.wind_field['wind_speed']))],
                'timestamp': self.wind_field.get('time', datetime.now()).isoformat()
            }
        
        # 全般的な品質情報
        quality_report['general_quality'] = {
            'available_boat_types': len(self.boat_types),
            'has_wind_field': self.wind_field is not None,
            'vmg_cache_size': len(self.vmg_cache),
            'calculation_engine_version': '1.0.0'
        }
        
        return quality_report
    
    def check_path_safety(self, path_data: Dict[str, Any], safety_margin_meters: float = None,
                       obstacle_points: List[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        計算されたパスの安全性をチェック
        
        Parameters:
        -----------
        path_data : Dict[str, Any]
            find_optimal_path の返り値
        safety_margin_meters : float, optional
            安全マージン（メートル）
        obstacle_points : List[Tuple[float, float]], optional
            障害物ポイントのリスト（緯度、経度）
            
        Returns:
        --------
        Dict[str, Any]
            安全性評価結果
        """
        if safety_margin_meters is None:
            safety_margin_meters = self.config['safety_margin']
        
        safety_report = {
            'is_safe': True,
            'warnings': [],
            'dangerous_points': []
        }
        
        # パスデータのチェック
        if not path_data or 'path_points' not in path_data or not path_data['path_points']:
            safety_report['is_safe'] = False
            safety_report['warnings'].append("パスデータが無効です")
            return safety_report
        
        # 障害物チェック
        if obstacle_points:
            for obstacle in obstacle_points:
                obs_lat, obs_lon = obstacle
                
                # 各パスポイントと障害物の距離をチェック
                for i, point in enumerate(path_data['path_points']):
                    distance = geodesic((point['lat'], point['lon']), (obs_lat, obs_lon)).meters
                    
                    if distance < safety_margin_meters:
                        safety_report['is_safe'] = False
                        safety_report['warnings'].append(f"パスポイント {i} が障害物に近すぎます")
                        safety_report['dangerous_points'].append({
                            'path_point_index': i,
                            'path_point': (point['lat'], point['lon']),
                            'obstacle': (obs_lat, obs_lon),
                            'distance': distance,
                            'safety_margin': safety_margin_meters
                        })
        
        # タックの安全性チェック
        if 'tack_points' in path_data and path_data['tack_points']:
            for i, tack in enumerate(path_data['tack_points']):
                # タック前後のパスポイントを見つける
                tack_time = tack['time']
                
                # 実際の操船でタックに必要な最小距離（メートル）
                min_tack_distance = 20.0
                
                # タック前後のポイントのインデックスを見つける
                pre_tack_idx = None
                post_tack_idx = None
                
                for j, point in enumerate(path_data['path_points']):
                    if 'time' in point and point['time'] <= tack_time:
                        pre_tack_idx = j
                    if 'time' in point and point['time'] > tack_time and post_tack_idx is None:
                        post_tack_idx = j
                        break
                
                # 前後のポイントがある場合
                if pre_tack_idx is not None and post_tack_idx is not None:
                    pre_point = path_data['path_points'][pre_tack_idx]
                    post_point = path_data['path_points'][post_tack_idx]
                    
                    # タック時の距離の確認
                    pre_distance = geodesic((pre_point['lat'], pre_point['lon']), 
                                          (tack['lat'], tack['lon'])).meters
                    post_distance = geodesic((post_point['lat'], post_point['lon']), 
                                          (tack['lat'], tack['lon'])).meters
                    
                    if pre_distance < min_tack_distance or post_distance < min_tack_distance:
                        safety_report['is_safe'] = False
                        safety_report['warnings'].append(f"タック {i+1} の前後の距離が短すぎます")
                        safety_report['dangerous_points'].append({
                            'tack_index': i,
                            'tack_point': (tack['lat'], tack['lon']),
                            'pre_distance': pre_distance,
                            'post_distance': post_distance,
                            'min_distance': min_tack_distance
                        })
        
        return safety_report
    
    def evaluate_strategy_risk(self, route_data: Dict[str, Any], 
                            wind_variability: float = 0.1,
                            tactical_difficulty: str = 'medium') -> Dict[str, Any]:
        """
        戦略のリスク評価
        
        Parameters:
        -----------
        route_data : Dict[str, Any]
            calculate_optimal_route_for_course の返り値
        wind_variability : float
            風の変動性（0-1の範囲、大きいほど変動大）
        tactical_difficulty : str
            戦術的難易度 ('easy', 'medium', 'hard')
            
        Returns:
        --------
        Dict[str, Any]
            リスク評価結果
        """
        risk_assessment = {
            'overall_risk_score': 0.0,  # 0-100の範囲
            'leg_risks': [],
            'risk_factors': {}
        }
        
        # レッグごとのリスク評価
        if 'legs' in route_data:
            leg_risk_scores = []
            
            for i, leg in enumerate(route_data['legs']):
                leg_risk = {}
                
                # 風上/風下によるリスク
                leg_risk['is_upwind'] = leg.get('is_upwind', False)
                
                # 風上走行は一般的にリスクが高い
                upwind_risk = 60 if leg_risk['is_upwind'] else 30
                
                # タック回数によるリスク
                tack_count = leg['path'].get('tack_count', 0)
                tack_risk = min(80, tack_count * 15)  # タック1回につき15ポイント（最大80）
                
                # レッグの長さによるリスク
                leg_distance = leg['path'].get('total_distance', 0)
                distance_risk = min(50, (leg_distance / 1000) * 10)  # 1kmにつき10ポイント（最大50）
                
                # 風の変動性によるリスク
                wind_risk = wind_variability * 100
                
                # 戦術的難易度によるリスク
                difficulty_risk_map = {'easy': 20, 'medium': 50, 'hard': 80}
                difficulty_risk = difficulty_risk_map.get(tactical_difficulty, 50)
                
                # 総合リスクスコア（0-100の範囲）
                leg_risk_score = (upwind_risk * 0.3) + (tack_risk * 0.2) + \
                               (distance_risk * 0.1) + (wind_risk * 0.2) + \
                               (difficulty_risk * 0.2)
                leg_risk_score = min(100, max(0, leg_risk_score))
                
                # リスクレベルの決定
                if leg_risk_score < 30:
                    risk_level = 'low'
                elif leg_risk_score < 60:
                    risk_level = 'medium'
                else:
                    risk_level = 'high'
                
                # レッグリスク情報の設定
                leg_risk.update({
                    'leg_number': i + 1,
                    'risk_score': leg_risk_score,
                    'risk_level': risk_level,
                    'risk_factors': {
                        'upwind_risk': upwind_risk,
                        'tack_risk': tack_risk,
                        'distance_risk': distance_risk,
                        'wind_risk': wind_risk,
                        'difficulty_risk': difficulty_risk
                    }
                })
                
                risk_assessment['leg_risks'].append(leg_risk)
                leg_risk_scores.append(leg_risk_score)
            
            # 全体のリスクスコア（レッグの最大リスクと平均リスクの加重平均）
            if leg_risk_scores:
                max_risk = max(leg_risk_scores)
                avg_risk = sum(leg_risk_scores) / len(leg_risk_scores)
                risk_assessment['overall_risk_score'] = (max_risk * 0.6) + (avg_risk * 0.4)
        
        # 全体のリスク要因
        risk_assessment['risk_factors'] = {
            'wind_variability': wind_variability,
            'tactical_difficulty': tactical_difficulty,
            'total_tack_count': sum(leg['path'].get('tack_count', 0) for leg in route_data.get('legs', [])),
            'total_distance': route_data.get('total_distance', 0),
            'total_time': route_data.get('total_time', 0)
        }
        
        return risk_assessment
    
    # ----- 内部ヘルパーメソッド -----
    
    def _is_upwind_leg(self, start_wp: Dict[str, Any], end_wp: Dict[str, Any]) -> bool:
        """
        レッグが風上かどうかを判定
        
        Parameters:
        -----------
        start_wp, end_wp : Dict[str, Any]
            始点と終点のウェイポイント
            
        Returns:
        --------
        bool
            風上レッグならTrue、風下ならFalse
        """
        if self.wind_field is None:
            return False
        
        # 中間点の風向を取得
        mid_lat = (start_wp['lat'] + end_wp['lat']) / 2
        mid_lon = (start_wp['lon'] + end_wp['lon']) / 2
        
        wind_info = self._get_wind_at_position(mid_lat, mid_lon)
        if wind_info is None:
            return False
        
        # レッグの方位を計算
        leg_bearing = self._calculate_bearing(
            start_wp['lat'], start_wp['lon'], 
            end_wp['lat'], end_wp['lon']
        )
        
        # 風向との相対角度を計算
        relative_angle = abs(self._angle_difference(leg_bearing, wind_info['direction']))
        
        # 90度未満なら風上、90度以上なら風下と判断
        return relative_angle < 90
    
    def _get_optimal_twa(self, boat_type: str, wind_speed: float, 
                       upwind: bool = True) -> Tuple[float, float]:
        """
        指定された風速に対する最適な風向角（最大VMG）を取得
        
        Parameters:
        -----------
        boat_type : str
            艇種の識別子
        wind_speed : float
            風速（ノット）
        upwind : bool
            True: 風上向き、False: 風下向き
            
        Returns:
        --------
        Tuple[float, float]
            (最適風向角, 最大VMG)
        """
        if boat_type not in self.boat_types:
            raise ValueError(f"未知の艇種: {boat_type}")
        
        boat_data = self.boat_types[boat_type]
        optimal_data = boat_data['upwind_optimal'] if upwind else boat_data['downwind_optimal']
        
        # 風速値を丸めて最も近い既知の風速を見つける
        wind_speeds = sorted(optimal_data.keys())
        if not wind_speeds:
            return 45.0 if upwind else 150.0, 0.0  # デフォルト値
        
        if wind_speed <= wind_speeds[0]:
            return optimal_data[wind_speeds[0]]
        
        if wind_speed >= wind_speeds[-1]:
            return optimal_data[wind_speeds[-1]]
        
        # 風速を補間
        for i in range(len(wind_speeds) - 1):
            if wind_speeds[i] <= wind_speed <= wind_speeds[i + 1]:
                low_speed = wind_speeds[i]
                high_speed = wind_speeds[i + 1]
                
                low_angle, low_vmg = optimal_data[low_speed]
                high_angle, high_vmg = optimal_data[high_speed]
                
                # 線形補間
                ratio = (wind_speed - low_speed) / (high_speed - low_speed)
                angle = low_angle + ratio * (high_angle - low_angle)
                vmg = low_vmg + ratio * (high_vmg - low_vmg)
                
                return angle, vmg
        
        # 通常ここには到達しないはず
        return 45.0 if upwind else 150.0, 0.0
    
    def _calculate_optimal_vmg_angles(self, polar_data: pd.DataFrame, 
                                    upwind: bool = True) -> Dict[float, Tuple[float, float]]:
        """
        ポーラーデータから最適VMG角度を計算
        
        Parameters:
        -----------
        polar_data : pd.DataFrame
            ポーラーデータ
        upwind : bool
            True: 風上向き、False: 風下向き
            
        Returns:
        --------
        Dict[float, Tuple[float, float]]
            風速 -> (最適風向角, 最大VMG) の辞書
        """
        result = {}
        
        # 各風速に対して最適角度を計算
        for wind_speed in polar_data.columns:
            try:
                wind_speed_float = float(wind_speed)
            except ValueError:
                continue  # 風速に変換できない列はスキップ
            
            # 角度範囲（風上または風下）
            angle_range = range(0, 91) if upwind else range(91, 181)
            
            max_vmg = 0.0
            optimal_angle = 0.0
            
            for angle in angle_range:
                try:
                    # その角度の艇速を取得
                    boat_speed = self._get_value_from_polar(polar_data, angle, wind_speed_float)
                    
                    # VMGを計算
                    vmg = boat_speed * math.cos(math.radians(angle)) if upwind else boat_speed * math.cos(math.radians(180 - angle))
                    
                    if vmg > max_vmg:
                        max_vmg = vmg
                        optimal_angle = angle
                except:
                    continue
            
            result[wind_speed_float] = (optimal_angle, max_vmg)
        
        return result
    
    def _get_value_from_polar(self, polar_data: pd.DataFrame, angle: float, 
                            wind_speed: float) -> float:
        """
        ポーラーデータから特定の角度と風速に対応する値を取得
        
        Parameters:
        -----------
        polar_data : pd.DataFrame
            ポーラーデータ
        angle : float
            風向角（度）
        wind_speed : float
            風速（ノット）
            
        Returns:
        --------
        float
            艇速（ノット）
        """
        # インデックスに角度が含まれているか確認
        angle_found = False
        for idx in polar_data.index:
            try:
                if float(idx) == angle:
                    angle_found = True
                    break
            except:
                continue
                
        if angle_found:
            # 該当する角度が見つかった場合
            wind_col = str(wind_speed)
            if wind_col in polar_data.columns:
                return float(polar_data.loc[angle, wind_col])
            else:
                # 風速列が見つからない場合は補間
                cols = sorted([float(col) for col in polar_data.columns])
                for i in range(len(cols) - 1):
                    if cols[i] <= wind_speed <= cols[i + 1]:
                        ratio = (wind_speed - cols[i]) / (cols[i + 1] - cols[i])
                        v1 = float(polar_data.loc[angle, str(cols[i])])
                        v2 = float(polar_data.loc[angle, str(cols[i + 1])])
                        return v1 + ratio * (v2 - v1)
                
                # 範囲外の場合は最も近い値を使用
                if wind_speed < cols[0]:
                    return float(polar_data.loc[angle, str(cols[0])])
                else:
                    return float(polar_data.loc[angle, str(cols[-1])])
        
        # 角度も補間する必要がある場合
        angles = sorted([float(ang) for ang in polar_data.index])
        
        # 最も近い2つの角度インデックスを見つける
        ang_lower = None
        ang_upper = None
        
        for i in range(len(angles) - 1):
            if angles[i] <= angle <= angles[i + 1]:
                ang_lower = angles[i]
                ang_upper = angles[i + 1]
                break
        
        # 範囲外の場合
        if ang_lower is None or ang_upper is None:
            if angle < angles[0]:
                ang_lower = ang_upper = angles[0]
            else:
                ang_lower = ang_upper = angles[-1]
        
        # 該当する風速列を見つける
        cols = sorted([float(col) for col in polar_data.columns])
        ws_lower = None
        ws_upper = None
        
        for i in range(len(cols) - 1):
            if cols[i] <= wind_speed <= cols[i + 1]:
                ws_lower = cols[i]
                ws_upper = cols[i + 1]
                break
        
        # 範囲外の場合
        if ws_lower is None or ws_upper is None:
            if wind_speed < cols[0]:
                ws_lower = ws_upper = cols[0]
            else:
                ws_lower = ws_upper = cols[-1]
        
        # 4点の値を取得
        try:
            v00 = float(polar_data.loc[ang_lower, str(ws_lower)])
            v01 = float(polar_data.loc[ang_lower, str(ws_upper)])
            v10 = float(polar_data.loc[ang_upper, str(ws_lower)])
            v11 = float(polar_data.loc[ang_upper, str(ws_upper)])
            
            # 双線形補間
            if ang_lower == ang_upper:
                # 角度の補間不要
                if ws_lower == ws_upper:
                    return v00
                else:
                    ratio = (wind_speed - ws_lower) / (ws_upper - ws_lower)
                    return v00 + ratio * (v01 - v00)
            elif ws_lower == ws_upper:
                # 風速の補間不要
                ratio = (angle - ang_lower) / (ang_upper - ang_lower)
                return v00 + ratio * (v10 - v00)
            else:
                # 双線形補間
                ratio_ang = (angle - ang_lower) / (ang_upper - ang_lower)
                ratio_ws = (wind_speed - ws_lower) / (ws_upper - ws_lower)
                
                v0 = v00 * (1 - ratio_ws) + v01 * ratio_ws
                v1 = v10 * (1 - ratio_ws) + v11 * ratio_ws
                
                return v0 * (1 - ratio_ang) + v1 * ratio_ang
        except:
            # 計算に失敗した場合は最も近い値を返す
            return float(polar_data.loc[angles[0], str(cols[0])])
    
    def _interpolate_boat_speed(self, polar_data: pd.DataFrame, 
                              wind_speed: float, wind_angle: float) -> float:
        """
        ポーラーデータから特定の風速・風向に対する艇速を補間
        
        Parameters:
        -----------
        polar_data : pd.DataFrame
            ポーラーデータ
        wind_speed : float
            風速（ノット）
        wind_angle : float
            風向角（度）
            
        Returns:
        --------
        float
            補間された艇速（ノット）
        """
        try:
            # ポーラーデータの角度と風速を取得
            angles = [float(a) for a in polar_data.index]
            speeds = [float(s) for s in polar_data.columns]
            
            # 風向角が範囲外の場合
            if wind_angle < min(angles):
                wind_angle = min(angles)
            elif wind_angle > max(angles):
                wind_angle = max(angles)
            
            # 風速が範囲外の場合
            if wind_speed < min(speeds):
                wind_speed = min(speeds)
            elif wind_speed > max(speeds):
                wind_speed = max(speeds)
            
            # 補間用の4つの点（2x2グリッド）を見つける
            angle_idx = np.searchsorted(angles, wind_angle)
            if angle_idx == 0:
                angle_idx = 1
            elif angle_idx == len(angles):
                angle_idx = len(angles) - 1
                
            speed_idx = np.searchsorted(speeds, wind_speed)
            if speed_idx == 0:
                speed_idx = 1
            elif speed_idx == len(speeds):
                speed_idx = len(speeds) - 1
            
            # 4点の座標
            angle_low = angles[angle_idx - 1]
            angle_high = angles[angle_idx]
            speed_low = speeds[speed_idx - 1]
            speed_high = speeds[speed_idx]
            
            # 4点の艇速値
            v11 = float(polar_data.loc[angle_low, str(speed_low)])
            v12 = float(polar_data.loc[angle_low, str(speed_high)])
            v21 = float(polar_data.loc[angle_high, str(speed_low)])
            v22 = float(polar_data.loc[angle_high, str(speed_high)])
            
            # 双線形補間
            angle_ratio = (wind_angle - angle_low) / (angle_high - angle_low)
            speed_ratio = (wind_speed - speed_low) / (speed_high - speed_low)
            
            v1 = v11 * (1 - angle_ratio) + v21 * angle_ratio
            v2 = v12 * (1 - angle_ratio) + v22 * angle_ratio
            
            return v1 * (1 - speed_ratio) + v2 * speed_ratio
            
        except Exception as e:
            # エラーが発生した場合は、より堅牢な方法で補間
            return self._get_value_from_polar(polar_data, wind_angle, wind_speed)
    
    def _get_wind_at_position(self, lat: float, lon: float) -> Optional[Dict[str, float]]:
        """
        指定された位置の風向風速を取得
        
        Parameters:
        -----------
        lat, lon : float
            位置の緯度・経度
            
        Returns:
        --------
        Dict[str, float] or None
            風情報（方向、速度、信頼度）
        """
        if self.wind_field is None:
            return None
        
        try:
            # グリッドデータを取得
            lat_grid = self.wind_field['lat_grid']
            lon_grid = self.wind_field['lon_grid']
            wind_directions = self.wind_field['wind_direction']
            wind_speeds = self.wind_field['wind_speed']
            confidence = self.wind_field.get('confidence', np.ones_like(wind_directions) * 0.8)
            
            # グリッド範囲外の場合
            if (lat < np.min(lat_grid) or lat > np.max(lat_grid) or
                lon < np.min(lon_grid) or lon > np.max(lon_grid)):
                return None
            
            # 最も近いグリッドポイントを見つける
            distances = (lat_grid - lat)**2 + (lon_grid - lon)**2
            closest_idx = np.unravel_index(np.argmin(distances), distances.shape)
            
            # そのポイントの風データを取得
            direction = float(wind_directions[closest_idx])
            speed = float(wind_speeds[closest_idx])
            conf = float(confidence[closest_idx])
            
            return {
                'direction': direction,
                'speed': speed,
                'confidence': conf
            }
            
        except Exception as e:
            print(f"風データ取得エラー: {e}")
            return None
    
    def _calculate_bearing(self, lat1: float, lon1: float, 
                         lat2: float, lon2: float) -> float:
        """
        2つの地点間の方位角を計算
        
        Parameters:
        -----------
        lat1, lon1 : float
            始点の緯度・経度
        lat2, lon2 : float
            終点の緯度・経度
            
        Returns:
        --------
        float
            方位角（度）
        """
        # 緯度・経度をラジアンに変換
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # 方位角計算
        dlon = lon2_rad - lon1_rad
        y = math.sin(dlon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        bearing_rad = math.atan2(y, x)
        
        # ラジアンから度に変換し、0-360度の範囲に正規化
        bearing = math.degrees(bearing_rad)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def _normalize_angle(self, angle: float) -> float:
        """角度を0-360度の範囲に正規化（オーバーヘッド削減のため内部実装）"""
        return angle % 360
    
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """2つの角度間の最小差分を計算（-180〜180度の範囲）（オーバーヘッド削減のため内部実装）"""
        return ((angle1 - angle2 + 180) % 360) - 180
    
    def _load_standard_boat_types(self) -> None:
        """
        標準艇種ライブラリを読み込む
        """
        # 標準艇種リスト
        standard_boats = [
            ('laser', 'Laser/ILCA'),
            ('470', '470 Class'),
            ('49er', '49er'),
            ('finn', 'Finn'),
            ('nacra17', 'Nacra 17')
        ]
        
        # モジュールのディレクトリを基準にデータディレクトリを特定
        module_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            # sailing_data_processor パッケージ内の場合
            data_dir = os.path.join(module_dir, 'data', 'polar')
        except:
            # スタンドアロン実行の場合
            data_dir = os.path.join(module_dir, 'sailing_data_processor', 'data', 'polar')
        
        # 各艇種のデータを読み込み
        for boat_id, display_name in standard_boats:
            file_path = os.path.join(data_dir, f"{boat_id}.csv")
            if os.path.exists(file_path):
                success = self.load_polar_data(boat_id, file_path)
                if success:
                    # 表示名を設定
                    if boat_id in self.boat_types:
                        self.boat_types[boat_id]['display_name'] = display_name
            else:
                print(f"警告: 標準艇種データが見つかりません: {file_path}")

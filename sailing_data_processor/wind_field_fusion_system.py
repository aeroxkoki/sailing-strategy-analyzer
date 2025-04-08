"""
sailing_data_processor.wind_field_fusion_system モジュール

複数の艇からの風データを統合し、風の場全体を生成するシステムを提供します。
風向風速の空間的・時間的変化を考慮した風の場の推定・予測を行います。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import math
from scipy.interpolate import griddata, NearestNDInterpolator
import warnings
from functools import lru_cache

# 内部モジュールのインポート
from .wind_field_interpolator import WindFieldInterpolator
from .wind_propagation_model import WindPropagationModel
from .prediction_evaluator import PredictionEvaluator

class WindFieldFusionSystem:
    """
    複数の艇からの風データを統合し、風の場を生成するクラス
    
    機能:
    - 複数艇データの統合
    - 風の場の時空間補間
    - 風の移動モデルを用いた予測
    """
    
    def __init__(self):
        """初期化"""
        # 風データポイントのキャッシュ
        self.wind_data_points = []
        
        # 現在の風の場
        self.current_wind_field = None
        
        # 風の場の履歴
        self.wind_field_history = []
        
        # 最大履歴サイズ
        self.max_history_size = 10
        
        # 補間器
        self.field_interpolator = WindFieldInterpolator()
        
        # 風の移動モデル
        self.propagation_model = WindPropagationModel()
        
        # 予測評価機能
        self.enable_prediction_evaluation = True
        self.prediction_evaluator = PredictionEvaluator()
        
        # 過去の予測履歴（評価用）
        self.previous_predictions = {}
        
        # 最終融合時間
        self.last_fusion_time = None
    
    def add_wind_data_point(self, data_point: Dict[str, Any]):
        """
        風データポイントを追加
        
        Parameters:
        -----------
        data_point : Dict[str, Any]
            風データポイント
            必須キー: 'timestamp', 'latitude', 'longitude', 'wind_direction', 'wind_speed'
        """
        # 必須キーの存在確認
        required_keys = ['timestamp', 'latitude', 'longitude', 'wind_direction', 'wind_speed']
        if not all(key in data_point for key in required_keys):
            warnings.warn("Wind data point missing required keys")
            return
        
        # タイムスタンプがdatetimeでない場合は変換
        if not isinstance(data_point['timestamp'], datetime):
            try:
                data_point['timestamp'] = datetime.fromtimestamp(data_point['timestamp'])
            except:
                warnings.warn("Invalid timestamp format")
                return
        
        # データを追加
        self.wind_data_points.append(data_point)
        
        # データポイントが一定数を超えたら融合処理を実行
        if len(self.wind_data_points) >= 5:
            self.fuse_wind_data()
            
    def update_with_boat_data(self, boats_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        複数の艇データから風の場を更新
        
        Parameters:
        -----------
        boats_data : Dict[str, pd.DataFrame]
            艇IDをキーとする艇データのデータフレームの辞書
            各データフレームは少なくとも以下のカラムを含む必要がある:
            - timestamp: 時間
            - latitude, longitude: 位置
            - wind_direction: 風向（度）
            - wind_speed_knots: 風速（ノット）
            - confidence: 信頼度（オプション）
            
        Returns:
        --------
        Dict[str, Any]
            更新された風の場
        """
        # データポイントをリセット
        self.wind_data_points = []
        
        # 各艇のデータを処理
        for boat_id, boat_df in boats_data.items():
            # データフレームが空の場合はスキップ
            if boat_df.empty:
                continue
                
            # 必要なカラムがあるか確認
            required_columns = ['timestamp', 'latitude', 'longitude', 'wind_direction', 'wind_speed_knots']
            if not all(col in boat_df.columns for col in required_columns):
                warnings.warn(f"Boat {boat_id} data missing required columns")
                continue
            
            # 各行をデータポイントとして追加
            for _, row in boat_df.iterrows():
                # 風速をノットからm/sに変換（1ノット = 0.51444 m/s）
                # 風速カラム名が違う場合に対応
                wind_speed = row.get('wind_speed_knots', 0) * 0.51444
                
                # データポイントを作成
                data_point = {
                    'timestamp': row['timestamp'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'wind_direction': row['wind_direction'],
                    'wind_speed': wind_speed,
                    'boat_id': boat_id
                }
                
                # 信頼度情報があれば追加
                if 'confidence' in row:
                    data_point['confidence'] = row['confidence']
                
                # データポイントを追加
                self.wind_data_points.append(data_point)
        
        # 十分なデータがあれば融合処理を実行
        if self.wind_data_points:
            self.fuse_wind_data()
        
        return self.current_wind_field
    
    def _scale_data_points(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        風データポイントを適切にスケーリングして補間処理を安定化
        
        Parameters:
        -----------
        data_points : List[Dict]
            風データポイントのリスト
            
        Returns:
        --------
        List[Dict]
            スケーリングされたデータポイントのリスト
        """
        if not data_points:
            return []
            
        # 座標データの範囲を取得
        lats = [p['latitude'] for p in data_points]
        lons = [p['longitude'] for p in data_points]
        winds = [p['wind_speed'] for p in data_points]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        min_wind, max_wind = min(winds), max(winds)
        
        # スケーリング係数の計算（範囲が狭すぎる場合に対応）
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        wind_range = max_wind - min_wind
        
        # データのコピーを作成して元のデータを保持
        scaled_data = []
        for point in data_points:
            scaled_point = point.copy()
            scaled_data.append(scaled_point)
        
        # 緯度・経度の範囲が狭すぎる場合は人工的に広げる
        min_range = 0.005  # 約500mの最小範囲に拡大（安全マージンを増やす）
        
        # 必要に応じて範囲を確保し、より広いパディングを追加
        if lat_range < min_range:
            lat_padding = (min_range - lat_range) / 2 + 0.001  # 追加のパディング
            min_lat -= lat_padding
            max_lat += lat_padding
            lat_range = min_range + 0.002  # パディング後の範囲更新
        
        if lon_range < min_range:
            lon_padding = (min_range - lon_range) / 2 + 0.001  # 追加のパディング
            min_lon -= lon_padding
            max_lon += lon_padding
            lon_range = min_range + 0.002  # パディング後の範囲更新
        
        # 風速の最小範囲を確保
        min_wind_range = 1.0  # 最小範囲を1.0に増加
        if wind_range < min_wind_range:
            wind_padding = (min_wind_range - wind_range) / 2 + 0.2  # 追加のパディング
            min_wind = max(0, min_wind - wind_padding)
            max_wind += wind_padding
            wind_range = min_wind_range + 0.4  # パディング後の範囲更新
        
        # 範囲の一貫性確保のために統一スケール係数を使用
        scale_lat = 1.0 / lat_range if lat_range > 0 else 1.0
        scale_lon = 1.0 / lon_range if lon_range > 0 else 1.0
        scale_wind = 1.0 / wind_range if wind_range > 0 else 1.0
        
        # すべてのデータポイントをスケーリングと正規化
        for point in scaled_data:
            # 1. 正規化: 0-1の範囲に変換
            norm_lat = (point['latitude'] - min_lat) * scale_lat
            norm_lon = (point['longitude'] - min_lon) * scale_lon
            norm_wind = (point['wind_speed'] - min_wind) * scale_wind
            
            # 2. 微小なランダムノイズを追加 (QJオプションと同様の効果)
            jitter_lat = 0.002  # 0.2%の緯度ジッター
            jitter_lon = 0.002  # 0.2%の経度ジッター
            jitter_wind = 0.005  # 0.5%の風速ジッター
            
            # スケールに合わせたジッターを追加
            norm_lat += np.random.normal(0, jitter_lat)
            norm_lon += np.random.normal(0, jitter_lon)
            norm_wind += np.random.normal(0, jitter_wind)
            
            # 3. スケーリングした座標を設定
            point['scaled_latitude'] = norm_lat
            point['scaled_longitude'] = norm_lon
            point['scaled_height'] = norm_wind
            
            # 元の値も保持（復元用）
            point['original_latitude'] = point['latitude']
            point['original_longitude'] = point['longitude']
            
            # スケーリングされた値を使用
            point['latitude'] = norm_lat
            point['longitude'] = norm_lon
            point['height'] = norm_wind
            
        return scaled_data
        
    def _restore_original_coordinates(self, scaled_data_points: List[Dict[str, Any]]) -> None:
        """
        スケーリングされたデータポイントの座標を元に戻す
        
        Parameters:
        -----------
        scaled_data_points : List[Dict]
            スケーリングされたデータポイントのリスト
        """
        for point in scaled_data_points:
            if 'original_latitude' in point and 'original_longitude' in point:
                point['latitude'] = point['original_latitude']
                point['longitude'] = point['original_longitude']
    
    def fuse_wind_data(self):
        """
        風データポイントを融合して風の場を生成
        """
        if not self.wind_data_points:
            return
        
        # データポイントを時間順にソート
        sorted_data = sorted(self.wind_data_points, key=lambda x: x['timestamp'])
        
        # 最新のタイムスタンプを取得
        latest_time = sorted_data[-1]['timestamp']
        self.last_fusion_time = latest_time
        
        # 最近のデータポイントのみを使用（30分以内）
        recent_data = []
        for point in sorted_data:
            time_diff = (latest_time - point['timestamp']).total_seconds()
            if time_diff <= 1800:  # 30分 = 1800秒
                recent_data.append(point)
        
        # grid_densityパラメータの設定
        grid_density = 20  # 20x20のグリッド
        
        # 基本的にデータをスケーリング - このステップにより多くのQhull関連エラーを回避
        scaled_data = self._scale_data_points(recent_data)
        
        # field_interpolatorを使用して風の場を生成 - 常にQJオプションを使用
        try:
            # まずidw方式で補間を試みる（最も安定した方法）
            self.current_wind_field = self.field_interpolator.interpolate_wind_field(
                scaled_data, 
                resolution=grid_density, 
                method='idw',  # より安定した逆距離加重法を使用
                qhull_options='QJ'  # Qhull精度エラー回避のためにQJオプションを常に追加
            )
            
            # 風の場のタイムスタンプを設定
            if self.current_wind_field:
                self.current_wind_field['time'] = latest_time
                
                # 履歴に追加
                self.wind_field_history.append({
                    'time': latest_time,
                    'field': self.current_wind_field
                })
                
                # 履歴サイズを制限
                if len(self.wind_field_history) > self.max_history_size:
                    self.wind_field_history.pop(0)
                
                # 予測評価が有効な場合、実測値と予測を比較
                if self.enable_prediction_evaluation:
                    for point in recent_data:
                        self._evaluate_previous_predictions(point['timestamp'], point)
                
                # 元の座標を復元
                self._restore_original_coordinates(scaled_data)
                    
        except Exception as e:
            warnings.warn(f"IDW interpolation failed, trying nearest method: {e}")
            try:
                # IDW方式が失敗した場合はnearest方式を試す (最も頑健だが精度は低い)
                self.current_wind_field = self.field_interpolator.interpolate_wind_field(
                    scaled_data, 
                    resolution=grid_density, 
                    method='nearest',
                    qhull_options='QJ'  # Qhull精度エラー回避のためにQJオプションを追加
                )
                
                # 風の場のタイムスタンプを設定
                if self.current_wind_field:
                    self.current_wind_field['time'] = latest_time
                    
                    # 履歴に追加
                    self.wind_field_history.append({
                        'time': latest_time,
                        'field': self.current_wind_field
                    })
                    
                    # 履歴サイズを制限
                    if len(self.wind_field_history) > self.max_history_size:
                        self.wind_field_history.pop(0)
                    
                    # 予測評価が有効な場合、実測値と予測を比較
                    if self.enable_prediction_evaluation:
                        for point in recent_data:
                            self._evaluate_previous_predictions(point['timestamp'], point)
                    
                    # 元の座標を復元
                    self._restore_original_coordinates(scaled_data)
            except Exception as e2:
                warnings.warn(f"Wind field interpolation retry also failed: {e2}")
                self.current_wind_field = None
        
        # 風の移動モデルを更新
        if len(recent_data) >= self.propagation_model.min_data_points:
            self.propagation_model.estimate_propagation_vector(recent_data)
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        2点間のHaversine距離を計算（メートル）
        
        Parameters:
        -----------
        lat1, lon1 : float
            始点の緯度・経度
        lat2, lon2 : float
            終点の緯度・経度
            
        Returns:
        --------
        float
            距離（メートル）
        """
        # 地球の半径（メートル）
        R = 6371000
        
        # 緯度・経度をラジアンに変換
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # 差分
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine公式
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def _interpolate_field_to_grid(self, source_field: Dict[str, Any], 
                                target_lat_grid: np.ndarray, 
                                target_lon_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        風の場を新しいグリッドに補間
        
        Parameters:
        -----------
        source_field : Dict[str, Any]
            元の風の場
        target_lat_grid : np.ndarray
            対象の緯度グリッド
        target_lon_grid : np.ndarray
            対象の経度グリッド
            
        Returns:
        --------
        Dict[str, Any] or None
            補間された風の場
        """
        try:
            # 元の風の場のグリッド情報を取得
            source_lat_grid = source_field['lat_grid']
            source_lon_grid = source_field['lon_grid']
            source_wind_dirs = source_field['wind_direction']
            source_wind_speeds = source_field['wind_speed']
            source_confidence = source_field['confidence']
            
            # 元のポイントを準備
            points = np.vstack([source_lat_grid.ravel(), source_lon_grid.ravel()]).T
            
            # 対象のポイントを準備
            xi = np.vstack([target_lat_grid.ravel(), target_lon_grid.ravel()]).T
            
            # 風向の補間（循環データなので特別な処理が必要）
            sin_dirs = np.sin(np.radians(source_wind_dirs.ravel()))
            cos_dirs = np.cos(np.radians(source_wind_dirs.ravel()))
            
            interp_sin = griddata(points, sin_dirs, xi, method='linear', fill_value=0)
            interp_cos = griddata(points, cos_dirs, xi, method='linear', fill_value=1)
            
            interp_dirs = np.degrees(np.arctan2(interp_sin, interp_cos)) % 360
            interp_dirs = interp_dirs.reshape(target_lat_grid.shape)
            
            # 風速の補間
            interp_speeds = griddata(points, source_wind_speeds.ravel(), xi, method='linear', fill_value=0)
            interp_speeds = interp_speeds.reshape(target_lat_grid.shape)
            
            # 信頼度の補間
            interp_conf = griddata(points, source_confidence.ravel(), xi, method='linear', fill_value=0.3)
            interp_conf = interp_conf.reshape(target_lat_grid.shape)
            
            # 補間された風の場を返す
            return {
                'lat_grid': target_lat_grid,
                'lon_grid': target_lon_grid,
                'wind_direction': interp_dirs,
                'wind_speed': interp_speeds,
                'confidence': interp_conf,
                'time': source_field.get('time')
            }
            
        except Exception as e:
            warnings.warn(f"Field interpolation failed: {e}")
            return None

    def _evaluate_previous_predictions(self, current_time: datetime, current_wind_data: Dict[str, Any]):
        """
        前回の予測結果と現在の実測値を比較して評価
        
        Parameters:
        -----------
        current_time : datetime
            現在の測定時間
        current_wind_data : Dict[str, Any]
            現在の風データ
        """
        # 前回の予測がない場合はスキップ
        if not self.previous_predictions:
            return
        
        # 不要になった予測を消去（古すぎるもの）
        obsolete_keys = []
        for key, pred_data in self.previous_predictions.items():
            pred_time = pred_data.get('prediction_time')
            target_time = pred_data.get('target_time')
            
            # 2時間以上前の予測は削除
            if pred_time and (current_time - pred_time).total_seconds() > 7200:
                obsolete_keys.append(key)
        
        # 不要なキーを削除
        for key in obsolete_keys:
            del self.previous_predictions[key]
        
        # 位置情報を取得
        if not all(k in current_wind_data for k in ['latitude', 'longitude']):
            return
        
        current_position = (current_wind_data['latitude'], current_wind_data['longitude'])
        
        # 各予測を評価
        for key, pred_data in list(self.previous_predictions.items()):
            pred_time = pred_data.get('prediction_time')
            target_time = pred_data.get('target_time')
            position = pred_data.get('position')
            prediction = pred_data.get('prediction')
            
            # 対象の予測時間と現在時間が近い場合は評価
            # (±1分程度の許容範囲)
            if target_time and abs((current_time - target_time).total_seconds()) < 60:
                # 位置も近い場合のみ評価（200m以内）
                if position and self._haversine_distance(
                    position[0], position[1],
                    current_position[0], current_position[1]
                ) < 200:
                    # 評価実行
                    if prediction:
                        self.prediction_evaluator.evaluate_prediction(
                            predicted=prediction,
                            actual={
                                'wind_direction': current_wind_data['wind_direction'],
                                'wind_speed': current_wind_data['wind_speed']
                            },
                            prediction_time=pred_time,
                            evaluation_time=current_time
                        )
                    
                    # 評価済みの予測を削除
                    del self.previous_predictions[key]
    
    def predict_wind_field(self, target_time: datetime, grid_resolution: int = 20) -> Optional[Dict[str, Any]]:
        """
        目標時間の風の場を予測
        
        Parameters:
        -----------
        target_time : datetime
            予測対象の時間
        grid_resolution : int
            グリッド解像度
            
        Returns:
        --------
        Dict[str, Any] or None
            予測された風の場
        """
        # 現在の風の場が利用可能かチェック
        if not self.current_wind_field:
            return None
            
        # 現在の時間
        current_time = self.last_fusion_time or datetime.now()
        
        # 時間差（秒）
        time_diff_seconds = (target_time - current_time).total_seconds()
        
        # 予測時間が現在に近い場合（5分以内）は補間器を使用
        if abs(time_diff_seconds) <= 300:
            result = self.field_interpolator.interpolate_wind_field(target_time, grid_resolution)
        else:
            # 長期予測の場合は風の移動モデルも活用
            
            # 風の場履歴からデータポイントを収集
            historical_data = []
            
            for history_item in self.wind_field_history:
                history_time = history_item.get('time')
                history_field = history_item.get('field')
                
                if history_time and history_field:
                    # グリッドからサンプリングポイントを抽出
                    lat_grid = history_field['lat_grid']
                    lon_grid = history_field['lon_grid']
                    dir_grid = history_field['wind_direction']
                    speed_grid = history_field['wind_speed']
                    
                    # グリッドサイズ
                    grid_size = lat_grid.shape
                    
                    # 1/4のポイントをサンプリング（計算効率のため）
                    sample_rate = max(1, min(grid_size) // 4)
                    
                    for i in range(0, grid_size[0], sample_rate):
                        for j in range(0, grid_size[1], sample_rate):
                            historical_data.append({
                                'timestamp': history_time,
                                'latitude': lat_grid[i, j],
                                'longitude': lon_grid[i, j],
                                'wind_direction': dir_grid[i, j],
                                'wind_speed': speed_grid[i, j]
                            })
            
            # 現在の風の場のグリッド情報を取得
            current_lat_grid = self.current_wind_field['lat_grid']
            current_lon_grid = self.current_wind_field['lon_grid']
            
            # グリッドサイズを調整（効率のため）
            sample_factor = max(1, grid_resolution // 10)
            pred_lat_grid = current_lat_grid[::sample_factor, ::sample_factor]
            pred_lon_grid = current_lon_grid[::sample_factor, ::sample_factor]
            
            # 各グリッドポイントでの風を予測
            predicted_dirs = np.zeros_like(pred_lat_grid)
            predicted_speeds = np.zeros_like(pred_lat_grid)
            predicted_conf = np.zeros_like(pred_lat_grid)
            
            # 予測評価用にサンプルポイントの予測を保存
            if self.enable_prediction_evaluation:
                # ランダムに5つのポイントを選択
                sample_indices = []
                if pred_lat_grid.size > 0:
                    flat_indices = np.random.choice(
                        pred_lat_grid.size, 
                        min(5, pred_lat_grid.size), 
                        replace=False
                    )
                    rows, cols = np.unravel_index(flat_indices, pred_lat_grid.shape)
                    sample_indices = list(zip(rows, cols))
            
            for i in range(pred_lat_grid.shape[0]):
                for j in range(pred_lat_grid.shape[1]):
                    position = (pred_lat_grid[i, j], pred_lon_grid[i, j])
                    
                    # 風の移動モデルを使用した予測
                    prediction = self.propagation_model.predict_future_wind(
                        position, target_time, historical_data
                    )
                    
                    if prediction:
                        predicted_dirs[i, j] = prediction.get('wind_direction', 0)
                        predicted_speeds[i, j] = prediction.get('wind_speed', 0)
                        predicted_conf[i, j] = prediction.get('confidence', 0.5)
                        
                        # 選択されたサンプルポイントの場合、予測を保存
                        if self.enable_prediction_evaluation and (i, j) in sample_indices:
                            # 一意なキーを生成
                            key = f"{position[0]:.6f}_{position[1]:.6f}_{target_time.timestamp()}"
                            
                            # 予測情報を保存
                            self.previous_predictions[key] = {
                                'prediction_time': current_time,
                                'target_time': target_time,
                                'position': position,
                                'prediction': {
                                    'wind_direction': prediction.get('wind_direction', 0),
                                    'wind_speed': prediction.get('wind_speed', 0),
                                    'confidence': prediction.get('confidence', 0.5)
                                }
                            }
                    else:
                        # 予測が失敗した場合は現在値を使用
                        i_full = i * sample_factor
                        j_full = j * sample_factor
                        
                        if i_full < current_lat_grid.shape[0] and j_full < current_lat_grid.shape[1]:
                            predicted_dirs[i, j] = self.current_wind_field['wind_direction'][i_full, j_full]
                            predicted_speeds[i, j] = self.current_wind_field['wind_speed'][i_full, j_full]
                            predicted_conf[i, j] = self.current_wind_field['confidence'][i_full, j_full] * 0.7  # 信頼度低下
                        else:
                            predicted_dirs[i, j] = 0
                            predicted_speeds[i, j] = 0
                            predicted_conf[i, j] = 0.3
            
            # 予測結果を目標解像度に補間
            if grid_resolution != pred_lat_grid.shape[0]:
                # 新しいグリッドの作成
                lat_min, lat_max = np.min(pred_lat_grid), np.max(pred_lat_grid)
                lon_min, lon_max = np.min(pred_lon_grid), np.max(pred_lon_grid)
                
                new_lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
                new_lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
                new_grid_lats, new_grid_lons = np.meshgrid(new_lat_grid, new_lon_grid)
                
                # 予測結果を新グリッドに補間
                predicted_field = {
                    'lat_grid': pred_lat_grid,
                    'lon_grid': pred_lon_grid,
                    'wind_direction': predicted_dirs,
                    'wind_speed': predicted_speeds,
                    'confidence': predicted_conf,
                    'time': target_time
                }
                
                result = self._interpolate_field_to_grid(
                    predicted_field, new_grid_lats, new_grid_lons
                )
                
                if not result:
                    # 補間に失敗した場合は元のグリッドを使用
                    result = {
                        'lat_grid': pred_lat_grid,
                        'lon_grid': pred_lon_grid,
                        'wind_direction': predicted_dirs,
                        'wind_speed': predicted_speeds,
                        'confidence': predicted_conf,
                        'time': target_time
                    }
            else:
                # 補間なしの場合は直接グリッドを返す
                result = {
                    'lat_grid': pred_lat_grid,
                    'lon_grid': pred_lon_grid,
                    'wind_direction': predicted_dirs,
                    'wind_speed': predicted_speeds,
                    'confidence': predicted_conf,
                    'time': target_time
                }
        
        return result
        
    def get_prediction_quality_report(self) -> Dict[str, Any]:
        """
        予測品質のレポートを取得
        
        Returns:
        --------
        Dict[str, Any]
            予測品質レポート
        """
        # 予測評価機能が無効の場合
        if not self.enable_prediction_evaluation:
            return {
                'status': 'Prediction evaluation is disabled',
                'enable_evaluation': False
            }
        
        # 予測評価器からレポートを取得
        report = self.prediction_evaluator.get_prediction_quality_report()
        
        # 風の移動モデルの情報を追加
        if hasattr(self.propagation_model, 'propagation_vector'):
            report['propagation_model'] = {
                'speed': self.propagation_model.propagation_vector.get('speed', 0),
                'direction': self.propagation_model.propagation_vector.get('direction', 0),
                'confidence': self.propagation_model.propagation_vector.get('confidence', 0.5),
                'wind_speed_factor': self.propagation_model.wind_speed_factor
            }
        
        # 保留中の予測数を追加
        report['pending_predictions'] = len(self.previous_predictions)
        
        return report
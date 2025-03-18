import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.cluster import KMeans
import warnings
from datetime import datetime, timedelta
import math
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree

class RaceWindTracker:
    """
    高精度リアルタイム風向風速推定クラス - レース環境向け
    艇のGPSデータからレースエリア全体の風向風速を高解像度で推定
    """
    
    def __init__(self):
        """初期化"""
        # 艇種ごとの係数を設定
        # 以下は各艇種の極座標データから導出される理想的な比率
        self.boat_coefficients = {
            'default': {'upwind': 3.0, 'downwind': 1.5},
            'laser': {'upwind': 3.2, 'downwind': 1.6},  # レーザー/ILCA - 軽量級一人乗り
            'ilca': {'upwind': 3.2, 'downwind': 1.6},   # ILCA (レーザーの新名称)
            '470': {'upwind': 3.0, 'downwind': 1.5},    # 470級 - ミディアム二人乗り
            '49er': {'upwind': 2.8, 'downwind': 1.3},   # 49er - 高性能スキフ
            'finn': {'upwind': 3.3, 'downwind': 1.7},   # フィン級 - 重量級一人乗り
            'nacra17': {'upwind': 2.5, 'downwind': 1.2}, # Nacra 17 - カタマラン
            'star': {'upwind': 3.4, 'downwind': 1.7}    # スター級 - キールボート
        }
        self.wind_estimates = {}  # 艇ID:風推定データの辞書
        self.min_valid_duration = 60  # 有効な風推定に必要な最小データ期間（秒）
        self.min_valid_points = 20    # 有効な風推定に必要な最小データポイント数
        
        # RaceWindTracker 固有の設定
        self.race_config = {
            'area_size': 2700,           # エリアサイズ (メートル)
            'grid_resolution': 60,        # グリッドの解像度 (60×60)
            'max_boats': 80,             # 最大艇数
            'min_boat_distance': 20,     # 最小艇間距離 (メートル)
            'response_time_ms': 250,     # 目標応答時間 (ミリ秒)
            'use_bayesian': True,        # ベイズ推定を使用
            'use_spatial_correlation': True,  # 空間相関モデルを使用
            'anomaly_detection_sensitivity': 0.8  # 異常値検出感度 (0-1)
        }
        
        # 最新の風の場データ
        self.latest_wind_field = None
        
        # 計算キャッシュ
        self.spatial_distribution_cache = None
        self.quality_weights_cache = None
        self.wind_field_cache = {}  # time: wind_field の辞書

    def _calculate_wind_direction(self, boat_bearing: float, sailing_state: str, 
                                vmg_angle: Optional[float] = None) -> float:
        """
        艇の進行方向と走行状態から風向を計算する中核メソッド
        
        Parameters:
        -----------
        boat_bearing : float
            艇の進行方向（度）
        sailing_state : str
            走行状態 ('upwind', 'downwind', 'reaching')
        vmg_angle : float, optional
            最適VMG角度（度）、指定しない場合はデフォルト値を使用
            
        Returns:
        --------
        float
            風向（0-360度、風が吹いてくる方向）
        """
        # デフォルトVMG角度の設定
        if vmg_angle is None:
            vmg_angle = 45.0  # 標準的な風上VMG角度
        
        if sailing_state == 'upwind':
            # 風上走行時：艇の進行方向 + 180度 - VMG角度 = 風向
            # 例：艇が30度を向いて風上走行している場合、風向は約165度
            return (boat_bearing + 180 - vmg_angle) % 360
        elif sailing_state == 'downwind':
            # 風下走行時：艇の進行方向 = 風向（VMG角度は考慮しない）
            # 例：艇が180度を向いて風下走行している場合、風向は180度
            return boat_bearing
        else:  # reaching
            # リーチング時：中間的な計算
            # 艇の進行方向と風向の角度差は、90度が基本
            return (boat_bearing + 90) % 360

    def _determine_sailing_state(self, bearings: List[float], speeds: List[float]) -> str:
        """
        艇の航路と速度パターンから走行状態を判断
        
        Parameters:
        -----------
        bearings : List[float]
            方位角のリスト
        speeds : List[float]
            速度のリスト
            
        Returns:
        --------
        str
            走行状態 ('upwind', 'downwind', 'reaching')
        """
        if not bearings or not speeds:
            return 'unknown'
        
        # 方位のばらつきを分析（風上はジグザグパターンで標準偏差が大きい）
        bearing_std = np.std(bearings)
        
        # 速度の統計を分析（風上は通常遅い）
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        
        # 2つの主要な方位クラスタを抽出
        if len(bearings) >= 10:
            # 角度データを単位円上の点に変換
            X = np.column_stack([
                np.cos(np.radians(bearings)),
                np.sin(np.radians(bearings))
            ])
            
            # クラスタリング（K-means）
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(2, len(X)), random_state=0).fit(X)
            clusters = kmeans.labels_
            
            # クラスタごとの平均速度を計算
            cluster_speeds = []
            for i in range(max(clusters) + 1):
                cluster_indices = np.where(clusters == i)[0]
                if len(cluster_indices) > 0:
                    cluster_speed = np.mean([speeds[j] for j in cluster_indices])
                    cluster_speeds.append((i, cluster_speed))
            
            # 速度でソート
            cluster_speeds.sort(key=lambda x: x[1])
            
            if len(cluster_speeds) >= 2:
                # 速度の低いクラスタが風上、高いクラスタが風下と判断
                upwind_cluster = cluster_speeds[0][0]
                downwind_cluster = cluster_speeds[-1][0]
                
                # クラスタサイズの判断
                upwind_size = np.sum(clusters == upwind_cluster)
                downwind_size = np.sum(clusters == downwind_cluster)
                
                # クラスタのバランスで判断
                if upwind_size > len(bearings) * 0.7:
                    return 'upwind'
                elif downwind_size > len(bearings) * 0.7:
                    return 'downwind'
                else:
                    # 両方のクラスタが十分な大きさ → 両方のレグを含む
                    return 'mixed'
        
        # より単純な判断基準
        if bearing_std > 40:
            # 方位のばらつきが大きい → 風上タッキング
            return 'upwind'
        elif avg_speed < max_speed * 0.7:
            # 平均速度が最大速度よりかなり遅い → 速度に大きなばらつき（風上の特徴）
            return 'upwind'
        elif avg_speed > max_speed * 0.85:
            # 平均速度が最大速度に近い → 安定した速度（風下の特徴）
            return 'downwind'
        else:
            # 判断が難しい場合
            return 'reaching'

    def _calculate_wind_speed(self, boat_speed: float, sailing_state: str, 
                            boat_type: str = 'default') -> float:
        """
        艇速と走行状態から風速を推定
        
        Parameters:
        -----------
        boat_speed : float
            艇の速度（m/s）
        sailing_state : str
            走行状態 ('upwind', 'downwind', 'reaching')
        boat_type : str
            艇種（速度係数に影響）
            
        Returns:
        --------
        float
            推定風速（m/s）
        """
        # 艇種係数の取得
        coefficients = self.boat_coefficients.get(boat_type.lower(), 
                                                self.boat_coefficients['default'])
        
        if sailing_state == 'upwind':
            # 風上係数
            ratio = coefficients['upwind']
        elif sailing_state == 'downwind':
            # 風下係数
            ratio = coefficients['downwind']
        else:  # reaching
            # リーチング係数（中間値）
            ratio = (coefficients['upwind'] + coefficients['downwind']) / 2
        
        # 風速の計算
        return boat_speed * ratio

    def _calculate_optimal_vmg_angle(self, boat_type: str, is_upwind: bool) -> float:
        """
        艇種と走行状態から最適VMG角度を計算
        
        Parameters:
        -----------
        boat_type : str
            艇種
        is_upwind : bool
            風上走行かどうか
            
        Returns:
        --------
        float
            最適VMG角度（度）
        """
        # 基本VMG角度
        base_vmg_angle = 45.0 if is_upwind else 150.0
        
        # 艇種による調整
        if boat_type.lower() in self.boat_coefficients:
            coef = self.boat_coefficients[boat_type.lower()]
            if is_upwind:
                # 風上係数に基づく調整（小さな調整のみ）
                vmg_angle = base_vmg_angle + (coef['upwind'] - 3.0) * 2.0
                return min(50, max(40, vmg_angle))
            else:
                # 風下係数に基づく調整（小さな調整のみ）
                vmg_angle = base_vmg_angle - (coef['downwind'] - 1.5) * 2.0
                return min(160, max(135, vmg_angle))
        
        return base_vmg_angle

    def _calculate_bearing_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        方位角の変化を計算（循環角度を考慮）
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータフレーム（bearing列が必要）
            
        Returns:
        --------
        pd.DataFrame
            bearing_change列が追加されたデータフレーム
        """
        # データフレームのコピーを作成
        df_result = df.copy()
        
        # 循環角度を考慮した方位角変化を計算
        bearing_change = []
        bearing_change.append(0)  # 最初の点は変化なし
        
        for i in range(1, len(df)):
            prev_bearing = df['bearing'].iloc[i-1]
            curr_bearing = df['bearing'].iloc[i]
            
            # 最小角度差を計算（0-360度の循環を考慮）
            diff = ((curr_bearing - prev_bearing + 180) % 360) - 180
            bearing_change.append(abs(diff))
        
        # 計算した変化を列として追加
        df_result['bearing_change'] = bearing_change
        
        return df_result

    def _detect_tacks_improved(self, df: pd.DataFrame, min_tack_angle: float = 30.0, 
                             window_size: int = 3) -> pd.DataFrame:
        """
        改良されたタック検出アルゴリズム
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータフレーム (bearing_changeを含む)
        min_tack_angle : float
            タックとして検出する最小の方位変化角度
        window_size : int
            タック判定に使用する移動ウィンドウのサイズ
            
        Returns:
        --------
        pd.DataFrame
            検出されたタックポイントのデータフレーム
        """
        if df is None or len(df) < window_size * 2:
            return pd.DataFrame()  # 十分なデータがない場合は空のデータフレームを返す
        
        # コピーを作成
        df_copy = df.copy()
        
        # 移動ウィンドウでの方位変化の計算
        # 単一フレームではなく、複数フレームにわたる変化を考慮
        df_copy['bearing_change_sum'] = df_copy['bearing_change'].rolling(window=window_size, center=True).sum()
        
        # タックの検出（移動ウィンドウ内の累積変化がmin_tack_angleを超える場合）
        df_copy['is_tack'] = df_copy['bearing_change_sum'] > min_tack_angle
        
        # 連続するタックを1つのイベントとしてグループ化
        df_copy['tack_group'] = (df_copy['is_tack'] != df_copy['is_tack'].shift(1)).cumsum()
        
        # タックグループごとに最大の方位変化点を見つける
        tack_points = []
        
        for group_id, group in df_copy[df_copy['is_tack']].groupby('tack_group'):
            if len(group) > 0:
                # グループ内で最大の方位変化がある点を代表点として選択
                max_change_idx = group['bearing_change'].idxmax()
                tack_points.append(df_copy.loc[max_change_idx].copy())
        
        # タックポイントのデータフレームを作成
        if tack_points:
            return pd.DataFrame(tack_points)
        else:
            return pd.DataFrame()

    def _weighted_angle_average(self, angles: List[float], weights: List[float]) -> float:
        """
        重み付き角度平均を計算（角度の循環性を考慮）
        
        Parameters:
        -----------
        angles : List[float]
            角度のリスト（度数法）
        weights : List[float]
            対応する重みのリスト
            
        Returns:
        --------
        float
            重み付き平均角度（0-360度）
        """
        if not angles or not weights or len(angles) != len(weights):
            return 0.0
            
        # 各角度のラジアン変換
        x_sum = 0.0
        y_sum = 0.0
        
        for angle, weight in zip(angles, weights):
            angle_rad = math.radians(angle)
            x_sum += math.cos(angle_rad) * weight
            y_sum += math.sin(angle_rad) * weight
            
        # 平均の計算
        avg_angle_rad = math.atan2(y_sum, x_sum)
        avg_angle_deg = (math.degrees(avg_angle_rad) + 360) % 360
        
        return avg_angle_deg

    def estimate_wind_from_single_boat(self, gps_data: pd.DataFrame, 
                                     min_tack_angle: float = 30.0, 
                                     boat_type: str = 'default', 
                                     use_bayesian: bool = False) -> Optional[pd.DataFrame]:
        """
        単一艇のGPSデータから風向風速を推定
        
        Parameters:
        -----------
        gps_data : pd.DataFrame
            GPSデータ（timestamp, latitude, longitude, speed, bearing等を含む）
        min_tack_angle : float
            タックと認識する最小の方向転換角度
        boat_type : str
            艇種識別子（速度係数に影響）
        use_bayesian : bool
            ベイズ推定を使用するかどうか
                
        Returns:
        --------
        pd.DataFrame or None
            推定された風向風速情報を含むDataFrame、推定失敗時はNone
        """
        # 1. データの検証
        if gps_data is None or len(gps_data) < self.min_valid_points:
            warnings.warn(f"有効なデータが不足しています")
            return None
        
        # 2. 必要列の存在チェック
        required_cols = ['bearing', 'speed']
        if not all(col in gps_data.columns for col in required_cols):
            warnings.warn(f"必要な列がありません")
            return None
        
        # 3. 方位変化の計算
        df = self._calculate_bearing_change(gps_data.copy())
        
        # 4. タック検出
        tack_points = self._detect_tacks_improved(df, min_tack_angle)
        
        # 5. 走行状態の判定
        sailing_state = self._determine_sailing_state(df['bearing'].tolist(), 
                                                  df['speed'].tolist())
        
        # 6. 最適VMG角度の計算
        vmg_angle = self._calculate_optimal_vmg_angle(
            boat_type, sailing_state == 'upwind')
        
        # 7. 風向と風速の計算
        avg_bearing = df['bearing'].mean()
        avg_speed = df['speed'].mean()
        
        # 風向の計算
        wind_direction = self._calculate_wind_direction(
            avg_bearing, sailing_state, vmg_angle)
        
        # 風速の計算
        wind_speed = self._calculate_wind_speed(
            avg_speed, sailing_state, boat_type)
        
        # 8. ウィンドウベースの時系列推定
        window_size = max(len(df) // 10, 20)
        wind_estimates = []
        
        for i in range(0, len(df), window_size//2):
            end_idx = min(i + window_size, len(df))
            if end_idx - i < window_size // 2:
                continue
                
            window_data = df.iloc[i:end_idx]
            
            window_bearing = window_data['bearing'].mean()
            window_speed = window_data['speed'].mean()
            window_state = self._determine_sailing_state(
                window_data['bearing'].tolist(), window_data['speed'].tolist())
            
            window_direction = self._calculate_wind_direction(
                window_bearing, window_state, vmg_angle)
            window_speed_val = self._calculate_wind_speed(
                window_speed, window_state, boat_type)
            
            center_time = window_data['timestamp'].iloc[len(window_data)//2] \
                if 'timestamp' in window_data.columns else None
            center_lat = window_data['latitude'].mean() \
                if 'latitude' in window_data.columns else None
            center_lon = window_data['longitude'].mean() \
                if 'longitude' in window_data.columns else None
            
            confidence = 0.7  # 基本信頼度
            
            # ベイズ推定を使用する場合の補正
            if use_bayesian and wind_estimates:
                prior = wind_estimates[-1]
                alpha = 0.3
                
                # ベイズ補正
                window_direction = self._weighted_angle_average(
                    [prior['wind_direction'], window_direction],
                    [prior['confidence'], confidence * alpha]
                )
                window_speed_val = prior['wind_speed_knots'] * 0.7 + window_speed_val * 0.3
                confidence = min(0.9, confidence + 0.1)
            
            wind_estimate = {
                'timestamp': center_time,
                'latitude': center_lat,
                'longitude': center_lon,
                'wind_direction': window_direction,
                'wind_speed_knots': window_speed_val * 1.94384,  # m/s → ノット
                'confidence': confidence,
                'boat_id': df['boat_id'].iloc[0] if 'boat_id' in df.columns else 'Unknown',
                'sailing_state': window_state
            }
            
            wind_estimates.append(wind_estimate)
        
        if wind_estimates:
            return pd.DataFrame(wind_estimates)
        else:
            return None

    def estimate_wind_from_multiple_boats(self, boats_data: Dict[str, pd.DataFrame], 
                                        boat_types: Dict[str, str] = None, 
                                        boat_weights: Dict[str, float] = None) -> Dict[str, pd.DataFrame]:
        """
        複数艇のGPSデータから風向風速を推定
        
        Parameters:
        -----------
        boats_data : Dict[str, pd.DataFrame]
            艇ID:GPSデータの辞書
        boat_types : Dict[str, str], optional
            艇ID:艇種の辞書
        boat_weights : Dict[str, float], optional
            艇ID:重み係数の辞書（技術レベルに基づく重み付け）
                
        Returns:
        --------
        Dict[str, pd.DataFrame]
            艇ID:風推定データの辞書
        """
        if not boats_data:
            return {}
                
        # デフォルト値の設定
        if boat_types is None:
            boat_types = {boat_id: 'default' for boat_id in boats_data.keys()}
                
        if boat_weights is None:
            boat_weights = {boat_id: 1.0 for boat_id in boats_data.keys()}
        
        # 各艇の風向風速を個別に推定
        for boat_id, gps_data in boats_data.items():
            boat_type = boat_types.get(boat_id, 'default')
            
            if boat_id not in self.wind_estimates:
                wind_estimate = self.estimate_wind_from_single_boat(
                    gps_data=gps_data,
                    min_tack_angle=30.0,
                    boat_type=boat_type,
                    use_bayesian=True
                )
                
                if wind_estimate is not None:
                    self.wind_estimates[boat_id] = wind_estimate
        
        return self.wind_estimates

    def _calculate_spatial_distribution(self, boat_positions: List[Tuple[float, float]]) -> float:
        """
        艇の位置分布の均一性を評価
        
        Parameters:
        -----------
        boat_positions : List[Tuple[float, float]]
            艇の位置のリスト [(lat1, lon1), (lat2, lon2), ...]
            
        Returns:
        --------
        float
            空間分布スコア (0-1、1が最も均一)
        """
        if len(boat_positions) <= 1:
            return 0.5  # デフォルト値
        
        # 位置を numpy 配列に変換
        positions = np.array(boat_positions)
        
        # 艇間の距離行列を計算
        n_boats = len(positions)
        distances = np.zeros((n_boats, n_boats))
        
        for i in range(n_boats):
            for j in range(i+1, n_boats):
                # 緯度・経度を距離（メートル）に変換
                lat1, lon1 = positions[i]
                lat2, lon2 = positions[j]
                
                # 緯度1度 ≈ 111km, 経度1度 = 111km * cos(lat)
                lat_dist = (lat2 - lat1) * 111000
                lon_dist = (lon2 - lon1) * 111000 * math.cos(math.radians((lat1 + lat2) / 2))
                
                dist = math.sqrt(lat_dist**2 + lon_dist**2)
                distances[i, j] = distances[j, i] = dist
        
        # 最小距離のヒストグラム分析
        min_distances = []
        for i in range(n_boats):
            dists = distances[i]
            dists = dists[dists > 0]  # 自分自身を除外
            if len(dists) > 0:
                min_distances.append(np.min(dists))
        
        if not min_distances:
            return 0.5
            
        # 最小距離の標準偏差を計算（小さいほど均一）
        std_min_dist = np.std(min_distances)
        mean_min_dist = np.mean(min_distances)
        
        # 変動係数（標準偏差/平均）を計算
        if mean_min_dist > 0:
            cv = std_min_dist / mean_min_dist
        else:
            cv = 1.0
            
        # スコアに変換 (0-1, 1が最も均一)
        score = max(0, min(1, 1 - cv))
        
        return score

    def _calculate_boat_data_quality(self, boat_data: pd.DataFrame) -> float:
        """
        個々の艇のデータ品質を評価
        
        Parameters:
        -----------
        boat_data : pd.DataFrame
            艇のGPSデータ
            
        Returns:
        --------
        float
            データ品質スコア (0-1、1が最高品質)
        """
        if boat_data is None or len(boat_data) < self.min_valid_points:
            return 0.0
            
        quality_score = 0.5  # デフォルト値
        
        # 1. データ密度の評価
        if 'timestamp' in boat_data.columns:
            # 時間的密度を評価
            timestamps = boat_data['timestamp']
            if len(timestamps) >= 2:
                time_diffs = np.diff(timestamps.astype(np.int64)) / 10**9  # 秒単位に変換
                avg_time_diff = np.mean(time_diffs)
                
                # 理想的なサンプリングレート（1秒あたり）
                ideal_rate = 1.0
                actual_rate = 1.0 / max(0.1, avg_time_diff)  # ゼロ除算を回避
                
                # レート比（理想/実際 or 実際/理想のうち小さい方）
                rate_ratio = min(ideal_rate / actual_rate, actual_rate / ideal_rate)
                
                # データ密度スコア (0-0.3)
                density_score = 0.3 * rate_ratio
            else:
                density_score = 0.0
        else:
            density_score = 0.1  # タイムスタンプがない場合は低めのスコア
        
        # 2. データの一貫性評価
        consistency_score = 0.0
        
        if 'speed' in boat_data.columns and len(boat_data) >= 3:
            # 速度の一貫性
            speeds = boat_data['speed']
            speed_diffs = np.abs(np.diff(speeds))
            
            # 異常な速度変化の検出
            anomaly_threshold = np.mean(speeds) * 0.5  # 平均速度の50%を閾値とする
            anomaly_ratio = np.sum(speed_diffs > anomaly_threshold) / len(speed_diffs)
            
            # スコア計算 (0-0.3)
            consistency_score += 0.3 * (1 - min(1, anomaly_ratio * 5))
        
        if 'bearing' in boat_data.columns and len(boat_data) >= 3:
            # 方位の一貫性
            bearings = boat_data['bearing']
            
            # 角度差の計算（循環を考慮）
            bearing_diffs = []
            for i in range(1, len(bearings)):
                diff = abs((bearings.iloc[i] - bearings.iloc[i-1] + 180) % 360 - 180)
                bearing_diffs.append(diff)
            
            # 異常な方位変化の検出
            anomaly_threshold = 30.0  # 30度を閾値とする
            anomaly_ratio = np.sum(np.array(bearing_diffs) > anomaly_threshold) / len(bearing_diffs)
            
            # スコア計算 (0-0.2)
            consistency_score += 0.2 * (1 - min(1, anomaly_ratio * 3))
        
        # 3. データの完全性評価
        completeness_score = 0.0
        
        # 必要なカラムの存在をチェック
        required_cols = ['latitude', 'longitude', 'speed', 'bearing']
        existing_cols = [col for col in required_cols if col in boat_data.columns]
        
        # スコア計算 (0-0.2)
        completeness_score = 0.2 * (len(existing_cols) / len(required_cols))
        
        # 総合スコア
        quality_score = density_score + consistency_score + completeness_score
        
        return quality_score

    def _detect_anomalies_realtime(self, wind_data: pd.DataFrame, 
                                 sensitivity: float = 0.8) -> pd.DataFrame:
        """
        風向風速データの異常値をリアルタイムで検出

        Parameters:
        -----------
        wind_data : pd.DataFrame
            風向風速データ
        sensitivity : float
            検出感度 (0-1、高いほど敏感)

        Returns:
        --------
        pd.DataFrame
            異常値フラグが追加されたデータフレーム
        """
        if wind_data is None or len(wind_data) < 3:
            return wind_data

        df = wind_data.copy()
        
        # 風向の異常検出
        if 'wind_direction' in df.columns:
            # 移動平均と標準偏差を計算
            window_size = max(3, int(len(df) * 0.1))  # データの10%か最低3ポイント
            
            # 風向の循環性を考慮した処理
            sin_vals = np.sin(np.radians(df['wind_direction']))
            cos_vals = np.cos(np.radians(df['wind_direction']))
            
            # 移動平均
            sin_ma = sin_vals.rolling(window=window_size, center=True).mean()
            cos_ma = cos_vals.rolling(window=window_size, center=True).mean()
            
            # 風向の移動平均を計算
            wind_dir_ma = np.degrees(np.arctan2(sin_ma, cos_ma)) % 360
            
            # 差異の計算
            angle_diffs = []
            for i in range(len(df)):
                # 実際の値と移動平均の差
                diff = abs((df['wind_direction'].iloc[i] - wind_dir_ma.iloc[i] + 180) % 360 - 180)
                angle_diffs.append(diff)
            
            # 閾値を計算（感度に基づく）
            threshold = (30 - 5 * sensitivity) + np.std(angle_diffs) * (3 - 2 * sensitivity)
            
            # 異常値フラグを設定
            df['wind_dir_anomaly'] = np.array(angle_diffs) > threshold
        
        # 風速の異常検出
        if 'wind_speed_knots' in df.columns:
            # 移動平均と標準偏差を計算
            window_size = max(3, int(len(df) * 0.1))
            wind_speed_ma = df['wind_speed_knots'].rolling(window=window_size, center=True).mean()
            wind_speed_std = df['wind_speed_knots'].rolling(window=window_size, center=True).std().fillna(0)
            
            # 差異の計算
            speed_diffs = abs(df['wind_speed_knots'] - wind_speed_ma)
            
            # 閾値を計算（感度に基づく）
            threshold = (2 - 1 * sensitivity) + wind_speed_std * (3 - 2 * sensitivity)
            
            # 異常値フラグを設定
            df['wind_speed_anomaly'] = speed_diffs > threshold
        
        # 総合異常値フラグ
        if 'wind_dir_anomaly' in df.columns and 'wind_speed_anomaly' in df.columns:
            df['is_anomaly'] = df['wind_dir_anomaly'] | df['wind_speed_anomaly']
        elif 'wind_dir_anomaly' in df.columns:
            df['is_anomaly'] = df['wind_dir_anomaly']
        elif 'wind_speed_anomaly' in df.columns:
            df['is_anomaly'] = df['wind_speed_anomaly']
        else:
            df['is_anomaly'] = False
            
        return df

    def estimate_high_resolution_wind_field(self, time_point: datetime) -> Dict[str, Any]:
        """
        高解像度の風向風速場を推定（レース仕様：60x60グリッド）

        Parameters:
        -----------
        time_point : datetime
            風の場を推定する時点

        Returns:
        --------
        Dict[str, Any]
            高解像度の風向風速場データ
        """
        # キャッシュの確認
        if time_point in self.wind_field_cache:
            return self.wind_field_cache[time_point]

        # 風の場の解像度
        grid_resolution = self.race_config['grid_resolution']
        
        # 風推定データがない場合は推定不能
        if not self.wind_estimates:
            return None
            
        # 各艇からのデータを時間でフィルタリング
        nearby_time_data = []
        
        for boat_id, wind_data in self.wind_estimates.items():
            if 'timestamp' not in wind_data.columns:
                continue
                    
            # 指定時間の前後30秒のデータを抽出
            time_diff = abs((wind_data['timestamp'] - time_point).dt.total_seconds())
            time_mask = time_diff < 30
            if time_mask.any():
                boat_time_data = wind_data[time_mask].copy()
                
                # 異常値検出と除外
                boat_time_data = self._detect_anomalies_realtime(
                    boat_time_data, 
                    self.race_config['anomaly_detection_sensitivity']
                )
                boat_time_data = boat_time_data[~boat_time_data['is_anomaly']]
                
                if not boat_time_data.empty:
                    nearby_time_data.append(boat_time_data)
        
        # データを統合
        if not nearby_time_data:
            return None
        
        combined_data = pd.concat(nearby_time_data)
        
        # 艇位置の空間分布評価
        boat_positions = list(zip(combined_data['latitude'], combined_data['longitude']))
        
        # 空間分布スコアを計算（キャッシュを使用）
        if self.spatial_distribution_cache is None:
            spatial_distribution_score = self._calculate_spatial_distribution(boat_positions)
            self.spatial_distribution_cache = (boat_positions, spatial_distribution_score)
        else:
            cached_positions, cached_score = self.spatial_distribution_cache
            
            # 位置が大きく変わった場合は再計算
            if len(boat_positions) != len(cached_positions):
                spatial_distribution_score = self._calculate_spatial_distribution(boat_positions)
                self.spatial_distribution_cache = (boat_positions, spatial_distribution_score)
            else:
                # ポジション差の計算
                position_diff = 0
                for i, (p1, p2) in enumerate(zip(boat_positions, cached_positions)):
                    lat_diff = (p1[0] - p2[0]) * 111000  # メートル単位
                    lon_diff = (p1[1] - p2[1]) * 111000 * math.cos(math.radians(p1[0]))
                    position_diff += math.sqrt(lat_diff**2 + lon_diff**2)
                
                avg_position_diff = position_diff / len(boat_positions)
                
                if avg_position_diff > 50:  # 平均50m以上移動していたら再計算
                    spatial_distribution_score = self._calculate_spatial_distribution(boat_positions)
                    self.spatial_distribution_cache = (boat_positions, spatial_distribution_score)
                else:
                    spatial_distribution_score = cached_score

        # データ品質の評価
        quality_weights = {}
        for boat_id in set(combined_data['boat_id']):
            boat_data = combined_data[combined_data['boat_id'] == boat_id]
            quality_weights[boat_id] = self._calculate_boat_data_quality(boat_data)
        
        # 境界を設定
        min_lat = combined_data['latitude'].min()
        max_lat = combined_data['latitude'].max()
        min_lon = combined_data['longitude'].min()
        max_lon = combined_data['longitude'].max()
        
        # 少し余裕を持たせる
        lat_margin = (max_lat - min_lat) * 0.1
        lon_margin = (max_lon - min_lon) * 0.1
        min_lat -= lat_margin
        max_lat += lat_margin
        min_lon -= lon_margin
        max_lon += lon_margin
        
        # グリッドの作成（レース仕様：60x60）
        lat_grid = np.linspace(min_lat, max_lat, grid_resolution)
        lon_grid = np.linspace(min_lon, max_lon, grid_resolution)
        grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
        
        # 風向風速の推定値を格納するグリッド
        grid_u = np.zeros_like(grid_lats)  # 東西風成分
        grid_v = np.zeros_like(grid_lons)  # 南北風成分
        grid_speeds = np.zeros_like(grid_lats)
        grid_weights = np.zeros_like(grid_lats)
        
        # 空間相関モデルパラメータ
        if self.race_config['use_spatial_correlation']:
            # KDTreeを使用した効率的な空間検索
            points = np.column_stack([combined_data['latitude'], combined_data['longitude']])
            tree = cKDTree(points)
            
            # 相関距離（メートル）
            correlation_distance = 300.0  # 300mの相関距離
            correlation_distance_deg = correlation_distance / 111000  # 度単位に変換
            
            # 各グリッドポイントに対する計算
            for i in range(grid_resolution):
                for j in range(grid_resolution):
                    query_point = np.array([grid_lats[i, j], grid_lons[i, j]])
                    
                    # 相関距離内のポイントを検索
                    indices = tree.query_ball_point(query_point, correlation_distance_deg)
                    
                    if not indices:
                        continue
                    
                    # 各データポイントの寄与を計算
                    for idx in indices:
                        row = combined_data.iloc[idx]
                        
                        # 距離の計算
                        lat_dist = (query_point[0] - row['latitude']) * 111000
                        lon_dist = (query_point[1] - row['longitude']) * 111000 * math.cos(math.radians(row['latitude']))
                        dist = math.sqrt(lat_dist**2 + lon_dist**2)
                        
                        # 空間相関重み（ガウスカーネル）
                        spatial_weight = math.exp(-(dist / correlation_distance)**2)
                        
                        # データ品質による重み
                        quality_weight = quality_weights.get(row['boat_id'], 0.5)
                        
                        # 空間分布評価を考慮した重み
                        distribution_factor = 0.5 + 0.5 * spatial_distribution_score
                        
                        # 信頼度を反映
                        confidence = row.get('confidence', 0.7)
                        
                        # 総合重み
                        weight = spatial_weight * quality_weight * confidence * distribution_factor
                        
                        # 風向を単位ベクトルに分解
                        dir_rad = math.radians(row['wind_direction'])
                        u = math.sin(dir_rad)  # 東西成分
                        v = math.cos(dir_rad)  # 南北成分
                        
                        # 重み付きでベクトル成分と風速を足し合わせる
                        grid_weights[i, j] += weight
                        grid_u[i, j] += weight * u
                        grid_v[i, j] += weight * v
                        grid_speeds[i, j] += weight * row['wind_speed_knots']
        else:
            # 従来の逆距離加重法（空間相関なし）
            for _, row in combined_data.iterrows():
                # 風向を単位ベクトルに分解
                dir_rad = math.radians(row['wind_direction'])
                u = math.sin(dir_rad)  # 東西成分
                v = math.cos(dir_rad)  # 南北成分
                
                # データ品質による重み
                quality_weight = quality_weights.get(row['boat_id'], 0.5)
                
                # 空間分布評価を考慮した重み
                distribution_factor = 0.5 + 0.5 * spatial_distribution_score
                
                # 各グリッドポイントへの寄与を計算
                for i in range(grid_resolution):
                    for j in range(grid_resolution):
                        # 空間距離（メートル）を計算
                        lat_dist = (grid_lats[i, j] - row['latitude']) * 111000
                        lon_dist = (grid_lons[i, j] - row['longitude']) * 111000 * math.cos(math.radians(row['latitude']))
                        dist = math.sqrt(lat_dist**2 + lon_dist**2)
                        
                        # 距離による重み（逆二乗加重）
                        if dist < 10:  # 非常に近い点
                            spatial_weight = 1.0
                        else:
                            spatial_weight = (500.0 / (dist + 500.0))**2
                        
                        # 信頼度を反映
                        confidence = row.get('confidence', 0.7)
                        
                        # 総合重み
                        weight = spatial_weight * quality_weight * confidence * distribution_factor
                        
                        # 重み付きでベクトル成分と風速を足し合わせる
                        grid_weights[i, j] += weight
                        grid_u[i, j] += weight * u
                        grid_v[i, j] += weight * v
                        grid_speeds[i, j] += weight * row['wind_speed_knots']
        
        # 重みで正規化して最終的な風向風速を計算
        mask = grid_weights > 0
        
        # 風向の計算（ベクトル成分から）
        u_normalized = np.zeros_like(grid_u)
        v_normalized = np.zeros_like(grid_v)
        u_normalized[mask] = grid_u[mask] / grid_weights[mask]
        v_normalized[mask] = grid_v[mask] / grid_weights[mask]
        
        # 風向角度に変換
        wind_directions = np.zeros_like(grid_lats)
        wind_directions[mask] = np.degrees(np.arctan2(u_normalized[mask], v_normalized[mask])) % 360
        
        # 風速の計算
        wind_speeds = np.zeros_like(grid_speeds)
        wind_speeds[mask] = grid_speeds[mask] / grid_weights[mask]
        
        # 信頼度の正規化
        confidence = np.zeros_like(grid_weights)
        if grid_weights.max() > 0:
            confidence = grid_weights / grid_weights.max()
        
        # 結果辞書
        wind_field = {
            'lat_grid': grid_lats,
            'lon_grid': grid_lons,
            'wind_direction': wind_directions,
            'wind_speed': wind_speeds,
            'confidence': confidence,
            'time': time_point,
            'spatial_distribution_score': spatial_distribution_score,
            'data_quality': {
                'avg_quality': np.mean(list(quality_weights.values())) if quality_weights else 0,
                'boat_qualities': quality_weights
            }
        }
        
        # 結果をキャッシュに保存
        self.wind_field_cache[time_point] = wind_field
        self.latest_wind_field = wind_field
        
        # キャッシュサイズを制限
        if len(self.wind_field_cache) > 20:
            oldest_time = min(self.wind_field_cache.keys())
            del self.wind_field_cache[oldest_time]
        
        return wind_field

    def optimize_wind_estimation(self, race_boats: List[Any]) -> Dict[str, Any]:
        """
        RaceWindTracker 提案仕様で風向風速を推定
        
        Parameters:
        -----------
        race_boats : List[Any]
            レース艇データオブジェクトのリスト
            
        Returns:
        --------
        Dict[str, Any]
            高解像度風向風速フィールド
        """
        # 艇の位置情報から風向風速を推定
        boat_positions = [boat.get_position() for boat in race_boats if hasattr(boat, 'get_position')]
        
        # 位置情報のないボートはスキップ
        if not boat_positions:
            return self.latest_wind_field  # 最新の風の場を返す
        
        # 空間分布の均一性評価
        spatial_distribution_score = self._calculate_spatial_distribution(boat_positions)
        
        # データ品質の重み付け
        data_quality_weights = {}
        for boat in race_boats:
            if hasattr(boat, 'get_data'):
                boat_data = boat.get_data()
                boat_id = getattr(boat, 'id', f"boat_{id(boat)}")
                data_quality_weights[boat_id] = self._calculate_boat_data_quality(boat_data)
            
        # 風向風速データを収集
        boats_data = {}
        boat_types = {}
        
        for boat in race_boats:
            if hasattr(boat, 'get_data'):
                boat_id = getattr(boat, 'id', f"boat_{id(boat)}")
                boats_data[boat_id] = boat.get_data()
                
                # 艇種情報があれば収集
                if hasattr(boat, 'boat_type'):
                    boat_types[boat_id] = boat.boat_type
        
        # 風向風速推定
        self.estimate_wind_from_multiple_boats(
            boats_data=boats_data,
            boat_types=boat_types,
            boat_weights=data_quality_weights
        )
        
        # 現在時刻
        current_time = datetime.now()
        
        # 高解像度風の場を推定
        wind_field = self.estimate_high_resolution_wind_field(
            time_point=current_time
        )
        
        return wind_field

    def estimate_wind_field(self, time_point: datetime, grid_resolution: int = None) -> Optional[Dict[str, Any]]:
        """
        特定時点での風の場を推定
        
        Parameters:
        -----------
        time_point : datetime
            風推定を行いたい時点
        grid_resolution : int, optional
            出力グリッドの解像度（指定しない場合はレース設定を使用）
                
        Returns:
        --------
        Dict[str, Any] or None
            緯度・経度グリッドと推定された風向風速データ
        """
        # グリッド解像度の決定
        resolution = grid_resolution if grid_resolution is not None else self.race_config['grid_resolution']
        
        # レース仕様で高解像度の風の場を推定
        if resolution == self.race_config['grid_resolution']:
            return self.estimate_high_resolution_wind_field(time_point)
            
        # その他の解像度の場合は元の実装を使用
        if not self.wind_estimates:
            return None
                
        # 各艇からのデータを時間でフィルタリング
        nearby_time_data = []
        
        for boat_id, wind_data in self.wind_estimates.items():
            if 'timestamp' not in wind_data.columns:
                continue
                    
            # 指定時間の前後30秒のデータを抽出
            time_diff = abs((wind_data['timestamp'] - time_point).dt.total_seconds())
            time_mask = time_diff < 30
            if time_mask.any():
                boat_time_data = wind_data[time_mask].copy()
                nearby_time_data.append(boat_time_data)
        
        # データを統合
        if not nearby_time_data:
            return None
        
        combined_data = pd.concat(nearby_time_data)
        
        # 境界を設定
        min_lat = combined_data['latitude'].min()
        max_lat = combined_data['latitude'].max()
        min_lon = combined_data['longitude'].min()
        max_lon = combined_data['longitude'].max()
        
        # 少し余裕を持たせる
        lat_margin = (max_lat - min_lat) * 0.1
        lon_margin = (max_lon - min_lon) * 0.1
        min_lat -= lat_margin
        max_lat += lat_margin
        min_lon -= lon_margin
        max_lon += lon_margin
        
        # グリッドの作成
        lat_grid = np.linspace(min_lat, max_lat, resolution)
        lon_grid = np.linspace(min_lon, max_lon, resolution)
        grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
        
        # 風向風速の推定値を格納するグリッド
        grid_u = np.zeros_like(grid_lats)  # 東西風成分
        grid_v = np.zeros_like(grid_lons)  # 南北風成分
        grid_speeds = np.zeros_like(grid_lats)
        grid_weights = np.zeros_like(grid_lats)
        
        # 各データポイントからグリッドへの寄与を計算
        for _, row in combined_data.iterrows():
            # 風向を単位ベクトルに分解
            dir_rad = np.radians(row['wind_direction'])
            u = np.sin(dir_rad)  # 東西成分
            v = np.cos(dir_rad)  # 南北成分
            
            # 各グリッドポイントへの寄与を計算
            for i in range(resolution):
                for j in range(resolution):
                    # 空間距離（メートル）を計算
                    lat_dist = (grid_lats[i, j] - row['latitude']) * 111000  # 緯度1度 ≈ 111km
                    lon_dist = (grid_lons[i, j] - row['longitude']) * 111000 * math.cos(math.radians(row['latitude']))
                    dist = math.sqrt(lat_dist**2 + lon_dist**2)
                    
                    # 距離による重み（逆二乗加重）
                    if dist < 10:  # 非常に近い点
                        weight = row['confidence'] * 1.0
                    else:
                        weight = row['confidence'] * (500.0 / (dist + 500.0))**2
                    
                    # 重み付きでベクトル成分と風速を足し合わせる
                    grid_weights[i, j] += weight
                    grid_u[i, j] += weight * u
                    grid_v[i, j] += weight * v
                    grid_speeds[i, j] += weight * row['wind_speed_knots']
        
        # 重みで正規化して最終的な風向風速を計算
        mask = grid_weights > 0
        
        # 風向の計算（ベクトル成分から）
        u_normalized = np.zeros_like(grid_u)
        v_normalized = np.zeros_like(grid_v)
        u_normalized[mask] = grid_u[mask] / grid_weights[mask]
        v_normalized[mask] = grid_v[mask] / grid_weights[mask]
        
        # 風向角度に変換
        wind_directions = np.zeros_like(grid_lats)
        wind_directions[mask] = np.degrees(np.arctan2(u_normalized[mask], v_normalized[mask])) % 360
        
        # 風速の計算
        wind_speeds = np.zeros_like(grid_speeds)
        wind_speeds[mask] = grid_speeds[mask] / grid_weights[mask]
        
        # 信頼度の正規化
        confidence = np.zeros_like(grid_weights)
        if grid_weights.max() > 0:
            confidence = grid_weights / grid_weights.max()
        
        return {
            'lat_grid': grid_lats,
            'lon_grid': grid_lons,
            'wind_direction': wind_directions,
            'wind_speed': wind_speeds,
            'confidence': confidence,
            'time': time_point
        }

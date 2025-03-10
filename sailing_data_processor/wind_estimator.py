import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.cluster import KMeans
import warnings
from datetime import datetime, timedelta
import math

class WindEstimator:
    """風向風速推定クラス - GPSデータから風向風速を推定する機能を提供"""
    
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
        # データが有効かチェック
        if gps_data is None or len(gps_data) < self.min_valid_points:
            warnings.warn(f"有効なデータが不足しています。最低{self.min_valid_points}ポイント必要です。")
            return None
        
        # データ期間の確認
        if 'timestamp' in gps_data.columns:
            duration = (gps_data['timestamp'].max() - gps_data['timestamp'].min()).total_seconds()
            if duration < self.min_valid_duration:
                warnings.warn(f"データ期間が短すぎます。最低{self.min_valid_duration}秒必要です。")
                return None
        
        # データのコピーを作成
        df = gps_data.copy()
        
        # 必要な列があるか確認
        required_cols = ['bearing', 'speed']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            warnings.warn(f"必要な列がありません: {missing_cols}")
            return None
        
        # 方向の変化を計算（絶対値）
        df['bearing_change'] = df['bearing'].diff().abs()
        
        # 大きな方向変化をタックまたはジャイブとして識別
        df['is_tack'] = df['bearing_change'] > min_tack_angle
        
        # === 改良ポイント1: より堅牢なタック検出 ===
        # 連続するタックを1つのイベントとしてグループ化
        df['tack_group'] = (df['is_tack'] != df['is_tack'].shift()).cumsum()
        tack_groups = df[df['is_tack']].groupby('tack_group')
        
        # 有意なタックのみを抽出（短すぎる角度変化を除外）
        significant_tacks = []
        for _, group in tack_groups:
            if len(group) >= 2:  # 最低2ポイント以上のタック
                total_angle_change = abs(group['bearing'].iloc[-1] - group['bearing'].iloc[0])
                if total_angle_change > min_tack_angle:
                    significant_tacks.append({
                        'start_idx': group.index[0],
                        'end_idx': group.index[-1],
                        'angle_before': group['bearing'].iloc[0],
                        'angle_after': group['bearing'].iloc[-1],
                        'timestamp': group['timestamp'].iloc[0] if 'timestamp' in group.columns else None
                    })
        
        # タック/ジャイブが少なすぎる場合は処理を中止
        if len(significant_tacks) < 2:
            boat_id = df['boat_id'].iloc[0] if 'boat_id' in df.columns else 'Unknown'
            warnings.warn(f"Boat {boat_id}: タック/ジャイブポイントが不足しているため、風向の推定が困難です。")
            
            # 代替：風上/風下レグの直接検出
            return self._estimate_wind_from_speed_patterns(df, boat_type)
        
        # === 改良ポイント2: 風上/風下レグの自動識別 ===
        # 有効な航路方向のみを抽出（タックを除く安定した区間）
        stable_bearings = []
        stable_speeds = []
        stable_sections = []
        
        for i in range(len(significant_tacks) - 1):
            current_tack = significant_tacks[i]
            next_tack = significant_tacks[i + 1]
            
            # タック間の安定した区間を抽出
            start_idx = current_tack['end_idx'] + 1
            end_idx = next_tack['start_idx'] - 1
            
            if start_idx <= end_idx and end_idx < len(df):  # インデックスの範囲をチェック
                stable_section = df.loc[start_idx:end_idx].copy()
                if len(stable_section) > 5:  # 十分なデータポイントがある場合
                    avg_bearing = stable_section['bearing'].mean()
                    avg_speed = stable_section['speed'].mean()
                    stable_bearings.append(avg_bearing)
                    stable_speeds.append(avg_speed)
                    stable_sections.append(stable_section)
        
        # 航路方向のクラスタリング（主に風上と風下のレグを分離）
        upwind_bearings = []
        upwind_speeds = []
        downwind_bearings = []
        downwind_speeds = []
        
        if len(stable_bearings) >= 4:  # 十分なデータポイント
            bearings_array = np.array(stable_bearings).reshape(-1, 1)
            # 角度の循環性を考慮するための変換
            X = np.column_stack([
                np.cos(np.radians(bearings_array.flatten())),
                np.sin(np.radians(bearings_array.flatten()))
            ])
            
            # 2つのクラスタにグループ化（風上と風下のレグを想定）
            kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
            clusters = kmeans.labels_
            
            # クラスタごとの平均速度を計算
            cluster_speeds = [np.mean([s for s, c in zip(stable_speeds, clusters) if c == i]) for i in range(2)]
            
            # 低速クラスタを風上、高速クラスタを風下と仮定
            upwind_cluster = 0 if cluster_speeds[0] < cluster_speeds[1] else 1
            downwind_cluster = 1 - upwind_cluster
            
            # 各クラスタの方向と速度を抽出
            upwind_bearings = [b for b, c in zip(stable_bearings, clusters) if c == upwind_cluster]
            upwind_speeds = [s for s, c in zip(stable_speeds, clusters) if c == upwind_cluster]
            downwind_bearings = [b for b, c in zip(stable_bearings, clusters) if c == downwind_cluster]
            downwind_speeds = [s for s, c in zip(stable_speeds, clusters) if c == downwind_cluster]
        else:
            # データが不足している場合はヒストグラムベースの方法にフォールバック
            hist, bin_edges = np.histogram(df['bearing'], bins=36, range=(0, 360))
            peak_indices = np.argsort(hist)[-2:]  # 上位2つのピーク
            peak_bins = [(bin_edges[i], bin_edges[i+1]) for i in peak_indices]
            peak_angles = [(bin_start + bin_end) / 2 for bin_start, bin_end in peak_bins]
            
            # 平均速度に基づいて風上/風下を推定
            speeds_per_angle = []
            
            for angle in peak_angles:
                # 特定の方向に近い区間の平均速度を計算
                mask = np.abs((df['bearing'] - angle + 180) % 360 - 180) < 30
                section_speed = df.loc[mask, 'speed'].mean() if sum(mask) > 0 else 0
                speeds_per_angle.append((angle, section_speed))
            
            # 速度でソートして風上（遅い）と風下（速い）を特定
            speeds_per_angle.sort(key=lambda x: x[1])
            
            if len(speeds_per_angle) >= 2:
                upwind_bearings = [speeds_per_angle[0][0]]
                upwind_speeds = [speeds_per_angle[0][1]]
                downwind_bearings = [speeds_per_angle[1][0]]
                downwind_speeds = [speeds_per_angle[1][1]]
            elif len(speeds_per_angle) == 1:
                upwind_bearings = [speeds_per_angle[0][0]]
                upwind_speeds = [speeds_per_angle[0][1]]
        
        # === 改良ポイント3: より精度の高い風向推定 ===
        estimated_wind_direction = None
        confidence_score = 0.5  # デフォルトの信頼度
        
        # 風上レグの方向から風向を推定
        if len(upwind_bearings) >= 2:
            # 複数の風上方向がある場合
            angle_diffs = []
            for i in range(len(upwind_bearings)):
                for j in range(i+1, len(upwind_bearings)):
                    diff = abs(upwind_bearings[i] - upwind_bearings[j])
                    if diff > 180:
                        diff = 360 - diff
                    angle_diffs.append((upwind_bearings[i], upwind_bearings[j], diff))
            
            # 最も角度差が大きいペアを見つける（おそらく反対タック）
            if angle_diffs:
                max_diff_pair = max(angle_diffs, key=lambda x: x[2])
                angle1, angle2 = max_diff_pair[0], max_diff_pair[1]
                
                # 二等分線を計算
                angle_diff = abs(angle1 - angle2)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                    bisector = (min(angle1, angle2) + angle_diff/2) % 360
                else:
                    bisector = (min(angle1, angle2) + angle_diff/2)
                
                # 風向は二等分線の反対方向（180度反転）
                estimated_wind_direction = (bisector + 180) % 360
                confidence_score = 0.8  # 複数の風上レグがある場合は信頼度高
        
        # 単一の風上方向または風下方向から推定
        if estimated_wind_direction is None:
            if len(upwind_bearings) == 1:
                # 1つの風上方向から推定（典型的な風上角度を考慮）
                estimated_wind_direction = (upwind_bearings[0] + 180) % 360
                confidence_score = 0.6  # 単一の風上レグなので中程度の信頼度
            elif len(downwind_bearings) >= 1:
                # 風下方向の反対が風向と仮定
                estimated_wind_direction = (downwind_bearings[0] + 180) % 360
                confidence_score = 0.5  # 風下のみからの推定は信頼度低め
            else:
                # データ不足のためデフォルト値を使用
                boat_id = df['boat_id'].iloc[0] if 'boat_id' in df.columns else 'Unknown'
                warnings.warn(f"Boat {boat_id}: 風向の推定に必要な十分なデータがありません。")
                return None
        
        # === 改良ポイント4: 風速の精度向上 ===
        # 艇種ごとの係数を設定
        # 使用する係数の決定
        use_boat_type = boat_type.lower() if boat_type and boat_type.lower() in self.boat_coefficients else 'default'
        upwind_ratio = self.boat_coefficients[use_boat_type]['upwind']
        downwind_ratio = self.boat_coefficients[use_boat_type]['downwind']
        
        # 風速推定値の計算
        avg_upwind_speed = np.mean(upwind_speeds) if upwind_speeds else 0
        avg_downwind_speed = np.mean(downwind_speeds) if downwind_speeds else 0
        
        est_wind_speed_from_upwind = avg_upwind_speed * upwind_ratio if avg_upwind_speed > 0 else 0
        est_wind_speed_from_downwind = avg_downwind_speed * downwind_ratio if avg_downwind_speed > 0 else 0
        
        # 両方の推定値の重み付き平均を取る
        if est_wind_speed_from_upwind > 0 and est_wind_speed_from_downwind > 0:
            # 風上からの推定値の方が信頼性が高いと仮定
            estimated_wind_speed = (est_wind_speed_from_upwind * 0.7 + est_wind_speed_from_downwind * 0.3)
            confidence_score = min(0.9, confidence_score + 0.1)  # 両方から推定できた場合は信頼度上昇
        elif est_wind_speed_from_upwind > 0:
            estimated_wind_speed = est_wind_speed_from_upwind
        elif est_wind_speed_from_downwind > 0:
            estimated_wind_speed = est_wind_speed_from_downwind
        else:
            estimated_wind_speed = 0
        
        # ノットに変換（1 m/s ≈ 1.94384 ノット）
        estimated_wind_speed_knots = estimated_wind_speed * 1.94384
        
        # === 改良ポイント5: 時間変化を考慮した風向風速推定 ===
        # ウィンドウ分析で時間による変化を推定
        window_size = max(len(df) // 10, 20)  # データの約10%、最低20ポイント
        
        wind_estimates = []
        for i in range(0, len(df), window_size//2):  # 50%オーバーラップのウィンドウ
            end_idx = min(i + window_size, len(df))
            if end_idx - i < window_size // 2:  # 小さすぎるウィンドウはスキップ
                continue
                
            window_data = df.iloc[i:end_idx]
            
            # 各ウィンドウでのタックパターン分析（簡略化）
            # ここでは全体の推定風向と近似値を使用
            
            center_time = window_data['timestamp'].iloc[len(window_data)//2] if 'timestamp' in window_data.columns else None
            center_lat = window_data['latitude'].mean() if 'latitude' in window_data.columns else None
            center_lon = window_data['longitude'].mean() if 'longitude' in window_data.columns else None
            
            # 基本的な時間変化モデル（仮の実装 - より高度な分析は将来的に追加）
            # 風向の時間変動要素を導入
            if 'timestamp' in df.columns:
                time_factor = (window_data['timestamp'].iloc[0] - df['timestamp'].iloc[0]).total_seconds() / \
                            max(1, (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds())
            else:
                time_factor = i / max(1, len(df) - 1)
            
            # 風向の時間変化をモデル化（単純な線形変化 + ノイズ）
            wind_direction_variation = (np.sin(time_factor * np.pi) * 5)  # ±5度程度の変動
            windowed_direction = (estimated_wind_direction + wind_direction_variation) % 360
            
            # 風速の時間変化もモデル化（単純化）
            wind_speed_variation = (np.cos(time_factor * np.pi * 2) * 0.5)  # ±0.5ノット程度の変動
            windowed_speed = max(0, estimated_wind_speed_knots + wind_speed_variation)
            
            # 風向風速推定の精度が時間とともに減少（開始点から離れるほど不確実性が増加）
            time_confidence = max(0.3, 1.0 - abs(0.5 - time_factor) * 0.3)
            window_confidence = confidence_score * time_confidence
            
            # ベイズ推定を使用する場合
            if use_bayesian:
                # シンプルなベイズ推定の実装（仮実装）
                # 実際の実装では過去の風向の履歴や物理モデルを考慮したより複雑なベイズ更新が必要
                
                # 前のウィンドウからの推定がある場合
                if len(wind_estimates) > 0:
                    prior_direction = wind_estimates[-1]['wind_direction']
                    prior_speed = wind_estimates[-1]['wind_speed_knots']
                    prior_confidence = wind_estimates[-1]['confidence']
                    
                    # 前の推定と現在の測定の重み付き平均
                    alpha = 0.3  # 現在の測定の重み
                    updated_direction = self._weighted_angle_average(
                        [prior_direction, windowed_direction],
                        [prior_confidence, window_confidence * alpha]
                    )
                    updated_speed = (prior_speed * prior_confidence + windowed_speed * window_confidence * alpha) / \
                                   (prior_confidence + window_confidence * alpha)
                    updated_confidence = min(0.95, (prior_confidence + window_confidence * alpha) / 2)
                    
                    windowed_direction = updated_direction
                    windowed_speed = updated_speed
                    window_confidence = updated_confidence
            
            wind_estimate = {
                'timestamp': center_time,
                'latitude': center_lat,
                'longitude': center_lon,
                'wind_direction': windowed_direction,
                'wind_speed_knots': windowed_speed,
                'confidence': window_confidence,
                'boat_id': df['boat_id'].iloc[0] if 'boat_id' in df.columns else 'Unknown'
            }
            
            wind_estimates.append(wind_estimate)
        
        # DataFrameに変換
        if wind_estimates:
            wind_df = pd.DataFrame(wind_estimates)
            boat_id = df['boat_id'].iloc[0] if 'boat_id' in df.columns else 'Unknown'
            self.wind_estimates[boat_id] = wind_df
            return wind_df
        else:
            return None
            
    def _estimate_wind_from_speed_patterns(self, df: pd.DataFrame, boat_type: str = 'default') -> Optional[pd.DataFrame]:
        """
        速度パターンからの風向風速推定（タックが検出できない場合のフォールバック）
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ
        boat_type : str
            艇種識別子
            
        Returns:
        --------
        pd.DataFrame or None
            推定風向風速
        """
        # この方法は、タックが検出できない場合の代替手段
        boat_id = df['boat_id'].iloc[0] if 'boat_id' in df.columns else 'Unknown'
        
        # 速度のヒストグラムを作成して最頻値を見つける
        speed_hist, speed_bins = np.histogram(df['speed'], bins=20)
        max_speed_idx = np.argmax(speed_hist)
        mode_speed = (speed_bins[max_speed_idx] + speed_bins[max_speed_idx + 1]) / 2
        
        # 方位のヒストグラムを作成して最頻値を見つける
        bearing_hist, bearing_bins = np.histogram(df['bearing'], bins=36, range=(0, 360))
        sorted_bearing_indices = np.argsort(-bearing_hist)  # 降順
        
        # 上位2つの方位（おそらく反対方向のレグ）
        if len(sorted_bearing_indices) >= 2:
            idx1, idx2 = sorted_bearing_indices[0], sorted_bearing_indices[1]
            bearing1 = (bearing_bins[idx1] + bearing_bins[idx1 + 1]) / 2
            bearing2 = (bearing_bins[idx2] + bearing_bins[idx2 + 1]) / 2
            
            # 2つの方位の角度差を計算
            angle_diff = abs(bearing1 - bearing2)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # 風向の推定（2つの主要方位の中間、180度反転）
            if 70 < angle_diff < 110:
                # 約90度差の場合、リーチング走行と判断
                estimated_wind_direction = (min(bearing1, bearing2) + angle_diff/2 + 180) % 360
            else:
                # そのほかは主に風上/風下走行と判断
                estimated_wind_direction = (min(bearing1, bearing2) + angle_diff/2 + 180) % 360
        else:
            # 単一の主要方位しかない場合
            bearing = (bearing_bins[sorted_bearing_indices[0]] + bearing_bins[sorted_bearing_indices[0] + 1]) / 2
            estimated_wind_direction = (bearing + 180) % 360
        
        # 風速の推定（速度の最頻値から）
        use_boat_type = boat_type.lower() if boat_type and boat_type.lower() in self.boat_coefficients else 'default'
        upwind_ratio = self.boat_coefficients[use_boat_type]['upwind']
        
        # 最頻速度は風上走行時と仮定
        estimated_wind_speed = mode_speed * upwind_ratio
        estimated_wind_speed_knots = estimated_wind_speed * 1.94384  # m/s -> ノット
        
        # 信頼度は低め（タックから推定する方法より精度が低い）
        confidence = 0.4
        
        # 時間変化を簡易的にモデル化
        window_size = max(len(df) // 5, 10)  # データの約20%、最低10ポイント
        
        wind_estimates = []
        for i in range(0, len(df), window_size):
            end_idx = min(i + window_size, len(df))
            window_data = df.iloc[i:end_idx]
            
            center_time = window_data['timestamp'].iloc[len(window_data)//2] if 'timestamp' in window_data.columns else None
            center_lat = window_data['latitude'].mean() if 'latitude' in window_data.columns else None
            center_lon = window_data['longitude'].mean() if 'longitude' in window_data.columns else None
            
            # 微小変動を加える
            variation = np.random.normal(0, 5)  # 平均0、標準偏差5度の正規分布
            windowed_direction = (estimated_wind_direction + variation) % 360
            
            speed_variation = np.random.normal(0, 0.5)  # 平均0、標準偏差0.5ノットの正規分布
            windowed_speed = max(0.1, estimated_wind_speed_knots + speed_variation)
            
            wind_estimate = {
                'timestamp': center_time,
                'latitude': center_lat,
                'longitude': center_lon,
                'wind_direction': windowed_direction,
                'wind_speed_knots': windowed_speed,
                'confidence': confidence,
                'boat_id': boat_id
            }
            wind_estimates.append(wind_estimate)
        
        # DataFrameに変換
        if wind_estimates:
            wind_df = pd.DataFrame(wind_estimates)
            self.wind_estimates[boat_id] = wind_df
            return wind_df
        else:
            return None
    
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
        
    def estimate_wind_field(self, time_point: datetime, grid_resolution: int = 20) -> Optional[Dict[str, Any]]:
        """
        特定時点での風の場を推定
        
        Parameters:
        -----------
        time_point : datetime
            風推定を行いたい時点
        grid_resolution : int
            出力グリッドの解像度
            
        Returns:
        --------
        Dict[str, Any] or None
            緯度・経度グリッドと推定された風向風速データ
        """
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
        lat_grid = np.linspace(min_lat, max_lat, grid_resolution)
        lon_grid = np.linspace(min_lon, max_lon, grid_resolution)
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
            for i in range(grid_resolution):
                for j in range(grid_resolution):
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

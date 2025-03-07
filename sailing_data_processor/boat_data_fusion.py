import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import math
from scipy.spatial import KDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

class BoatDataFusionModel:
    """
    複数艇のデータを融合して風向風速場を推定するモデル
    """
    
    def __init__(self):
        """初期化"""
        # 艇のスキルレベルの辞書（0.0〜1.0のスコア、高いほど信頼度が高い）
        self.boat_skill_levels = {}
        
        # 艇種の特性辞書（デフォルト特性は基準値1.0）
        self.boat_type_characteristics = {
            'default': {'upwind_efficiency': 1.0, 'downwind_efficiency': 1.0, 'pointing_ability': 1.0},
            'laser': {'upwind_efficiency': 1.1, 'downwind_efficiency': 0.9, 'pointing_ability': 1.1},
            'ilca': {'upwind_efficiency': 1.1, 'downwind_efficiency': 0.9, 'pointing_ability': 1.1},
            '470': {'upwind_efficiency': 1.05, 'downwind_efficiency': 1.0, 'pointing_ability': 1.05},
            '49er': {'upwind_efficiency': 0.95, 'downwind_efficiency': 1.2, 'pointing_ability': 0.9},
            'finn': {'upwind_efficiency': 1.15, 'downwind_efficiency': 0.9, 'pointing_ability': 1.15},
            'nacra17': {'upwind_efficiency': 0.9, 'downwind_efficiency': 1.3, 'pointing_ability': 0.85},
            'star': {'upwind_efficiency': 1.2, 'downwind_efficiency': 0.85, 'pointing_ability': 1.2}
        }
        
        # 風向風速推定の履歴
        self.estimation_history = []
        
        # 風向の時間変化モデル
        self.direction_time_change = 0.0  # 度/分
        self.direction_time_change_std = 0.0  # 標準偏差
        
        # 風速の時間変化モデル
        self.speed_time_change = 0.0  # ノット/分
        self.speed_time_change_std = 0.0  # 標準偏差
        
        # ベイズ推定に使用するハイパーパラメータ
        self.wind_dir_prior_mean = None  # 風向の事前確率平均
        self.wind_dir_prior_std = 45.0  # 風向の事前確率標準偏差（度）
        self.wind_speed_prior_mean = None  # 風速の事前確率平均
        self.wind_speed_prior_std = 3.0  # 風速の事前確率標準偏差（ノット）
    
    def set_boat_skill_levels(self, skill_levels: Dict[str, float]):
        """
        各艇のスキルレベルを設定
        
        Parameters:
        -----------
        skill_levels : Dict[str, float]
            艇ID:スキルレベルの辞書（0.0〜1.0のスコア）
        """
        # 値の範囲を0.0〜1.0に制限
        for boat_id, level in skill_levels.items():
            self.boat_skill_levels[boat_id] = max(0.0, min(1.0, level))
    
    def set_boat_types(self, boat_types: Dict[str, str]):
        """
        各艇の艇種を設定
        
        Parameters:
        -----------
        boat_types : Dict[str, str]
            艇ID:艇種の辞書
        """
        self.boat_types = boat_types
    
    def set_wind_priors(self, direction_mean: float = None, direction_std: float = 45.0,
                      speed_mean: float = None, speed_std: float = 3.0):
        """
        風向風速の事前確率を設定
        
        Parameters:
        -----------
        direction_mean : float, optional
            風向の事前確率平均（度）
        direction_std : float
            風向の事前確率標準偏差（度）
        speed_mean : float, optional
            風速の事前確率平均（ノット）
        speed_std : float
            風速の事前確率標準偏差（ノット）
        """
        self.wind_dir_prior_mean = direction_mean
        self.wind_dir_prior_std = direction_std
        self.wind_speed_prior_mean = speed_mean
        self.wind_speed_prior_std = speed_std
    
    def calc_boat_reliability(self, boat_id: str, wind_estimates: pd.DataFrame) -> float:
        """
        各艇の信頼性係数を計算（艇のスキル、データの一貫性、過去の履歴に基づく）
        
        Parameters:
        -----------
        boat_id : str
            艇ID
        wind_estimates : pd.DataFrame
            艇の風向風速推定データ
            
        Returns:
        --------
        float
            信頼性係数（0.0〜1.0）
        """
        # 基本信頼係数はスキルレベル（未設定の場合は中央値0.5）
        base_reliability = self.boat_skill_levels.get(boat_id, 0.5)
        
        # データの一貫性を評価
        consistency_score = 0.7  # デフォルト値
        
        if len(wind_estimates) > 2:
            # 風向と風速の標準偏差を計算
            dir_std = np.std(wind_estimates['wind_direction'])
            speed_std = np.std(wind_estimates['wind_speed_knots'])
            
            # 風向の一貫性（循環データなので特殊処理）
            sin_vals = np.sin(np.radians(wind_estimates['wind_direction']))
            cos_vals = np.cos(np.radians(wind_estimates['wind_direction']))
            r_mean = math.sqrt(np.mean(sin_vals)**2 + np.mean(cos_vals)**2)
            
            # r_meanは0（完全にランダム）から1（完全に一定）の範囲
            dir_consistency = r_mean
            
            # 風速の変動係数（標準偏差/平均）
            if wind_estimates['wind_speed_knots'].mean() > 0:
                speed_cv = speed_std / wind_estimates['wind_speed_knots'].mean()
                speed_consistency = max(0, 1 - speed_cv / 0.5)  # 変動係数0.5以上で信頼性0
            else:
                speed_consistency = 0.5
            
            # 総合一貫性スコア
            consistency_score = 0.7 * dir_consistency + 0.3 * speed_consistency
        
        # 過去の推定履歴との一致度
        history_score = 0.8  # デフォルト値
        
        if self.estimation_history and len(wind_estimates) > 0:
            latest_estimate = wind_estimates.iloc[-1]
            
            # 最新の推定と過去の推定との差異
            deviations = []
            
            for hist_entry in self.estimation_history[-5:]:  # 最新5件を使用
                # 時間差（分）
                time_diff_minutes = (latest_estimate['timestamp'] - hist_entry['timestamp']).total_seconds() / 60
                if abs(time_diff_minutes) > 30:  # 30分以上離れた履歴は使用しない
                    continue
                
                # 風向の差（循環性を考慮）
                dir_diff = abs((latest_estimate['wind_direction'] - hist_entry['wind_direction'] + 180) % 360 - 180)
                
                # 風速の差
                speed_diff = abs(latest_estimate['wind_speed_knots'] - hist_entry['wind_speed_knots'])
                
                # 時間経過を考慮した差異を計算
                expected_dir_change = self.direction_time_change * abs(time_diff_minutes)
                expected_speed_change = self.speed_time_change * abs(time_diff_minutes)
                
                adjusted_dir_diff = max(0, dir_diff - expected_dir_change)
                adjusted_speed_diff = max(0, speed_diff - expected_speed_change)
                
                # 正規化スコア
                if hist_entry['wind_speed_knots'] > 0:
                    norm_dir_score = max(0, 1 - adjusted_dir_diff / 45)  # 45度以上の差で0点
                    norm_speed_score = max(0, 1 - adjusted_speed_diff / (hist_entry['wind_speed_knots'] * 0.3))  # 30%以上の差で0点
                else:
                    norm_dir_score = max(0, 1 - adjusted_dir_diff / 45)
                    norm_speed_score = max(0, 1 - adjusted_speed_diff / 3)  # 3ノット以上の差で0点
                
                deviations.append(0.7 * norm_dir_score + 0.3 * norm_speed_score)
            
            if deviations:
                history_score = np.mean(deviations)
        
        # 総合信頼性スコア
        reliability = 0.4 * base_reliability + 0.4 * consistency_score + 0.2 * history_score
        
        # 艇種の特性を考慮した調整
        if boat_id in self.boat_types:
            boat_type = self.boat_types[boat_id]
            if boat_type in self.boat_type_characteristics:
                # 総合的な艇種性能スコア
                type_score = np.mean([
                    self.boat_type_characteristics[boat_type]['upwind_efficiency'],
                    self.boat_type_characteristics[boat_type]['downwind_efficiency'],
                    self.boat_type_characteristics[boat_type]['pointing_ability']
                ])
                
                # 信頼性スコアを艇種性能で微調整
                reliability *= (0.9 + 0.2 * (type_score - 1.0))
        
        # 0.0〜1.0の範囲に制限
        return max(0.0, min(1.0, reliability))
    
    def fuse_wind_estimates(self, boats_estimates: Dict[str, pd.DataFrame], 
                          time_point: datetime = None) -> Optional[Dict[str, Any]]:
        """
        複数艇からの風向風速推定を融合
        
        Parameters:
        -----------
        boats_estimates : Dict[str, pd.DataFrame]
            艇ID:風向風速推定DataFrameの辞書
        time_point : datetime, optional
            対象時間点（指定がない場合は最新の共通時間）
            
        Returns:
        --------
        Dict[str, Any] or None
            融合された風向風速データと信頼度
        """
        if not boats_estimates or len(boats_estimates) == 0:
            return None
        
        # 時間点の決定
        if time_point is None:
            # 各艇の最後の推定時刻を取得
            all_times = []
            for boat_id, df in boats_estimates.items():
                if 'timestamp' in df.columns and not df.empty:
                    all_times.append(df['timestamp'].iloc[-1])
            
            if not all_times:
                return None
            
            # 最も古い「最新」時刻を使用
            time_point = min(all_times)
        
        # 各艇の推定データを収集
        boat_data = []
        
        for boat_id, df in boats_estimates.items():
            if 'timestamp' not in df.columns or df.empty:
                continue
            
            # 指定時間に最も近いデータを探す
            time_diffs = abs((df['timestamp'] - time_point).dt.total_seconds())
            
            if time_diffs.min() <= 60:  # 60秒以内のデータのみ使用
                closest_idx = time_diffs.idxmin()
                
                # データを取得
                wind_dir = df.loc[closest_idx, 'wind_direction']
                wind_speed = df.loc[closest_idx, 'wind_speed_knots']
                confidence = df.loc[closest_idx, 'confidence'] if 'confidence' in df.columns else 0.7
                
                # 位置情報（あれば）
                latitude = df.loc[closest_idx, 'latitude'] if 'latitude' in df.columns else None
                longitude = df.loc[closest_idx, 'longitude'] if 'longitude' in df.columns else None
                
                # 信頼性係数を計算
                reliability = self.calc_boat_reliability(boat_id, df)
                
                # 統合スコア
                combined_weight = confidence * reliability
                
                boat_data.append({
                    'boat_id': boat_id,
                    'timestamp': df.loc[closest_idx, 'timestamp'],
                    'wind_direction': wind_dir,
                    'wind_speed_knots': wind_speed,
                    'latitude': latitude,
                    'longitude': longitude,
                    'raw_confidence': confidence,
                    'reliability': reliability,
                    'weight': combined_weight
                })
        
        if not boat_data:
            return None
        
        # ベイズ更新を使用した風向風速の統合
        integrated_estimate = self._bayesian_wind_integration(boat_data, time_point)
        
        # 履歴に追加
        self.estimation_history.append({
            'timestamp': time_point,
            'wind_direction': integrated_estimate['wind_direction'],
            'wind_speed_knots': integrated_estimate['wind_speed']
        })
        
        # 履歴が長すぎる場合は古いエントリを削除
        if len(self.estimation_history) > 100:
            self.estimation_history = self.estimation_history[-100:]
        
        # 時間変化モデルの更新
        self._update_time_change_model()
        
        return integrated_estimate
    
    def _bayesian_wind_integration(self, boat_data: List[Dict[str, Any]], 
                                 time_point: datetime) -> Dict[str, Any]:
        """
        ベイズ更新を使用した風向風速の統合
        
        Parameters:
        -----------
        boat_data : List[Dict[str, Any]]
            各艇の風推定データ
        time_point : datetime
            対象時間点
            
        Returns:
        --------
        Dict[str, Any]
            ベイズ更新された風向風速データ
        """
        # 十分なデータがなければ単純な重み付き平均を使用
        if len(boat_data) < 3:
            return self._weighted_average_integration(boat_data)
        
        # 1. 風向の統合（循環データなので特殊処理）
        
        # 事前分布の平均（指定がない場合は最初の推定値を使用）
        if self.wind_dir_prior_mean is None:
            self.wind_dir_prior_mean = boat_data[0]['wind_direction']
        
        # 風向のsin/cos成分を変換
        dir_sin_values = [math.sin(math.radians(d['wind_direction'])) for d in boat_data]
        dir_cos_values = [math.cos(math.radians(d['wind_direction'])) for d in boat_data]
        
        # 風向の重み
        dir_weights = [d['weight'] for d in boat_data]
        
        # 事前確率の組み込み
        prior_weight = 0.3  # 事前確率の重み
        
        # 事前分布のsin/cos
        prior_sin = math.sin(math.radians(self.wind_dir_prior_mean))
        prior_cos = math.cos(math.radians(self.wind_dir_prior_mean))
        
        # 重み付き平均のsin/cos
        weighted_sin = np.average(dir_sin_values, weights=dir_weights)
        weighted_cos = np.average(dir_cos_values, weights=dir_weights)
        
        # 事前確率と観測値の統合
        posterior_sin = (prior_sin * prior_weight + weighted_sin * (1 - prior_weight))
        posterior_cos = (prior_cos * prior_weight + weighted_cos * (1 - prior_weight))
        
        # 風向の復元
        integrated_direction = math.degrees(math.atan2(posterior_sin, posterior_cos)) % 360
        
        # 分散の計算
        dir_variance = 0.0
        for i, d in enumerate(boat_data):
            # 風向の差（循環性を考慮）
            dir_diff = abs((d['wind_direction'] - integrated_direction + 180) % 360 - 180)
            dir_variance += dir_diff**2 * dir_weights[i]
        
        dir_variance /= sum(dir_weights)
        dir_std = math.sqrt(dir_variance)
        
        # 方向の不確実性（0-1の範囲で、0が最も確実）
        dir_uncertainty = min(1.0, dir_std / 90.0)  # 90度の標準偏差で最大不確実性
        
        # 2. 風速の統合
        
        # 事前分布の平均（指定がない場合は最初の推定値を使用）
        if self.wind_speed_prior_mean is None:
            self.wind_speed_prior_mean = boat_data[0]['wind_speed_knots']
        
        # 風速の値と重み
        speed_values = [d['wind_speed_knots'] for d in boat_data]
        speed_weights = [d['weight'] for d in boat_data]
        
        # ロバスト重み付き平均（外れ値に強い）
        speed_data = np.array(speed_values)
        speed_median = np.median(speed_data)
        
        # 中央値からの差を計算
        speed_deviations = np.abs(speed_data - speed_median)
        max_deviation = np.max(speed_deviations) if len(speed_deviations) > 0 else 1.0
        if max_deviation == 0:
            max_deviation = 1.0
        
        # 中央値からの距離に基づいて重みを調整
        robust_weights = np.array(speed_weights) * (1 - speed_deviations / max_deviation)
        robust_weights = np.maximum(0.1, robust_weights)  # 最小重みを0.1に設定
        
        # 重み付き平均
        if sum(robust_weights) > 0:
            weighted_speed = np.average(speed_data, weights=robust_weights)
        else:
            weighted_speed = np.mean(speed_data)
        
        # 事前確率と観測値の統合
        integrated_speed = (self.wind_speed_prior_mean * prior_weight + 
                         weighted_speed * (1 - prior_weight))
        
        # 分散の計算
        speed_variance = 0.0
        for i, speed in enumerate(speed_values):
            speed_variance += (speed - integrated_speed)**2 * robust_weights[i]
        
        if sum(robust_weights) > 0:
            speed_variance /= sum(robust_weights)
        else:
            speed_variance = np.var(speed_values)
        
        speed_std = math.sqrt(speed_variance)
        
        # 速度の不確実性（0-1の範囲で、0が最も確実）
        speed_uncertainty = min(1.0, speed_std / (integrated_speed * 0.5 + 0.1))  # 風速の50%の標準偏差で最大不確実性
        
        # 3. 信頼度の計算
        
        # 観測数に基づく信頼度向上
        n_factor = min(0.2, 0.05 * len(boat_data))  # 最大0.2の信頼度向上
        
        # 総合的な信頼度
        base_confidence = 0.5 + n_factor
        adjusted_confidence = base_confidence * (1 - 0.6 * dir_uncertainty) * (1 - 0.4 * speed_uncertainty)
        
        # 経度緯度の計算（重み付き平均）
        latitude, longitude = None, None
        valid_positions = [(d['latitude'], d['longitude'], d['weight']) 
                         for d in boat_data if d['latitude'] is not None and d['longitude'] is not None]
        
        if valid_positions:
            total_weight = sum(w for _, _, w in valid_positions)
            if total_weight > 0:
                latitude = sum(lat * w for lat, _, w in valid_positions) / total_weight
                longitude = sum(lon * w for _, lon, w in valid_positions) / total_weight
        
        # 結果の整理
        return {
            'timestamp': time_point,
            'wind_direction': integrated_direction,
            'wind_speed': integrated_speed,
            'confidence': adjusted_confidence,
            'direction_std': dir_std,
            'speed_std': speed_std,
            'latitude': latitude,
            'longitude': longitude,
            'boat_count': len(boat_data)
        }
    
    def _weighted_average_integration(self, boat_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        単純な重み付き平均による風向風速の統合
        
        Parameters:
        -----------
        boat_data : List[Dict[str, Any]]
            各艇の風推定データ
            
        Returns:
        --------
        Dict[str, Any]
            重み付き平均による風向風速データ
        """
        # 重み付き平均のための準備
        dir_sin_values = [math.sin(math.radians(d['wind_direction'])) for d in boat_data]
        dir_cos_values = [math.cos(math.radians(d['wind_direction'])) for d in boat_data]
        speed_values = [d['wind_speed_knots'] for d in boat_data]
        weights = [d['weight'] for d in boat_data]
        timestamps = [d['timestamp'] for d in boat_data]
        
        # 風向の重み付き平均（sin/cosを使用）
        if sum(weights) > 0:
            weighted_sin = np.average(dir_sin_values, weights=weights)
            weighted_cos = np.average(dir_cos_values, weights=weights)
            integrated_direction = math.degrees(math.atan2(weighted_sin, weighted_cos)) % 360
            
            # 風速の重み付き平均
            integrated_speed = np.average(speed_values, weights=weights)
        else:
            # 重みがゼロの場合は単純平均
            weighted_sin = np.mean(dir_sin_values)
            weighted_cos = np.mean(dir_cos_values)
            integrated_direction = math.degrees(math.atan2(weighted_sin, weighted_cos)) % 360
            
            integrated_speed = np.mean(speed_values)
        
        # 時間の中央値
        if timestamps:
            integrated_time = sorted(timestamps)[len(timestamps) // 2]
        else:
            integrated_time = datetime.now()
        
        # 信頼度の計算（艇数と重みの平均に基づく）
        avg_weight = np.mean([d['weight'] for d in boat_data])
        confidence = min(0.9, 0.4 + 0.1 * len(boat_data) + 0.4 * avg_weight)
        
        # 経度緯度の計算（重み付き平均）
        latitude, longitude = None, None
        valid_positions = [(d['latitude'], d['longitude'], d['weight']) 
                         for d in boat_data if d['latitude'] is not None and d['longitude'] is not None]
        
        if valid_positions:
            total_weight = sum(w for _, _, w in valid_positions)
            if total_weight > 0:
                latitude = sum(lat * w for lat, _, w in valid_positions) / total_weight
                longitude = sum(lon * w for _, lon, w in valid_positions) / total_weight
        
        # 標準偏差の計算
        dir_var = 0.0
        speed_var = 0.0
        
        for d, w in zip(boat_data, weights):
            # 風向の差（循環性を考慮）
            dir_diff = abs((d['wind_direction'] - integrated_direction + 180) % 360 - 180)
            dir_var += dir_diff**2 * w
            
            # 風速の差
            speed_diff = d['wind_speed_knots'] - integrated_speed
            speed_var += speed_diff**2 * w
        
        if sum(weights) > 0:
            dir_var /= sum(weights)
            speed_var /= sum(weights)
        
        dir_std = math.sqrt(dir_var)
        speed_std = math.sqrt(speed_var)
        
        return {
            'timestamp': integrated_time,
            'wind_direction': integrated_direction,
            'wind_speed': integrated_speed,
            'confidence': confidence,
            'direction_std': dir_std,
            'speed_std': speed_std,
            'latitude': latitude,
            'longitude': longitude,
            'boat_count': len(boat_data)
        }
    
    def _update_time_change_model(self):
        """
        風向風速の時間変化モデルを更新
        """
        if len(self.estimation_history) < 3:
            return
        
        # 直近の履歴のみを使用
        recent_history = self.estimation_history[-10:]
        
        # 隣接する時間点間の変化率を計算
        dir_changes = []
        speed_changes = []
        
        for i in range(1, len(recent_history)):
            prev = recent_history[i-1]
            curr = recent_history[i]
            
            # 時間差（分）
            time_diff_minutes = (curr['timestamp'] - prev['timestamp']).total_seconds() / 60
            if time_diff_minutes <= 0:
                continue
            
            # 風向の変化率（度/分）
            dir_diff = (curr['wind_direction'] - prev['wind_direction'] + 180) % 360 - 180
            dir_change_rate = dir_diff / time_diff_minutes
            
            # 風速の変化率（ノット/分）
            speed_diff = curr['wind_speed_knots'] - prev['wind_speed_knots']
            speed_change_rate = speed_diff / time_diff_minutes
            
            dir_changes.append(dir_change_rate)
            speed_changes.append(speed_change_rate)
        
        # 変化率の統計を更新
        if dir_changes:
            self.direction_time_change = np.median(dir_changes)
            self.direction_time_change_std = np.std(dir_changes)
        
        if speed_changes:
            self.speed_time_change = np.median(speed_changes)
            self.speed_time_change_std = np.std(speed_changes)
    
    def create_spatiotemporal_wind_field(self, time_points: List[datetime], 
                                       grid_resolution: int = 20) -> Dict[datetime, Dict[str, Any]]:
        """
        時空間的な風の場を作成
        
        Parameters:
        -----------
        time_points : List[datetime]
            対象時間点のリスト
        grid_resolution : int
            空間グリッドの解像度
            
        Returns:
        --------
        Dict[datetime, Dict[str, Any]]
            時間点ごとの風の場データ
        """
        # 履歴が不十分な場合は空の辞書を返す
        if len(self.estimation_history) < 3:
            return {}
        
        wind_fields = {}
        
        # 各時間点での風の場を推定
        for time_point in time_points:
            wind_field = self._estimate_wind_field_at_time(time_point, grid_resolution)
            if wind_field is not None:
                wind_fields[time_point] = wind_field
        
        return wind_fields
    
    def _estimate_wind_field_at_time(self, time_point: datetime, 
                                   grid_resolution: int = 20) -> Optional[Dict[str, Any]]:
        """
        特定時点での風の場を推定
        
        Parameters:
        -----------
        time_point : datetime
            対象時間点
        grid_resolution : int
            空間グリッドの解像度
            
        Returns:
        --------
        Dict[str, Any] or None
            風の場データ
        """
        # 時間的に近い履歴エントリを探す
        nearby_entries = []
        
        for entry in self.estimation_history:
            # 時間差（分）
            time_diff_minutes = abs((entry['timestamp'] - time_point).total_seconds() / 60)
            
            if time_diff_minutes <= 30:  # 30分以内のデータを使用
                # 時間差に基づく重み
                time_weight = max(0.1, 1.0 - time_diff_minutes / 30)
                
                nearby_entries.append({
                    'timestamp': entry['timestamp'],
                    'wind_direction': entry['wind_direction'],
                    'wind_speed_knots': entry['wind_speed_knots'],
                    'time_weight': time_weight
                })
        
        if not nearby_entries:
            return None
        
        # 標準的なグリッド境界
        lat_min, lat_max = 35.6, 35.7  # 仮の値
        lon_min, lon_max = 139.7, 139.8  # 仮の値
        
        # 位置情報がある場合は境界を調整
        location_entries = [entry for entry in self.estimation_history 
                          if 'latitude' in entry and entry['latitude'] is not None 
                          and 'longitude' in entry and entry['longitude'] is not None]
        
        if location_entries:
            lat_values = [entry['latitude'] for entry in location_entries]
            lon_values = [entry['longitude'] for entry in location_entries]
            
            lat_min, lat_max = min(lat_values), max(lat_values)
            lon_min, lon_max = min(lon_values), max(lon_values)
            
            # 少し余裕を持たせる
            lat_margin = (lat_max - lat_min) * 0.1
            lon_margin = (lon_max - lon_min) * 0.1
            lat_min -= lat_margin
            lat_max += lat_margin
            lon_min -= lon_margin
            lon_max += lon_margin
        
        # グリッドの作成
        lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
        lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
        grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
        
        # 時間に最も近い風向風速を基準とし、時間変化モデルで補正
        closest_entry = min(nearby_entries, key=lambda e: abs((e['timestamp'] - time_point).total_seconds()))
        time_diff_minutes = (time_point - closest_entry['timestamp']).total_seconds() / 60
        
        # 基準風向風速の算出
        base_direction = closest_entry['wind_direction']
        base_speed = closest_entry['wind_speed_knots']
        
        # 時間変化モデルによる補正
        projected_direction = (base_direction + self.direction_time_change * time_diff_minutes) % 360
        projected_speed = max(0, base_speed + self.speed_time_change * time_diff_minutes)
        
        # 時間変化の不確実性
        direction_uncertainty = min(1.0, abs(time_diff_minutes) * self.direction_time_change_std / 30)
        speed_uncertainty = min(1.0, abs(time_diff_minutes) * self.speed_time_change_std / (base_speed * 0.2 + 0.1))
        
        # 風向風速グリッドを初期化
        wind_directions = np.ones_like(grid_lats) * projected_direction
        wind_speeds = np.ones_like(grid_lats) * projected_speed
        confidence = np.ones_like(grid_lats) * max(0.1, 0.8 - 0.4 * direction_uncertainty - 0.4 * speed_uncertainty)
        
        # 空間的な変動（単純な実装）
        # TODO: 空間補間モデルの高度化
        
        # 位置情報がある場合は空間変動を計算
        position_entries = [e for e in nearby_entries if 'latitude' in e and e['latitude'] is not None 
                         and 'longitude' in e and e['longitude'] is not None]
        
        if position_entries and len(position_entries) >= 3:
            # ガウス過程回帰を使用した空間補間
            points = np.array([[e['latitude'], e['longitude']] for e in position_entries])
            
            # 風向のsin/cos成分
            dir_sin = np.array([math.sin(math.radians(e['wind_direction'])) for e in position_entries])
            dir_cos = np.array([math.cos(math.radians(e['wind_direction'])) for e in position_entries])
            speeds = np.array([e['wind_speed_knots'] for e in position_entries])
            
            # カーネルの定義
            kernel = ConstantKernel(1.0) * RBF(length_scale=0.01) + WhiteKernel(noise_level=0.1)
            
            try:
                # sin/cos成分のGP回帰
                gp_sin = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
                gp_cos = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
                gp_speed = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
                
                gp_sin.fit(points, dir_sin)
                gp_cos.fit(points, dir_cos)
                gp_speed.fit(points, speeds)
                
                # グリッドポイント形状の変更
                X_pred = np.column_stack([grid_lats.ravel(), grid_lons.ravel()])
                
                # 各グリッドポイントでの予測
                sin_pred = gp_sin.predict(X_pred).reshape(grid_lats.shape)
                cos_pred = gp_cos.predict(X_pred).reshape(grid_lats.shape)
                speed_pred = gp_speed.predict(X_pred).reshape(grid_lats.shape)
                
                # 風向の復元
                wind_directions = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360
                wind_speeds = np.maximum(0, speed_pred)
                
                # ガウス過程の不確実性を考慮した信頼度
                _, sin_std = gp_sin.predict(X_pred, return_std=True)
                _, cos_std = gp_cos.predict(X_pred, return_std=True)
                _, speed_std = gp_speed.predict(X_pred, return_std=True)
                
                sin_std = sin_std.reshape(grid_lats.shape)
                cos_std = cos_std.reshape(grid_lats.shape)
                speed_std = speed_std.reshape(grid_lats.shape)
                
                # 正規化された不確実性
                dir_uncertainty = np.minimum(1.0, np.sqrt(sin_std**2 + cos_std**2) / 0.5)
                speed_uncertainty = np.minimum(1.0, speed_std / (wind_speeds * 0.3 + 0.1))
                
                # 信頼度を更新
                confidence = np.maximum(0.1, 0.8 - 0.4 * dir_uncertainty - 0.4 * speed_uncertainty)
                
            except Exception as e:
                # ガウス過程回帰に失敗した場合はIDW（逆距離加重）を使用
                for i in range(grid_lats.shape[0]):
                    for j in range(grid_lats.shape[1]):
                        grid_lat, grid_lon = grid_lats[i, j], grid_lons[i, j]
                        
                        # 各エントリとの距離を計算
                        distances = np.array([
                            np.sqrt((grid_lat - e['latitude'])**2 + (grid_lon - e['longitude'])**2)
                            for e in position_entries
                        ])
                        
                        # 重みを計算
                        if np.any(distances < 1.0):  # 非常に近い点がある場合
                            idx = np.argmin(distances)
                            wind_directions[i, j] = position_entries[idx]['wind_direction']
                            wind_speeds[i, j] = position_entries[idx]['wind_speed_knots']
                        else:
                            weights = 1.0 / (distances + 1.0)**2
                            weights /= weights.sum()
                            
                            # 風向のsin/cos成分を使用
                            sin_vals = np.array([
                                math.sin(math.radians(e['wind_direction'])) * weights[idx]
                                for idx, e in enumerate(position_entries)
                            ])
                            cos_vals = np.array([
                                math.cos(math.radians(e['wind_direction'])) * weights[idx]
                                for idx, e in enumerate(position_entries)
                            ])
                            
                            # 風向の復元
                            wind_directions[i, j] = math.degrees(math.atan2(sin_vals.sum(), cos_vals.sum())) % 360
                            
                            # 風速の重み付き平均
                            wind_speeds[i, j] = sum(
                                e['wind_speed_knots'] * weights[idx]
                                for idx, e in enumerate(position_entries)
                            )
                            
                            # 距離に基づく信頼度の減衰
                            confidence[i, j] = max(0.1, 0.8 - 0.3 * (min(distances) / 1000))  # 1km以上で0.5減少
        
        return {
            'lat_grid': grid_lats,
            'lon_grid': grid_lons,
            'wind_direction': wind_directions,
            'wind_speed': wind_speeds,
            'confidence': confidence,
            'time': time_point
        }

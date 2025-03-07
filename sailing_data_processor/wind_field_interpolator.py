import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.interpolate import Rbf, LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource, Normalize
import matplotlib.cm as cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel

class WindFieldInterpolator:
    """
    風向風速場の高精度時空間補間を行うクラス
    時間的変化と空間的分布を統合して滑らかな風向風速場を推定
    """
    
    def __init__(self):
        """初期化"""
        # 推定された風の場データ
        self.wind_field_data = {}  # 時間: 風の場データの辞書
        
        # 時間変化モデルパラメータ
        self.time_kernel_length_scale = 10.0  # 分単位（デフォルト：10分）
        self.space_kernel_length_scale = 0.01  # 度単位（約1.1km）
        
        # 物理モデルパラメータ
        self.terrain_influence = 0.0  # 地形の影響度
        self.thermal_influence = 0.0  # 熱的影響度
        self.shore_influence = 0.0  # 海岸線影響度
        
        # 風向の時間変化トレンド（度/分）
        self.wind_dir_trend = 0.0
        
        # 風速の時間変化トレンド（ノット/分）
        self.wind_speed_trend = 0.0
        
        # 補間方法
        self.interp_method = 'gp'  # 'gp'（ガウス過程）, 'rbf', 'idw'
        
        # 最後に更新された時刻
        self.last_update_time = None
    
    def add_wind_field(self, time_point: datetime, wind_field: Dict[str, Any]):
        """
        風の場データを追加
        
        Parameters:
        -----------
        time_point : datetime
            時間点
        wind_field : Dict[str, Any]
            風の場データ
        """
        self.wind_field_data[time_point] = wind_field.copy()
        
        # 時間ソート
        self.wind_field_data = dict(sorted(self.wind_field_data.items()))
        
        # データが増えすぎないように古いデータを削除
        if len(self.wind_field_data) > 20:
            oldest_time = min(self.wind_field_data.keys())
            del self.wind_field_data[oldest_time]
        
        # 時間変化トレンドを更新
        self._update_time_trends()
        
        # 最後の更新時刻を記録
        self.last_update_time = datetime.now()
    
    def _update_time_trends(self):
        """時間変化トレンドを更新"""
        if len(self.wind_field_data) < 2:
            return
        
        time_points = sorted(self.wind_field_data.keys())
        
        # 風向トレンドの計算
        dir_changes = []
        speed_changes = []
        
        for i in range(1, len(time_points)):
            t1, t2 = time_points[i-1], time_points[i]
            field1, field2 = self.wind_field_data[t1], self.wind_field_data[t2]
            
            # 時間差（分）
            time_diff_minutes = (t2 - t1).total_seconds() / 60
            if time_diff_minutes <= 0:
                continue
            
            # 中心点の風向と風速を取得
            center_i = field1['wind_direction'].shape[0] // 2
            center_j = field1['wind_direction'].shape[1] // 2
            
            dir1 = field1['wind_direction'][center_i, center_j]
            dir2 = field2['wind_direction'][center_i, center_j]
            
            speed1 = field1['wind_speed'][center_i, center_j]
            speed2 = field2['wind_speed'][center_i, center_j]
            
            # 風向変化（循環を考慮）
            dir_diff = (dir2 - dir1 + 180) % 360 - 180
            dir_change = dir_diff / time_diff_minutes
            
            # 風速変化
            speed_diff = speed2 - speed1
            speed_change = speed_diff / time_diff_minutes
            
            dir_changes.append(dir_change)
            speed_changes.append(speed_change)
        
        # メディアンを使用して外れ値の影響を抑制
        if dir_changes:
            self.wind_dir_trend = np.median(dir_changes)
        
        if speed_changes:
            self.wind_speed_trend = np.median(speed_changes)
    
    def interpolate_wind_field(self, target_time: datetime, 
                             resolution: int = None, 
                             method: str = None) -> Optional[Dict[str, Any]]:
        """
        指定時間の風の場を補間
        
        Parameters:
        -----------
        target_time : datetime
            対象時間
        resolution : int, optional
            出力解像度（指定がない場合は元データと同解像度）
        method : str, optional
            補間方法（'gp', 'rbf', 'idw'）、指定がない場合はインスタンス設定を使用
            
        Returns:
        --------
        Dict[str, Any] or None
            補間された風の場
        """
        if not self.wind_field_data:
            return None
            
        # 使用する補間方法
        if method is None:
            method = self.interp_method
            
        # 時間的に最も近いデータポイントを見つける
        time_diffs = [(abs((t - target_time).total_seconds()), t) for t in self.wind_field_data.keys()]
        time_diffs.sort()
        
        # 対象時間の前後のデータを抽出
        before_times = []
        after_times = []
        
        for _, t in time_diffs:
            if t <= target_time:
                before_times.append(t)
            else:
                after_times.append(t)
        
        before_times.sort(reverse=True)  # 直近の時間順
        after_times.sort()  # 直近の時間順
        
        if not before_times and not after_times:
            return None
        
        # 最近傍の時間を使用
        nearest_time = time_diffs[0][1]
        base_field = self.wind_field_data[nearest_time]
        
        # 出力解像度の決定
        if resolution is None:
            resolution = base_field['wind_direction'].shape[0]
        
        # 時間差（分単位）
        time_diff_minutes = (target_time - nearest_time).total_seconds() / 60
        
        # 時間変化が小さい場合は最近傍の風の場をそのまま使用
        if abs(time_diff_minutes) < 1.0:
            return self._resample_wind_field(base_field, resolution)
        
        # 補間方法に応じて処理
        if method == 'gp':
            return self._gp_interpolate(target_time, resolution)
        elif method == 'rbf':
            return self._rbf_interpolate(target_time, resolution)
        else:  # 'idw' or fallback
            return self._idw_interpolate(target_time, resolution)
    
    def _gp_interpolate(self, target_time: datetime, resolution: int) -> Dict[str, Any]:
        """
        ガウス過程による補間
        
        Parameters:
        -----------
        target_time : datetime
            対象時間
        resolution : int
            出力解像度
            
        Returns:
        --------
        Dict[str, Any]
            補間された風の場
        """
        # 時空間データポイントの収集
        data_points = []
        wind_dir_sin = []
        wind_dir_cos = []
        wind_speeds = []
        
        for t, field in self.wind_field_data.items():
            # 時間差（分単位）
            time_diff_minutes = (t - target_time).total_seconds() / 60
            
            # グリッドサイズの取得
            grid_shape = field['wind_direction'].shape
            
            # グリッドからいくつかのポイントをサンプリング（すべては使わない）
            sample_rate = max(1, grid_shape[0] // 5)  # 5x5程度のポイントを使用
            
            for i in range(0, grid_shape[0], sample_rate):
                for j in range(0, grid_shape[1], sample_rate):
                    lat = field['lat_grid'][i, j]
                    lon = field['lon_grid'][i, j]
                    
                    # 時空間座標（時間、緯度、経度）
                    data_points.append([time_diff_minutes, lat, lon])
                    
                    # 風向のsin/cos成分
                    dir_rad = math.radians(field['wind_direction'][i, j])
                    wind_dir_sin.append(math.sin(dir_rad))
                    wind_dir_cos.append(math.cos(dir_rad))
                    
                    # 風速
                    wind_speeds.append(field['wind_speed'][i, j])
        
        if not data_points:
            return None
        
        # データ変換
        X = np.array(data_points)
        y_sin = np.array(wind_dir_sin)
        y_cos = np.array(wind_dir_cos)
        y_speed = np.array(wind_speeds)
        
        # カーネルの定義（時間と空間の両方を考慮）
        time_kernel = ConstantKernel(1.0) * Matern(length_scale=[self.time_kernel_length_scale, 
                                                            self.space_kernel_length_scale, 
                                                            self.space_kernel_length_scale], 
                                              nu=1.5)
        noise_kernel = WhiteKernel(noise_level=0.1)
        full_kernel = time_kernel + noise_kernel
        
        try:
            # ガウス過程回帰モデルの作成
            gp_sin = GaussianProcessRegressor(kernel=full_kernel, n_restarts_optimizer=2)
            gp_cos = GaussianProcessRegressor(kernel=full_kernel, n_restarts_optimizer=2)
            gp_speed = GaussianProcessRegressor(kernel=full_kernel, n_restarts_optimizer=2)
            
            # モデルの学習
            gp_sin.fit(X, y_sin)
            gp_cos.fit(X, y_cos)
            gp_speed.fit(X, y_speed)
            
            # 出力グリッドの作成
            # 緯度経度範囲の決定
            all_lats = np.concatenate([field['lat_grid'].flatten() for field in self.wind_field_data.values()])
            all_lons = np.concatenate([field['lon_grid'].flatten() for field in self.wind_field_data.values()])
            
            lat_min, lat_max = all_lats.min(), all_lats.max()
            lon_min, lon_max = all_lons.min(), all_lons.max()
            
            lat_grid = np.linspace(lat_min, lat_max, resolution)
            lon_grid = np.linspace(lon_min, lon_max, resolution)
            grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
            
            # 予測用の座標
            XX = np.column_stack([
                np.zeros(grid_lats.size),  # 時間差はゼロ（対象時間）
                grid_lats.flatten(), 
                grid_lons.flatten()
            ])
            
            # ガウス過程による予測
            sin_pred, sin_std = gp_sin.predict(XX, return_std=True)
            cos_pred, cos_std = gp_cos.predict(XX, return_std=True)
            speed_pred, speed_std = gp_speed.predict(XX, return_std=True)
            
            # グリッド形状に変換
            sin_pred = sin_pred.reshape(grid_lats.shape)
            cos_pred = cos_pred.reshape(grid_lats.shape)
            speed_pred = speed_pred.reshape(grid_lats.shape)
            
            sin_std = sin_std.reshape(grid_lats.shape)
            cos_std = cos_std.reshape(grid_lats.shape)
            speed_std = speed_std.reshape(grid_lats.shape)
            
            # 風向の復元
            wind_directions = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360
            
            # 風速（負の値は0に制限）
            wind_speeds = np.maximum(0, speed_pred)
            
            # 信頼度の計算
            direction_uncertainty = np.sqrt(sin_std**2 + cos_std**2)
            speed_uncertainty = speed_std
            
            max_dir_uncertainty = np.max(direction_uncertainty)
            max_speed_uncertainty = np.max(speed_uncertainty)
            
            if max_dir_uncertainty > 0:
                norm_dir_uncertainty = direction_uncertainty / max_dir_uncertainty
            else:
                norm_dir_uncertainty = np.zeros_like(direction_uncertainty)
                
            if max_speed_uncertainty > 0:
                norm_speed_uncertainty = speed_uncertainty / max_speed_uncertainty
            else:
                norm_speed_uncertainty = np.zeros_like(speed_uncertainty)
            
            confidence = 1.0 - 0.5 * norm_dir_uncertainty - 0.3 * norm_speed_uncertainty
            
            # 結果の整理
            return {
                'lat_grid': grid_lats,
                'lon_grid': grid_lons,
                'wind_direction': wind_directions,
                'wind_speed': wind_speeds,
                'confidence': confidence,
                'dir_uncertainty': direction_uncertainty,
                'speed_uncertainty': speed_uncertainty,
                'time': target_time
            }
            
        except Exception as e:
            # ガウス過程が失敗した場合はIDWにフォールバック
            print(f"ガウス過程補間に失敗しました: {e}")
            return self._idw_interpolate(target_time, resolution)
    
    def _rbf_interpolate(self, target_time: datetime, resolution: int) -> Dict[str, Any]:
        """
        Radial Basis Function (RBF) による補間
        
        Parameters:
        -----------
        target_time : datetime
            対象時間
        resolution : int
            出力解像度
            
        Returns:
        --------
        Dict[str, Any]
            補間された風の場
        """
        # 時空間データポイントの収集
        data_points = []
        wind_dir_sin = []
        wind_dir_cos = []
        wind_speeds = []
        
        for t, field in self.wind_field_data.items():
            # 時間差（分単位）
            time_diff_minutes = (t - target_time).total_seconds() / 60
            
            # グリッドサイズの取得
            grid_shape = field['wind_direction'].shape
            
            # グリッドからいくつかのポイントをサンプリング
            sample_rate = max(1, grid_shape[0] // 5)
            
            for i in range(0, grid_shape[0], sample_rate):
                for j in range(0, grid_shape[1], sample_rate):
                    lat = field['lat_grid'][i, j]
                    lon = field['lon_grid'][i, j]
                    
                    # 時空間座標（時間、緯度、経度）
                    data_points.append([time_diff_minutes, lat, lon])
                    
                    # 風向のsin/cos成分
                    dir_rad = math.radians(field['wind_direction'][i, j])
                    wind_dir_sin.append(math.sin(dir_rad))
                    wind_dir_cos.append(math.cos(dir_rad))
                    
                    # 風速
                    wind_speeds.append(field['wind_speed'][i, j])
        
        if not data_points:
            return None
        
        # データ変換
        data_points = np.array(data_points)
        
        try:
            # RBF補間器の作成
            rbf_sin = Rbf(data_points[:, 0], data_points[:, 1], data_points[:, 2], 
                        wind_dir_sin, function='multiquadric')
            rbf_cos = Rbf(data_points[:, 0], data_points[:, 1], data_points[:, 2], 
                         wind_dir_cos, function='multiquadric')
            rbf_speed = Rbf(data_points[:, 0], data_points[:, 1], data_points[:, 2], 
                          wind_speeds, function='multiquadric')
            
            # 出力グリッドの作成
            all_lats = np.concatenate([field['lat_grid'].flatten() for field in self.wind_field_data.values()])
            all_lons = np.concatenate([field['lon_grid'].flatten() for field in self.wind_field_data.values()])
            
            lat_min, lat_max = all_lats.min(), all_lats.max()
            lon_min, lon_max = all_lons.min(), all_lons.max()
            
            lat_grid = np.linspace(lat_min, lat_max, resolution)
            lon_grid = np.linspace(lon_min, lon_max, resolution)
            grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
            
            # 予測時刻（対象時間との差異はゼロ）
            time_grid = np.zeros_like(grid_lats)
            
            # RBFによる予測
            sin_pred = rbf_sin(time_grid, grid_lats, grid_lons)
            cos_pred = rbf_cos(time_grid, grid_lats, grid_lons)
            speed_pred = rbf_speed(time_grid, grid_lats, grid_lons)
            
            # 風向の復元
            wind_directions = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360
            
            # 風速（負の値は0に制限）
            wind_speeds = np.maximum(0, speed_pred)
            
            # 単純な信頼度モデル（時間と距離に基づく）
            min_dist = np.inf
            for t in self.wind_field_data.keys():
                dist = abs((t - target_time).total_seconds() / 60)
                min_dist = min(min_dist, dist)
            
            # 時間距離に基づく信頼度
            time_confidence = max(0.4, 1.0 - min_dist / 30)  # 30分以上離れると0.4
            
            # 空間の平均信頼度
            space_confidence = np.ones_like(grid_lats) * 0.8
            
            # 総合信頼度
            confidence = space_confidence * time_confidence
            
            # 結果の整理
            return {
                'lat_grid': grid_lats,
                'lon_grid': grid_lons,
                'wind_direction': wind_directions,
                'wind_speed': wind_speeds,
                'confidence': confidence,
                'time': target_time
            }
            
        except Exception as e:
            # RBFが失敗した場合はIDWにフォールバック
            print(f"RBF補間に失敗しました: {e}")
            return self._idw_interpolate(target_time, resolution)
    
    def _idw_interpolate(self, target_time: datetime, resolution: int) -> Dict[str, Any]:
        """
        逆距離加重法（IDW）による補間
        
        Parameters:
        -----------
        target_time : datetime
            対象時間
        resolution : int
            出力解像度
            
        Returns:
        --------
        Dict[str, Any]
            補間された風の場
        """
        # 出力グリッドの作成
        all_lats = np.concatenate([field['lat_grid'].flatten() for field in self.wind_field_data.values()])
        all_lons = np.concatenate([field['lon_grid'].flatten() for field in self.wind_field_data.values()])
        
        lat_min, lat_max = all_lats.min(), all_lats.max()
        lon_min, lon_max = all_lons.min(), all_lons.max()
        
        lat_grid = np.linspace(lat_min, lat_max, resolution)
        lon_grid = np.linspace(lon_min, lon_max, resolution)
        grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
        
        # 最も時間的に近い2つのデータを使用
        time_diffs = [(abs((t - target_time).total_seconds()), t) for t in self.wind_field_data.keys()]
        time_diffs.sort()
        
        if len(time_diffs) == 1:
            # 1つしかデータがない場合はそのまま返す
            nearest_time = time_diffs[0][1]
            nearest_field = self.wind_field_data[nearest_time]
            return self._resample_wind_field(nearest_field, resolution)
        
        # 2つの最近傍時間
        t1 = time_diffs[0][1]
        t2 = time_diffs[1][1]
        
        field1 = self.wind_field_data[t1]
        field2 = self.wind_field_data[t2]
        
        # 時間重み付け（線形補間）
        if t1 != t2:
            alpha = abs((target_time - t1).total_seconds()) / abs((t2 - t1).total_seconds())
            if t1 > t2:  # t1が後の時間の場合は重みを反転
                alpha = 1.0 - alpha
        else:
            alpha = 0.5
            
        # 線形補間の制限（0-1）
        alpha = max(0.0, min(1.0, alpha))
        
        # 両方のフィールドをリサンプリング
        field1_resampled = self._resample_wind_field(field1, resolution)
        field2_resampled = self._resample_wind_field(field2, resolution)
        
        # 風向の補間（sin/cosを使用）
        dir1_rad = np.radians(field1_resampled['wind_direction'])
        dir2_rad = np.radians(field2_resampled['wind_direction'])
        
        sin1 = np.sin(dir1_rad)
        cos1 = np.cos(dir1_rad)
        sin2 = np.sin(dir2_rad)
        cos2 = np.cos(dir2_rad)
        
        sin_interp = sin1 * (1 - alpha) + sin2 * alpha
        cos_interp = cos1 * (1 - alpha) + cos2 * alpha
        
        # 風向の復元
        wind_directions = np.degrees(np.arctan2(sin_interp, cos_interp)) % 360
        
        # 風速の線形補間
        wind_speeds = field1_resampled['wind_speed'] * (1 - alpha) + field2_resampled['wind_speed'] * alpha
        
        # 時間トレンドを考慮した調整
        # t1からtarget_timeまでの時間差（分）
        time_diff_minutes = (target_time - t1).total_seconds() / 60
        
        # 風向と風速のトレンド調整
        dir_adjustment = self.wind_dir_trend * time_diff_minutes
        speed_adjustment = self.wind_speed_trend * time_diff_minutes
        
        # 調整を適用
        wind_directions = (wind_directions + dir_adjustment) % 360
        wind_speeds = np.maximum(0, wind_speeds + speed_adjustment)
        
        # 信頼度は時間差に基づいて減衰
        min_time_diff = min(abs((t - target_time).total_seconds()) for t in [t1, t2])
        time_confidence = max(0.5, 1.0 - min_time_diff / (30 * 60))  # 30分で0.5まで減衰
        
        # 基本信頼度を継承
        base_confidence = field1_resampled['confidence'] * (1 - alpha) + field2_resampled['confidence'] * alpha
        
        # 最終信頼度
        confidence = base_confidence * time_confidence
        
        # 結果の整理
        return {
            'lat_grid': grid_lats,
            'lon_grid': grid_lons,
            'wind_direction': wind_directions,
            'wind_speed': wind_speeds,
            'confidence': confidence,
            'time': target_time
        }
    
    def _resample_wind_field(self, wind_field: Dict[str, Any], resolution: int) -> Dict[str, Any]:
        """
        風の場をリサンプリング
        
        Parameters:
        -----------
        wind_field : Dict[str, Any]
            リサンプリングする風の場
        resolution : int
            出力解像度
            
        Returns:
        --------
        Dict[str, Any]
            リサンプリングされた風の場
        """
        # 元のグリッドを取得
        orig_lats = wind_field['lat_grid']
        orig_lons = wind_field['lon_grid']
        orig_dirs = wind_field['wind_direction']
        orig_speeds = wind_field['wind_speed']
        
        if 'confidence' in wind_field:
            orig_conf = wind_field['confidence']
        else:
            orig_conf = np.ones_like(orig_dirs) * 0.8
        
        # 元グリッドと同じ解像度の場合はそのままコピーを返す
        if orig_lats.shape[0] == resolution:
            return wind_field.copy()
        
        # 新しいグリッドを作成
        lat_min, lat_max = orig_lats.min(), orig_lats.max()
        lon_min, lon_max = orig_lons.min(), orig_lons.max()
        
        lat_grid = np.linspace(lat_min, lat_max, resolution)
        lon_grid = np.linspace(lon_min, lon_max, resolution)
        grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
        
        # データを平坦化
        points = np.column_stack([orig_lats.flatten(), orig_lons.flatten()])
        
        # 平坦なデータを準備
        values_dir_sin = np.sin(np.radians(orig_dirs)).flatten()
        values_dir_cos = np.cos(np.radians(orig_dirs)).flatten()
        values_speed = orig_speeds.flatten()
        values_conf = orig_conf.flatten()
        
        try:
            # LinearNDInterpolatorはDelaunay三角形分割を使用
            interp_sin = LinearNDInterpolator(points, values_dir_sin)
            interp_cos = LinearNDInterpolator(points, values_dir_cos)
            interp_speed = LinearNDInterpolator(points, values_speed)
            interp_conf = LinearNDInterpolator(points, values_conf)
            
            # フォールバックのためのNearestNDInterpolator
            nearest_sin = NearestNDInterpolator(points, values_dir_sin)
            nearest_cos = NearestNDInterpolator(points, values_dir_cos)
            nearest_speed = NearestNDInterpolator(points, values_speed)
            nearest_conf = NearestNDInterpolator(points, values_conf)
            
            # 予測点の準備
            xi = np.column_stack([grid_lats.flatten(), grid_lons.flatten()])
            
            # 予測
            sin_pred = interp_sin(xi)
            cos_pred = interp_cos(xi)
            speed_pred = interp_speed(xi)
            conf_pred = interp_conf(xi)
            
            # NaNを近傍値で埋める
            mask = np.isnan(sin_pred)
            sin_pred[mask] = nearest_sin(xi[mask])
            
            mask = np.isnan(cos_pred)
            cos_pred[mask] = nearest_cos(xi[mask])
            
            mask = np.isnan(speed_pred)
            speed_pred[mask] = nearest_speed(xi[mask])
            
            mask = np.isnan(conf_pred)
            conf_pred[mask] = nearest_conf(xi[mask])
            
            # グリッドに変換
            sin_grid = sin_pred.reshape(grid_lats.shape)
            cos_grid = cos_pred.reshape(grid_lats.shape)
            speed_grid = speed_pred.reshape(grid_lats.shape)
            conf_grid = conf_pred.reshape(grid_lats.shape)
            
            # 風向の復元
            dir_grid = np.degrees(np.arctan2(sin_grid, cos_grid)) % 360
            
            # 風速の制限（負の値は0に）
            speed_grid = np.maximum(0, speed_grid)
            
            # 信頼度の制限（0-1の範囲に）
            conf_grid = np.maximum(0, np.minimum(1, conf_grid))
            
        except Exception as e:
            # 補間失敗時は最近傍法を使用
            print(f"リサンプリング補間に失敗しました: {e}")
            
            # 最近傍法
            nearest = NearestNDInterpolator(points, np.arange(len(points)))
            indices = nearest(np.column_stack([grid_lats.flatten(), grid_lons.flatten()]))
            
            sin_grid = np.sin(np.radians(orig_dirs.flatten()[indices])).reshape(grid_lats.shape)
            cos_grid = np.cos(np.radians(orig_dirs.flatten()[indices])).reshape(grid_lats.shape)
            dir_grid = np.degrees(np.arctan2(sin_grid, cos_grid)) % 360
            
            speed_grid = orig_speeds.flatten()[indices].reshape(grid_lats.shape)
            conf_grid = orig_conf.flatten()[indices].reshape(grid_lats.shape)
        
        # 結果の整理
        return {
            'lat_grid': grid_lats,
            'lon_grid': grid_lons,
            'wind_direction': dir_grid,
            'wind_speed': speed_grid,
            'confidence': conf_grid,
            'time': wind_field.get('time', datetime.now())
        }
    
    def create_wind_field_animation(self, start_time: datetime, end_time: datetime, 
                                  time_steps: int = 10, resolution: int = 20) -> List[Dict[str, Any]]:
        """
        時間経過とともに変化する風の場アニメーションを作成
        
        Parameters:
        -----------
        start_time : datetime
            開始時間
        end_time : datetime
            終了時間
        time_steps : int
            時間ステップ数
        resolution : int
            空間解像度
            
        Returns:
        --------
        List[Dict[str, Any]]
            各時間ステップでの風の場データ
        """
        # 時間間隔の計算
        total_seconds = (end_time - start_time).total_seconds()
        interval_seconds = total_seconds / (time_steps - 1) if time_steps > 1 else total_seconds
        
        # 各時間ステップでの風の場を生成
        wind_fields = []
        
        for i in range(time_steps):
            time_point = start_time + timedelta(seconds=i * interval_seconds)
            wind_field = self.interpolate_wind_field(time_point, resolution)
            
            if wind_field is not None:
                wind_fields.append(wind_field)
        
        return wind_fields
    
    def adjust_wind_field_with_observations(self, wind_field: Dict[str, Any], 
                                         observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        観測データに基づいて風の場を調整
        
        Parameters:
        -----------
        wind_field : Dict[str, Any]
            調整する風の場
        observations : List[Dict[str, Any]]
            観測データのリスト（例: 気象ブイやレース委員会の観測）
            
        Returns:
        --------
        Dict[str, Any]
            調整された風の場
        """
        if not observations:
            return wind_field
        
        # 観測データの形式：
        # {'latitude': float, 'longitude': float, 'wind_direction': float, 'wind_speed': float, 'confidence': float}
        
        # 風の場をコピー
        adjusted_field = {
            'lat_grid': wind_field['lat_grid'].copy(),
            'lon_grid': wind_field['lon_grid'].copy(),
            'wind_direction': wind_field['wind_direction'].copy(),
            'wind_speed': wind_field['wind_speed'].copy(),
            'confidence': wind_field['confidence'].copy(),
            'time': wind_field.get('time', datetime.now())
        }
        
        # 観測データを処理
        for obs in observations:
            # 観測位置
            obs_lat = obs['latitude']
            obs_lon = obs['longitude']
            
            # 観測風向風速
            obs_dir = obs['wind_direction']
            obs_speed = obs['wind_speed']
            obs_conf = obs.get('confidence', 0.9)  # デフォルト信頼度は0.9
            
            # グリッド上の最も近い点を見つける
            distances = (adjusted_field['lat_grid'] - obs_lat)**2 + (adjusted_field['lon_grid'] - obs_lon)**2
            closest_idx = np.unravel_index(np.argmin(distances), distances.shape)
            
            # 観測地点の影響範囲（ガウス分布で重み付け）
            influence_range = 0.01  # 度単位（約1.1km）
            
            # 距離に基づく重みを計算
            weights = np.exp(-distances / (2 * influence_range**2))
            weights = weights / np.max(weights)  # 0-1に正規化
            
            # 風向の調整（sin/cosを使用）
            orig_dir_rad = np.radians(adjusted_field['wind_direction'])
            orig_sin = np.sin(orig_dir_rad)
            orig_cos = np.cos(orig_dir_rad)
            
            obs_dir_rad = math.radians(obs_dir)
            obs_sin = math.sin(obs_dir_rad)
            obs_cos = math.cos(obs_dir_rad)
            
            # 観測との違いを計算
            diff_sin = obs_sin - orig_sin[closest_idx]
            diff_cos = obs_cos - orig_cos[closest_idx]
            
            # 重みに基づいて調整
            adjusted_sin = orig_sin + weights * diff_sin * obs_conf
            adjusted_cos = orig_cos + weights * diff_cos * obs_conf
            
            # 新しい風向に変換
            adjusted_field['wind_direction'] = np.degrees(np.arctan2(adjusted_sin, adjusted_cos)) % 360
            
            # 風速の調整も同様
            speed_diff = obs_speed - adjusted_field['wind_speed'][closest_idx]
            adjusted_field['wind_speed'] += weights * speed_diff * obs_conf
            
            # 負の風速を制限
            adjusted_field['wind_speed'] = np.maximum(0, adjusted_field['wind_speed'])
            
            # 信頼度も更新
            adjusted_field['confidence'] = np.maximum(adjusted_field['confidence'], weights * obs_conf)
        
        return adjusted_field
    
    def visualize_wind_field(self, wind_field: Dict[str, Any], 
                           show_confidence: bool = False, 
                           title: str = None,
                           save_path: str = None) -> plt.Figure:
        """
        風の場をプロットして可視化
        
        Parameters:
        -----------
        wind_field : Dict[str, Any]
            風の場データ
        show_confidence : bool
            信頼度を表示するかどうか
        title : str, optional
            図のタイトル
        save_path : str, optional
            保存先パス
            
        Returns:
        --------
        plt.Figure
            プロット図
        """
        grid_lats = wind_field['lat_grid']
        grid_lons = wind_field['lon_grid']
        wind_directions = wind_field['wind_direction']
        wind_speeds = wind_field['wind_speed']
        confidence = wind_field.get('confidence', np.ones_like(wind_directions) * 0.8)
        
        # プロットサイズの調整
        plt.figure(figsize=(12, 10))
        
        # 風速をカラーマップで表示
        plt.contourf(grid_lons, grid_lats, wind_speeds, cmap='viridis', levels=20)
        plt.colorbar(label='Wind Speed (knots)')
        
        # 風向を矢印で表示
        # グリッドが大きい場合は間引く
        skip = max(1, grid_lats.shape[0] // 20)
        
        # 矢印の向きを風向から180度反転（風が吹いてくる方向に）
        u = -np.sin(np.radians(wind_directions[::skip, ::skip]))
        v = -np.cos(np.radians(wind_directions[::skip, ::skip]))
        
        # 信頼度に基づいて矢印の透明度を設定
        if show_confidence:
            alpha = confidence[::skip, ::skip]
        else:
            alpha = np.ones_like(u) * 0.8
        
        # 矢印プロット
        plt.quiver(grid_lons[::skip, ::skip], grid_lats[::skip, ::skip], u, v, 
                 alpha=alpha, color='black', scale=30)
        
        # 信頼度のオーバーレイ表示（オプション）
        if show_confidence:
            plt.contour(grid_lons, grid_lats, confidence, 
                      levels=[0.3, 0.5, 0.7, 0.9], colors='red', alpha=0.5, linestyles='dashed')
        
        # タイトルを設定
        if title is None and 'time' in wind_field:
            title = f"Wind Field at {wind_field['time'].strftime('%Y-%m-%d %H:%M:%S')}"
        
        if title:
            plt.title(title)
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 保存処理
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
        return plt.gcf()  # 現在の図を返す
    
    def visualize_wind_field_3d(self, wind_field: Dict[str, Any], 
                              show_confidence: bool = True,
                              title: str = None,
                              save_path: str = None) -> plt.Figure:
        """
        風の場を3Dプロットで可視化
        
        Parameters:
        -----------
        wind_field : Dict[str, Any]
            風の場データ
        show_confidence : bool
            信頼度を表示するかどうか
        title : str, optional
            図のタイトル
        save_path : str, optional
            保存先パス
            
        Returns:
        --------
        plt.Figure
            プロット図
        """
        grid_lats = wind_field['lat_grid']
        grid_lons = wind_field['lon_grid']
        wind_speeds = wind_field['wind_speed']
        confidence = wind_field.get('confidence', np.ones_like(wind_speeds) * 0.8)
        
        # 3Dプロットの作成
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 色マップを風速に基づいて設定
        norm = Normalize(vmin=wind_speeds.min(), vmax=wind_speeds.max())
        cmap = plt.cm.viridis
        
        # 地形光源効果
        ls = LightSource(azdeg=315, altdeg=45)
        
        # 信頼度を高さとしてプロット
        if show_confidence:
            # 信頼度を風速のスケールに合わせる
            z_scale = wind_speeds.max() / confidence.max() if confidence.max() > 0 else 1.0
            z_values = confidence * z_scale
            
            # Z軸ラベル
            z_label = 'Confidence (scaled)'
        else:
            # 風速自体を高さに
            z_values = wind_speeds
            z_label = 'Wind Speed (knots)'
        
        # サーフェスプロット
        surf = ax.plot_surface(grid_lons, grid_lats, z_values, 
                             rstride=1, cstride=1, 
                             facecolors=cmap(norm(wind_speeds)), 
                             linewidth=0, antialiased=True, shade=True,
                             alpha=0.8)
        
        # 風速用カラーバー
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)
        cbar.set_label('Wind Speed (knots)')
        
        # タイトル設定
        if title is None and 'time' in wind_field:
            title = f"3D Wind Field at {wind_field['time'].strftime('%Y-%m-%d %H:%M:%S')}"
        
        if title:
            ax.set_title(title)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel(z_label)
        
        # 視点の設定
        ax.view_init(elev=35, azim=45)
        
        # 等高線をベース面に追加
        offset = z_values.min() - z_values.max() * 0.1
        cont = ax.contourf(grid_lons, grid_lats, wind_speeds, 
                         zdir='z', offset=offset, 
                         cmap='viridis', alpha=0.5, levels=15)
        
        # Z軸の範囲を調整
        ax.set_zlim(offset, z_values.max() * 1.1)
        
        # 保存処理
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
        return fig

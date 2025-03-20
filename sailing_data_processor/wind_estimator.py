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

    def estimate_wind_direction_hybrid(self, df: pd.DataFrame, boat_type: str = 'default') -> Tuple[float, float]:
        """
        ハイブリッド風向推定アルゴリズム - 複数アプローチの組み合わせ
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ（少なくとも'bearing'と'speed'列が必要）
        boat_type : str
            艇種識別子
            
        Returns:
        --------
        Tuple[float, float]
            (推定風向（度、0-360の範囲）, 信頼度スコア(0-1の範囲))
        """
        if df is None or len(df) < 10:  # 最低限のデータポイント要件
            return 0.0, 0.0
        
        # データ品質の評価
        data_quality = self._evaluate_data_quality(df)
        
        # 方位の分布範囲を評価
        bearing_range = self._evaluate_bearing_range(df)
        
        # 結果を格納する辞書
        estimates = {}
        
        # 1. 速度パターン分析による推定
        direction_speed, confidence_speed = self._estimate_from_speed_patterns(df)
        estimates['speed_patterns'] = (direction_speed, confidence_speed)
        
        # 2. ポーラーデータを用いた推定（データが利用可能な場合）
        direction_polar, confidence_polar = self._estimate_using_polar(df, boat_type)
        if confidence_polar > 0:
            estimates['polar_data'] = (direction_polar, confidence_polar)
        
        # 3. 最適VMG角度に基づく推定
        direction_vmg, confidence_vmg = self._estimate_from_optimal_vmg(df, boat_type)
        estimates['optimal_vmg'] = (direction_vmg, confidence_vmg)
        
        # 各推定結果の信頼度に応じた重み付け平均を計算
        if estimates:
            final_direction = self._weighted_angle_consensus(estimates)
            
            # 最終的な信頼度は各推定法の最大信頼度と平均信頼度の加重平均
            max_confidence = max([conf for _, conf in estimates.values()])
            avg_confidence = sum([conf for _, conf in estimates.values()]) / len(estimates)
            final_confidence = 0.7 * max_confidence + 0.3 * avg_confidence
            
            # データ品質による信頼度の調整
            final_confidence = final_confidence * data_quality
            
            return final_direction, final_confidence
        
        # 推定失敗
        return 0.0, 0.0

    def _evaluate_data_quality(self, df: pd.DataFrame) -> float:
        """
        データの品質を評価
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ
            
        Returns:
        --------
        float
            品質スコア（0-1の範囲）
        """
        # 十分なデータポイントがあるか
        min_required_points = 10
        points_score = min(1.0, len(df) / min_required_points)
        
        # 異常値や欠損値の検出
        missing_data = df[['bearing', 'speed']].isna().any(axis=1).mean()
        missing_score = 1.0 - missing_data
        
        # 速度の一貫性をチェック（極端な変化がないか）
        if 'speed' in df.columns and len(df) > 2:
            speed_std = df['speed'].std()
            speed_mean = df['speed'].mean() if df['speed'].mean() > 0 else 1.0
            speed_variation = speed_std / speed_mean
            
            # 変動係数が0.5以下なら高品質（1.0）、1.0以上なら低品質（0.5）とする
            speed_score = max(0.5, min(1.0, 1.5 - speed_variation))
        else:
            speed_score = 0.8  # デフォルト値
        
        # タイムスタンプの連続性をチェック（利用可能な場合）
        if 'timestamp' in df.columns and len(df) > 2:
            # 時間差分を計算
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff().dt.total_seconds()
            
            if not time_diffs.isna().all():
                # 大きなギャップ（5秒以上）の比率
                large_gaps = (time_diffs > 5).mean()
                time_score = 1.0 - large_gaps
            else:
                time_score = 0.8  # デフォルト値
        else:
            time_score = 0.8  # デフォルト値
        
        # 総合スコアの計算（各要素に重み付け）
        quality_score = (
            0.3 * points_score +
            0.2 * missing_score +
            0.3 * speed_score +
            0.2 * time_score
        )
        
        return max(0.1, min(1.0, quality_score))  # 0.1-1.0の範囲に制限

    def _evaluate_bearing_range(self, df: pd.DataFrame) -> float:
        """
        方位の分布範囲を評価
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ
            
        Returns:
        --------
        float
            方位の範囲（度、0-180の範囲）
        """
        if 'bearing' not in df.columns or len(df) < 2:
            return 0.0
        
        # 有効な方位データを抽出
        bearings = df['bearing'].dropna().values
        
        if len(bearings) < 2:
            return 0.0
        
        # 正弦・余弦成分に変換して循環性を考慮
        sin_vals = np.sin(np.radians(bearings))
        cos_vals = np.cos(np.radians(bearings))
        
        # 主方向の計算
        mean_sin = sin_vals.mean()
        mean_cos = cos_vals.mean()
        main_bearing = (np.degrees(np.arctan2(mean_sin, mean_cos)) + 360) % 360
        
        # 各方位と主方向との差を計算（-180〜180度の範囲）
        angle_diffs = np.abs((bearings - main_bearing + 180) % 360 - 180)
        
        # 95パーセンタイルを範囲として使用
        bearing_range = np.percentile(angle_diffs, 95) * 2  # 片側→両側の範囲
        
        return min(180.0, bearing_range)  # 最大180度に制限

    def _estimate_from_speed_patterns(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        速度パターン分析による風向推定
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ
            
        Returns:
        --------
        Tuple[float, float]
            (推定風向, 信頼度)
        """
        if 'bearing' not in df.columns or 'speed' not in df.columns or len(df) < 5:
            return 0.0, 0.0
        
        # 方位角を36ビン（10度刻み）に分類
        num_bins = 36
        bins = np.linspace(0, 360, num_bins + 1)
        bin_centers = bins[:-1] + 5  # 各ビンの中央値
        
        # 各方位ビンごとの平均速度を計算
        bin_speeds = []
        bin_counts = []
        
        for i in range(num_bins):
            # ビンの範囲（循環性を考慮）
            lower = bins[i]
            upper = bins[i+1]
            
            if lower <= upper:
                mask = (df['bearing'] >= lower) & (df['bearing'] < upper)
            else:  # 0度をまたぐ場合
                mask = (df['bearing'] >= lower) | (df['bearing'] < upper)
            
            bin_data = df[mask]
            bin_counts.append(len(bin_data))
            
            if len(bin_data) > 0:
                bin_speeds.append(bin_data['speed'].mean())
            else:
                bin_speeds.append(0)
        
        # データポイントが十分あるビンのみを対象
        valid_bins = np.array(bin_counts) >= max(2, len(df) * 0.05)
        
        if not any(valid_bins):
            return 0.0, 0.0
        
        # 有効なビンの方位と速度
        valid_bin_centers = bin_centers[valid_bins]
        valid_bin_speeds = np.array(bin_speeds)[valid_bins]
        
        # 最低速度と最高速度を特定
        if len(valid_bin_speeds) < 2:
            return 0.0, 0.0
        
        min_speed_idx = np.argmin(valid_bin_speeds)
        max_speed_idx = np.argmax(valid_bin_speeds)
        
        # 風上（最も遅い）と風下（最も速い）の方位
        upwind_bearing = valid_bin_centers[min_speed_idx]
        downwind_bearing = valid_bin_centers[max_speed_idx]
        
        # 風向の推定（風上の反対側）
        wind_direction = (upwind_bearing + 180) % 360
        
        # 風上と風下の角度差（風向推定の信頼性指標）
        angle_diff = abs((upwind_bearing - downwind_bearing + 180) % 360 - 180)
        
        # 角度差が約180度に近いほど信頼性が高い
        angle_confidence = 1.0 - min(1.0, abs(angle_diff - 180) / 90)
        
        # 速度比に基づく信頼度
        min_speed = valid_bin_speeds[min_speed_idx]
        max_speed = valid_bin_speeds[max_speed_idx]
        
        # 最大速度が0に近い場合の対処
        if max_speed < 0.1:
            speed_ratio = 0
        else:
            speed_ratio = min_speed / max_speed
        
        # 風上と風下の速度比（理想的には0.5前後）
        speed_confidence = 1.0 - min(1.0, abs(speed_ratio - 0.5) / 0.5)
        
        # 総合信頼度（角度差と速度比の両方を考慮）
        confidence = 0.6 * angle_confidence + 0.4 * speed_confidence
        
        # 極めて少ないデータポイントの場合は信頼度を下げる
        if len(df) < 20:
            confidence *= max(0.5, len(df) / 20)
        
        return wind_direction, confidence

    def _get_polar_data(self, boat_type: str) -> Optional[pd.DataFrame]:
        """
        ポーラーデータを取得または生成する
        
        Parameters:
        -----------
        boat_type : str
            艇種識別子
            
        Returns:
        --------
        pd.DataFrame or None
            風向角(行)と風速(列)ごとの艇速データフレーム
        """
        try:
            # OptimalVMGCalculatorなどの外部クラスからポーラーデータを取得する処理
            # この例では仮にこのようなアクセスが可能だと想定
            import sys
            import os
            
            # 上位モジュールの検索
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            
            try:
                # OptimalVMGCalculator からポーラーデータへのアクセスを試みる
                from optimal_vmg_calculator import OptimalVMGCalculator
                
                vmg_calculator = OptimalVMGCalculator()
                
                # 艇種が存在するか確認
                if boat_type.lower() in vmg_calculator.boat_types:
                    return vmg_calculator.boat_types[boat_type.lower()]['polar_data']
                else:
                    # 艇種が見つからない場合はデフォルトを使用
                    if 'default' in vmg_calculator.boat_types:
                        return vmg_calculator.boat_types['default']['polar_data']
            except (ImportError, AttributeError):
                # インポートエラーやアクセスエラーの場合は簡易ポーラーデータを生成
                pass
        except Exception:
            # その他のエラー時も簡易ポーラーデータにフォールバック
            pass
        
        # ポーラーデータが取得できない場合は簡易ポーラーデータを生成
        return self._generate_simplified_polar(boat_type)

    def _generate_simplified_polar(self, boat_type: str) -> pd.DataFrame:
        """
        簡易的なポーラーデータを生成
        
        Parameters:
        -----------
        boat_type : str
            艇種識別子
            
        Returns:
        --------
        pd.DataFrame
            風向角(行)と風速(列)ごとの艇速データフレーム
        """
        # 使用する係数の決定
        use_boat_type = boat_type.lower() if boat_type.lower() in self.boat_coefficients else 'default'
        upwind_ratio = self.boat_coefficients[use_boat_type]['upwind']
        downwind_ratio = self.boat_coefficients[use_boat_type]['downwind']
        
        # 風向角（TWA）の値（0-180度）
        twa_values = np.array([0, 30, 40, 45, 52, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
        
        # 風速（TWS）の値（ノット）
        tws_values = np.array([4, 6, 8, 10, 12, 14, 16, 20])
        
        # 空のデータフレームを作成
        polar_data = pd.DataFrame(index=twa_values, columns=tws_values)
        
        # 係数に基づいた簡易ポーラーデータの生成
        for twa in twa_values:
            for tws in tws_values:
                # 風向角に応じた基本速度係数
                if twa == 0:
                    # 風に向かって真っ直ぐには進めない
                    boat_speed = 0.0
                elif twa <= 45:
                    # クローズホールド（風上向き）
                    ratio = 1.0 / upwind_ratio
                    # 45度からの角度による調整
                    angle_factor = 0.7 + 0.3 * (twa / 45.0)
                    boat_speed = tws * ratio * angle_factor
                elif twa <= 90:
                    # リーチング
                    # 風上と風下の間の線形補間
                    upwind_speed = tws / upwind_ratio
                    downwind_speed = tws / downwind_ratio
                    blend = (twa - 45) / 45.0  # 0-1の範囲
                    boat_speed = upwind_speed * (1 - blend) + downwind_speed * blend
                else:
                    # ランニング（風下向き）
                    ratio = 1.0 / downwind_ratio
                    # 風下では風速の2/3乗に比例
                    boat_speed = tws * ratio * (tws / 10.0) ** 0.1
                
                # データフレームに格納
                polar_data.loc[twa, tws] = boat_speed
        
        return polar_data

    def _estimate_using_polar(self, df: pd.DataFrame, boat_type: str) -> Tuple[float, float]:
        """
        ポーラーデータを用いた風向推定
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ
        boat_type : str
            艇種識別子
            
        Returns:
        --------
        Tuple[float, float]
            (推定風向, 信頼度)
        """
        if 'bearing' not in df.columns or 'speed' not in df.columns or len(df) < 5:
            return 0.0, 0.0
        
        # ポーラーデータの取得
        polar_data = self._get_polar_data(boat_type)
        if polar_data is None or polar_data.empty:
            return 0.0, 0.0
        
        # 風速の仮定値（後でより正確に推定）
        assumed_wind_speed = 10.0  # 仮の風速10ノット
        
        # 方位を36ビン（10度刻み）に分類
        num_bins = 36
        bins = np.linspace(0, 360, num_bins + 1)
        bin_centers = bins[:-1] + 5  # 各ビンの中央値
        
        # 各方位ビンごとの平均速度を計算
        bin_bearings = []
        bin_speeds = []
        bin_counts = []
        
        # 有効なデータポイント
        valid_data = df.dropna(subset=['bearing', 'speed'])
        
        for i in range(num_bins):
            # ビンの範囲（循環性を考慮）
            lower = bins[i]
            upper = bins[i+1]
            
            if lower <= upper:
                mask = (valid_data['bearing'] >= lower) & (valid_data['bearing'] < upper)
            else:  # 0度をまたぐ場合
                mask = (valid_data['bearing'] >= lower) | (valid_data['bearing'] < upper)
            
            bin_data = valid_data[mask]
            bin_counts.append(len(bin_data))
            
            if len(bin_data) > 0:
                bin_bearings.append(bin_centers[i])
                bin_speeds.append(bin_data['speed'].mean())
        
        if not bin_bearings:
            return 0.0, 0.0
        
        # 各可能な風向に対してポーラー一致度を評価
        best_score = -1
        best_wind_direction = 0
        
        # 10度ごとに可能な風向をスキャン（36方位）
        for test_wind_dir in bin_centers:
            score = 0
            valid_comparisons = 0
            
            for bearing, speed in zip(bin_bearings, bin_speeds):
                # 相対風向（TWA）を計算
                # TWAは艇の進行方向と風向の差（0-180度）
                twa = abs((bearing - test_wind_dir + 180) % 360 - 180)
                
                # ポーラーデータから期待される速度を取得
                expected_speed = self._get_expected_speed_from_polar(
                    polar_data, twa, assumed_wind_speed)
                
                if expected_speed > 0:
                    # 観測速度と期待速度の比率
                    speed_ratio = speed / expected_speed
                    
                    # 比率が1に近いほど良いスコア
                    comparison_score = 1.0 - min(1.0, abs(speed_ratio - 1.0))
                    score += comparison_score
                    valid_comp

    def _get_expected_speed_from_polar(self, polar_data: pd.DataFrame, twa: float, tws: float) -> float:
        """
        ポーラーデータから特定のTWA（風向角）とTWS（風速）での期待される艇速を取得
        
        Parameters:
        -----------
        polar_data : pd.DataFrame
            ポーラーデータ
        twa : float
            風向角（度、0-180）
        tws : float
            風速（ノット）
            
        Returns:
        --------
        float
            期待される艇速（ノット）
        """
        # 風向角を0-180度の範囲に制限
        twa = min(180, max(0, twa))
        
        # ポーラーデータのインデックス（TWA）と列（TWS）
        twa_index = polar_data.index.values
        tws_columns = polar_data.columns.values
        
        # 最も近いTWA値を見つける
        twa_lower_idx = np.searchsorted(twa_index, twa) - 1
        if twa_lower_idx < 0:
            twa_lower_idx = 0
        if twa_lower_idx >= len(twa_index) - 1:
            twa_lower_idx = len(twa_index) - 2
        
        twa_lower = twa_index[twa_lower_idx]
        twa_upper = twa_index[twa_lower_idx + 1]
        
        # 最も近いTWS値を見つける
        tws_lower_idx = np.searchsorted(tws_columns, tws) - 1
        if tws_lower_idx < 0:
            tws_lower_idx = 0
        if tws_lower_idx >= len(tws_columns) - 1:
            tws_lower_idx = len(tws_columns) - 2
        
        tws_lower = tws_columns[tws_lower_idx]
        tws_upper = tws_columns[tws_lower_idx + 1]
        
        # 4点の値を取得
        val_ll = polar_data.iloc[twa_lower_idx, tws_lower_idx]  # 左下
        val_lu = polar_data.iloc[twa_lower_idx, tws_lower_idx + 1]  # 左上
        val_rl = polar_data.iloc[twa_lower_idx + 1, tws_lower_idx]  # 右下
        val_ru = polar_data.iloc[twa_lower_idx + 1, tws_lower_idx + 1]  # 右上
        
        # TWA方向の補間率
        if twa_upper == twa_lower:
            twa_ratio = 0
        else:
            twa_ratio = (twa - twa_lower) / (twa_upper - twa_lower)
        
        # TWS方向の補間率
        if tws_upper == tws_lower:
            tws_ratio = 0
        else:
            tws_ratio = (tws - tws_lower) / (tws_upper - tws_lower)
        
        # 双線形補間
        val_l = val_ll * (1 - tws_ratio) + val_lu * tws_ratio  # 左側の補間
        val_r = val_rl * (1 - tws_ratio) + val_ru * tws_ratio  # 右側の補間
        
        val = val_l * (1 - twa_ratio) + val_r * twa_ratio  # 最終補間値
        
        return val

    def _estimate_from_optimal_vmg(self, df: pd.DataFrame, boat_type: str) -> Tuple[float, float]:
        """
        クローズホールドの最適角度に基づく風向推定
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ
        boat_type : str
            艇種識別子
            
        Returns:
        --------
        Tuple[float, float]
            (推定風向, 信頼度)
        """
        if 'bearing' not in df.columns or len(df) < 5:
            return 0.0, 0.0
        
        # 艇種から最適風上角度を取得
        use_boat_type = boat_type.lower() if boat_type.lower() in self.boat_coefficients else 'default'
        upwind_ratio = self.boat_coefficients[use_boat_type]['upwind']
        
        # 艇種に基づく最適風上角度の推定
        # 一般に、性能が高い艇ほど風上に切り込める
        optimal_upwind_angle = max(30, min(50, 45 - (upwind_ratio - 3.0) * 5))
        
        # 方位の分布を分析
        bearings = df['bearing'].dropna().values
        
        if len(bearings) < 5:
            return 0.0, 0.0
        
        # KMeansクラスタリングを使用して主要な航行方向を特定
        # 角度の循環性を考慮するため、sin/cosに変換
        X = np.column_stack([
            np.sin(np.radians(bearings)),
            np.cos(np.radians(bearings))
        ])
        
        # クラスタ数の決定（データポイント数に基づく）
        n_clusters = min(4, max(2, len(bearings) // 10))
        
        # KMeansを実行
        from sklearn.cluster import KMeans
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X)
            
            # 各クラスタの中心点を方位角に変換
            cluster_centers_sin_cos = kmeans.cluster_centers_
            cluster_angles = np.degrees(
                np.arctan2(
                    cluster_centers_sin_cos[:, 0],
                    cluster_centers_sin_cos[:, 1]
                )
            ) % 360
            
            # 各クラスタのサイズ
            cluster_sizes = np.bincount(kmeans.labels_)
            
            # 最も人口の多い2つのクラスタを選択
            top_clusters_idx = np.argsort(-cluster_sizes)[:2]
            top_cluster_angles = cluster_angles[top_clusters_idx]
            
            # 2つのクラスタ間の角度差
            if len(top_cluster_angles) >= 2:
                angle_diff = abs((top_cluster_angles[0] - top_cluster_angles[1] + 180) % 360 - 180)
                
                # クラスタ間の角度差が70-110度なら対称的なタックと判断
                if 70 <= angle_diff <= 110:
                    # 二等分線を計算（風上方向）
                    bisector = (top_cluster_angles[0] + top_cluster_angles[1]) / 2
                    if abs(top_cluster_angles[0] - top_cluster_angles[1]) > 180:
                        # 0度線を跨ぐ場合の補正
                        bisector = (bisector + 180) % 360
                    
                    # 最適風上角度を考慮した風向の推定
                    wind_direction = (bisector + 180 - optimal_upwind_angle) % 360
                    
                    # 信頼度はクラスタのサイズと角度差に基づく
                    angle_confidence = 1.0 - abs(angle_diff - 90) / 90  # 90度で最大
                    
                    # クラスタサイズの合計が全体に占める割合
                    size_ratio = sum(cluster_sizes[top_clusters_idx]) / len(bearings)
                    size_confidence = min(1.0, size_ratio * 2)  # 50%以上で最大
                    
                    confidence = 0.6 * angle_confidence + 0.4 * size_confidence
                    
                    return wind_direction, min(0.8, confidence)  # 最大0.8の信頼度
                
                # そうでなければ単一方向からの推定
                # 最大クラスタから推定
                main_angle = top_cluster_angles[0]
                wind_direction = (main_angle + 180 - optimal_upwind_angle) % 360
                
                # 信頼度は低め
                confidence = 0.3 * min(1.0, cluster_sizes[top_clusters_idx[0]] / (len(bearings) / 2))
                
                return wind_direction, confidence
            
        except Exception:
            # KMeans失敗時は方位のヒストグラムを使用
            pass
        
        # フォールバック: 方位ヒストグラム分析
        hist, bin_edges = np.histogram(bearings, bins=36, range=(0, 360))
        
        # 上位2つのピークを見つける
        peak_indices = np.argsort(-hist)[:2]
        
        if len(peak_indices) >= 2:
            peak_angles = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in peak_indices]
            
            # ピーク間の角度差
            angle_diff = abs((peak_angles[0] - peak_angles[1] + 180) % 360 - 180)
            
            # 角度差が70-110度なら対称的なタックと判断
            if 70 <= angle_diff <= 110:
                # 風上方向の推定（ピークの二等分線）
                bisector = (peak_angles[0] + peak_angles[1]) / 2
                if abs(peak_angles[0] - peak_angles[1]) > 180:
                    # 0度線を跨ぐ場合の補正
                    bisector = (bisector + 180) % 360
                
                # 風向の推定
                wind_direction = (bisector + 180 - optimal_upwind_angle) % 360
                
                # ピークの強さに基づく信頼度
                peak_strength = (hist[peak_indices[0]] + hist[peak_indices[1]]) / len(bearings)
                confidence = min(0.6, peak_strength)
                
                return wind_direction, confidence
        
        # 単一ピークからの推定（信頼度低）
        if len(peak_indices) > 0:
            main_angle = (bin_edges[peak_indices[0]] + bin_edges[peak_indices[0]+1]) / 2
            wind_direction = (main_angle + 180 - optimal_upwind_angle) % 360
            
            # 信頼度は低め
            confidence = 0.3 * min(1.0, hist[peak_indices[0]] / (len(bearings) / 3))
            
            return wind_direction, confidence
        
        # 推定失敗
        return 0.0, 0.0

    def _weighted_angle_consensus(self, results_dict: Dict[str, Tuple[float, float]]) -> float:
        """
        複数推定結果から重み付けされた最終風向を算出
        
        Parameters:
        -----------
        results_dict : Dict[str, Tuple[float, float]]
            推定手法: (風向, 信頼度) の辞書
            
        Returns:
        --------
        float
            合意形成された風向（度）
        """
        if not results_dict:
            return 0.0
        
        # 風向と重み（信頼度）のリスト
        directions = []
        confidences = []
        
        for method, (direction, confidence) in results_dict.items():
            if confidence > 0:
                directions.append(direction)
                confidences.append(confidence)
        
        if not directions:
            return 0.0
        
        # 正弦・余弦成分の重み付き平均
        sin_sum = 0
        cos_sum = 0
        weight_sum = 0
        
        for direction, weight in zip(directions, confidences):
            angle_rad = math.radians(direction)
            sin_sum += math.sin(angle_rad) * weight
            cos_sum += math.cos(angle_rad) * weight
            weight_sum += weight
        
        # 重み総和が0の場合の対策
        if weight_sum <= 0:
            return directions[0]  # 最初の方向を返す
        
        # 正弦・余弦から角度に戻す
        consensus_direction = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
        
        return consensus_direction

    def _get_main_bearing(self, df: pd.DataFrame) -> float:
        """
        主航行方位を特定
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ
            
        Returns:
        --------
        float
            主方位（度）
        """
        if 'bearing' not in df.columns or len(df) < 2:
            return 0.0
        
        # 有効な方位データを抽出
        bearings = df['bearing'].dropna().values
        
        if len(bearings) < 2:
            return 0.0
        
        # 正弦・余弦成分に変換して循環性を考慮
        sin_vals = np.sin(np.radians(bearings))
        cos_vals = np.cos(np.radians(bearings))
        
        # 主方向の計算
        mean_sin = sin_vals.mean()
        mean_cos = cos_vals.mean()
        main_bearing = (np.degrees(np.arctan2(mean_sin, mean_cos)) + 360) % 360
        
        return main_bearing

    def _detect_tacks_improved(self, df: pd.DataFrame, min_tack_angle: float = 30.0, 
                         window_size: int = 3) -> pd.DataFrame:
        """
        後方互換性のためのラッパーメソッド
        
        Note:
        -----
        このメソッドは次期バージョンでは非推奨となり、完全に detect_maneuvers に置き換えられる予定
        
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
        # 仮の風向を計算（主航行方向から180度反転）
        provisional_wind_dir = (self._get_main_bearing(df) + 180) % 360
        
        # detect_maneuversを呼び出し、結果からmaneuver_type列を削除
        maneuvers = self.detect_maneuvers(df, provisional_wind_dir, min_tack_angle, window_size)
        if not maneuvers.empty and 'maneuver_type' in maneuvers.columns:
            maneuvers = maneuvers.drop(columns=['maneuver_type'])
        
        return maneuvers

    def detect_maneuvers(self, df: pd.DataFrame, wind_direction: float, 
                    min_angle_change: float = 30.0, window_size: int = 3) -> pd.DataFrame:
        """
        方向転換を検出し、風向を基にタックとジャイブを区別する
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ
        wind_direction : float
            推定風向（度）
        min_angle_change : float
            方向転換と判定する最小角度変化
        window_size : int
            移動ウィンドウサイズ
            
        Returns:
        --------
        pd.DataFrame
            検出された方向転換とその分類（タック/ジャイブ）
        """
        if df is None or len(df) < window_size * 2:
            return pd.DataFrame()  # 十分なデータがない場合は空のデータフレームを返す
        
        # 方位角の変化を計算
        df_with_changes = self._calculate_bearing_change(df)
        
        # 移動ウィンドウでの方位変化の計算
        df_copy = df_with_changes.copy()
        df_copy['bearing_change_sum'] = df_copy['bearing_change'].rolling(window=window_size, center=True).sum()
        
        # 方向転換の検出（移動ウィンドウ内の累積変化がmin_angle_changeを超える場合）
        df_copy['is_maneuver'] = df_copy['bearing_change_sum'] > min_angle_change
        
        # 連続する方向転換を1つのイベントとしてグループ化
        df_copy['maneuver_group'] = (df_copy['is_maneuver'] != df_copy['is_maneuver'].shift(1)).cumsum()
        
        # 方向転換グループごとに最大の方位変化点を見つける
        maneuver_points = []
        
        for group_id, group in df_copy[df_copy['is_maneuver']].groupby('maneuver_group'):
            if len(group) > 0:
                # グループ内で最大の方位変化がある点を代表点として選択
                max_change_idx = group['bearing_change'].idxmax()
                maneuver = df_copy.loc[max_change_idx].copy()
                
                # 方向転換の前後の方位を取得
                before_idx = max(0, max_change_idx - window_size)
                after_idx = min(len(df_copy) - 1, max_change_idx + window_size)
                
                before_bearing = df_copy.loc[before_idx, 'bearing']
                after_bearing = df_copy.loc[after_idx, 'bearing']
                
                # 方向転換のタイプを分類
                maneuver_type = self._categorize_maneuver(
                    before_bearing, after_bearing, wind_direction)
                
                # 分類結果を追加
                maneuver['maneuver_type'] = maneuver_type
                maneuver['before_bearing'] = before_bearing
                maneuver['after_bearing'] = after_bearing
                
                maneuver_points.append(maneuver)
        
        # 方向転換ポイントのデータフレームを作成
        if maneuver_points:
            return pd.DataFrame(maneuver_points)
        else:
            return pd.DataFrame()

    def _categorize_maneuver(self, before_bearing: float, after_bearing: float, 
                        wind_direction: float) -> str:
        """
        方向転換のタイプを風向に基づいて分類
        
        Parameters:
        -----------
        before_bearing : float
            転換前の艇の方位
        after_bearing : float
            転換後の艇の方位
        wind_direction : float
            風向（風が吹いてくる方向）
            
        Returns:
        --------
        str
            'tack', 'jibe', 'bear_away', 'head_up' いずれかのタイプ
        """
        # 風向に対する相対角度を計算
        rel_before = (before_bearing - wind_direction + 360) % 360
        rel_after = (after_bearing - wind_direction + 360) % 360
        
        # 風上側か風下側か判定
        is_upwind_before = 0 <= rel_before <= 90 or 270 <= rel_before <= 360
        is_upwind_after = 0 <= rel_after <= 90 or 270 <= rel_after <= 360
        
        # マニューバータイプを判別
        if is_upwind_before and is_upwind_after:
            return 'tack'  # 風上→風上の転換はタック
        elif not is_upwind_before and not is_upwind_after:
            return 'jibe'  # 風下→風下の転換はジャイブ 
        elif is_upwind_before and not is_upwind_after:
            return 'bear_away'  # 風上→風下は落す操作
        else:
            return 'head_up'  # 風下→風上は上る操作

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
        
        # ハイブリッド風向推定を使用
        estimated_wind_direction, confidence_score = self.estimate_wind_direction_hybrid(df, boat_type)
        
        if estimated_wind_direction == 0.0 and confidence_score == 0.0:
            boat_id = df['boat_id'].iloc[0] if 'boat_id' in df.columns else 'Unknown'
            warnings.warn(f"Boat {boat_id}: 風向の推定に失敗しました。")
            return None
        
        # タック/ジャイブの検出と分類
        maneuvers = self.detect_maneuvers(df, estimated_wind_direction, min_tack_angle)
        
        # === 以下の風速推定部分はほぼ元のコードを維持 ===
        
        # 使用する係数の決定
        use_boat_type = boat_type.lower() if boat_type and boat_type.lower() in self.boat_coefficients else 'default'
        upwind_ratio = self.boat_coefficients[use_boat_type]['upwind']
        downwind_ratio = self.boat_coefficients[use_boat_type]['downwind']
        
        # 風速推定のための航行データの分類
        upwind_speeds = []
        downwind_speeds = []
        
        # 各データポイントの風に対する相対位置を計算
        for idx, row in df.iterrows():
            if 'bearing' in row and not pd.isna(row['bearing']) and 'speed' in row and not pd.isna(row['speed']):
                # 風向に対する相対角度
                relative_angle = abs((row['bearing'] - estimated_wind_direction + 180) % 360 - 180)
                
                # 風上か風下かを判定
                if relative_angle < 60:  # 風上
                    upwind_speeds.append(row['speed'])
                elif relative_angle > 120:  # 風下
                    downwind_speeds.append(row['speed'])
        
        # 風速推定値の計算
        avg_upwind_speed = np.mean(upwind_speeds) if upwind_speeds else 0
        avg_downwind_speed = np.mean(downwind_speeds) if downwind_speeds else 0
        
        est_wind_speed_from_upwind = avg_upwind_speed * upwind_ratio if avg_upwind_speed > 0 else 0
        est_wind_speed_from_downwind = avg_downwind_speed * downwind_ratio if avg_downwind_speed > 0 else 0
        
        # 両方の推定値の重み付き平均を取る
        if est_wind_speed_from_upwind > 0 and est_wind_speed_from_downwind > 0:
            # 風上からの推定値の方が信頼性が高いと仮定
            estimated_wind_speed = (est_wind_speed_from_upwind * 0.7 + est_wind_speed_from_downwind * 0.3)
        elif est_wind_speed_from_upwind > 0:
            estimated_wind_speed = est_wind_speed_from_upwind
        elif est_wind_speed_from_downwind > 0:
            estimated_wind_speed = est_wind_speed_from_downwind
        else:
            estimated_wind_speed = 0
        
        # ノットに変換（1 m/s ≈ 1.94384 ノット）
        estimated_wind_speed_knots = estimated_wind_speed * 1.94384
        
        # ウィンドウ分析で時間による変化を推定
        window_size = max(len(df) // 10, 20)  # データの約10%、最低20ポイント
        
        wind_estimates = []
        for i in range(0, len(df), window_size//2):  # 50%オーバーラップのウィンドウ
            end_idx = min(i + window_size, len(df))
            if end_idx - i < window_size // 2:  # 小さすぎるウィンドウはスキップ
                continue
                
            window_data = df.iloc[i:end_idx]
            
            center_time = window_data['timestamp'].iloc[len(window_data)//2] if 'timestamp' in window_data.columns else None
            center_lat = window_data['latitude'].mean() if 'latitude' in window_data.columns else None
            center_lon = window_data['longitude'].mean() if 'longitude' in window_data.columns else None
            
            # 風向の時間変化をモデル化
            if 'timestamp' in df.columns:
                time_factor = (window_data['timestamp'].iloc[0] - df['timestamp'].iloc[0]).total_seconds() / \
                            max(1, (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds())
            else:
                time_factor = i / max(1, len(df) - 1)
            
            # 風向の時間変化を考慮
            wind_direction_variation = (np.sin(time_factor * np.pi) * 5)  # ±5度程度の変動
            windowed_direction = (estimated_wind_direction + wind_direction_variation) % 360
            
            # 風速の時間変化も考慮
            wind_speed_variation = (np.cos(time_factor * np.pi * 2) * 0.5)  # ±0.5ノット程度の変動
            windowed_speed = max(0, estimated_wind_speed_knots + wind_speed_variation)
            
            # 風向風速推定の精度が時間とともに減少
            time_confidence = max(0.3, 1.0 - abs(0.5 - time_factor) * 0.3)
            window_confidence = confidence_score * time_confidence
            
            # ベイズ推定を使用する場合
            if use_bayesian and len(wind_estimates) > 0:
                # 前のウィンドウからの推定がある場合
                prior_direction = wind_estimates[-1]['wind_direction']
                prior_speed = wind_estimates[-1]['wind_speed_knots']
                prior_confidence = wind_estimates[-1]['confidence']
                
                # 前の推定と現在の測定の重み付き平均
                alpha = 0.3  # 現在の測定の重み
                updated_direction = self._weighted_angle_consensus(
                    {'prior': (prior_direction, prior_confidence), 
                     'current': (windowed_direction, window_confidence * alpha)}
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

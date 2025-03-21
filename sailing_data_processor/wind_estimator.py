import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.cluster import KMeans
import warnings
from datetime import datetime, timedelta
import math

class WindEstimator:
    """
    風向風速推定クラス - GPSデータから風向風速を推定する機能を提供
    
    このクラスは、セーリング艇のGPSデータ（位置、速度、方位など）から
    風向風速を推定するための様々なアルゴリズムを実装しています。
    
    重要な概念：
    
    - 風向（Wind Direction）：風が吹いてくる方向を示す角度（0-360度）。
      例えば、北風（0度）は北から南へ吹く風、東風（90度）は東から西へ吹く風。
      
    - 風上（Upwind）：風が吹いてくる方向に向かうこと。艇の進行方向と風向の
      相対角度が±90度以内の場合。
      
    - 風下（Downwind）：風に背を向けて進むこと。艇の進行方向と風向の
      相対角度が±90度を超える場合。
      
    - タック（Tack）：風上走行中に風向をまたいで方向転換する操作。
    
    - ジャイブ（Jibe/Gybe）：風下走行中に風向をまたいで方向転換する操作。
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
            # 修正: 風下走行時も風向は風が吹いてくる方向（艇の進行方向と逆）
            # 例：艇が180度を向いて風下走行している場合、風向は0度
            return (boat_bearing + 180) % 360
        else:  # reaching
            # リーチング時：艇の進行方向と風向の角度差は、90度が基本
            # 例：艇が90度を向いてリーチングしている場合、風向は約180度
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
        
        GPSデータから複数の手法（速度パターン、ポーラーデータ、最適VMG角度）を
        組み合わせて風向を推定します。各手法の結果を信頼度に基づいて統合し、
        より正確で堅牢な風向推定を実現します。
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ（少なくとも'bearing'と'speed'列が必要）
        boat_type : str
            艇種識別子
                
        Returns:
        --------
        Tuple[float, float]
            (推定風向（度、0-360の範囲、風が吹いてくる方向）, 信頼度スコア(0-1の範囲))
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
        データの品質を評価（ノイズレベル、変動性、バランス）
        
        Parameters:
        -----------
        df : pd.DataFrame
            評価対象のGPSデータ
            
        Returns:
        --------
        float
            品質スコア（0-1）、高いほど良質
        """
        try:
            # データのサイズ
            data_size = len(df)
            if data_size < 10:
                return 0.3  # 少なすぎるデータは低品質
                
            # 欠損値の割合 - 問題のある行を修正
            missing_data = 0.0
            if 'bearing' in df.columns:
                missing_bearing = df['bearing'].isna().mean()
                missing_data += missing_bearing * 0.5
            
            if 'speed' in df.columns:
                missing_speed = df['speed'].isna().mean()
                missing_data += missing_speed * 0.5
            
            if 'bearing' not in df.columns or 'speed' not in df.columns:
                missing_data = 0.5  # 必要な列がない場合は中程度のペナルティ
            
            # 速度の変動係数（標準偏差/平均）- 安定性の指標
            speed_cv = 1.0  # デフォルト値
            if 'speed' in df.columns and df['speed'].mean() > 0:
                speed_cv = df['speed'].std() / df['speed'].mean()
                
            # 方位角の標準偏差（安定したパターンでは低い）
            bearing_stddev = 1.0  # デフォルト値
            if 'bearing' in df.columns:
                # 循環データ（角度）なので特別な処理が必要
                sin_bearing = np.sin(np.radians(df['bearing']))
                cos_bearing = np.cos(np.radians(df['bearing']))
                
                bearing_stddev = np.sqrt(sin_bearing.var() + cos_bearing.var())
                
            # 品質スコアの計算（各要素に重み付け）
            quality_score = (
                (1.0 - missing_data) * 0.4 +          # 欠損値の少なさ
                (1.0 - min(1.0, speed_cv)) * 0.3 +    # 速度の安定性
                (1.0 - min(1.0, bearing_stddev)) * 0.3  # 方位の安定性
            )
            
            # 制限（0-1の範囲）
            return max(0.1, min(1.0, quality_score))
            
        except Exception as e:
            warnings.warn(f"データ品質評価エラー: {e}")
            return 0.5  # エラー時は中間値を返す

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

    def _estimate_from_tack_patterns(self, df):
        """
        タックパターン分析による風向推定
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ
            
        Returns:
        --------
        Tuple[float, float]
            (推定風向（風が吹いてくる方向）, 信頼度)
        """
        if df is None or len(df) < 10:
            return 0.0, 0.1  # データが少ない場合、低信頼度の風向を返す
        
        # 方位変化を計算し、タックを検出
        df_with_change = self._calculate_bearing_change(df)
        min_tack_angle = 45.0  # 最小タック角度
        tack_points = self._detect_tacks_improved(df_with_change, min_tack_angle)
        
        if tack_points is None or len(tack_points) < 2:
            # タックが十分に検出されない場合
            return 0.0, 0.2
        
        # タック前後の方位を収集
        tack_pairs = []  # タック前後のペアを保存
        
        for i, tack in tack_points.iterrows():
            tack_idx = tack.name if hasattr(tack, 'name') else i
            
            # タックの前後のインデックスを計算
            before_idx = max(0, tack_idx - 3)
            after_idx = min(len(df) - 1, tack_idx + 3)
            
            # タック前後の方位を取得
            if before_idx != tack_idx and after_idx != tack_idx:
                bearing_before = df.iloc[before_idx]['bearing']
                bearing_after = df.iloc[after_idx]['bearing']
                
                # 方位差を計算して実際にタックかどうか検証
                bearing_diff = abs(((bearing_after - bearing_before + 180) % 360) - 180)
                
                # タックは通常80-100度程度の方位変化を伴う
                if 70 <= bearing_diff <= 110:
                    tack_pairs.append((bearing_before, bearing_after))
        
        # 有効なタックペアから風向を推定
        wind_directions = []
        
        for bearing_before, bearing_after in tack_pairs:
            # タック前後の方位から風向を推定
            # 両方位の二等分線を計算（風向または風向+180度）
            bisector = self._calculate_bisector(bearing_before, bearing_after)
            
            # タックは風上での操作なので、二等分線は風向軸を示す
            # 実際の風向かその逆かを決定する必要がある
            
            # 最も可能性が高いのは、二等分線そのものが風向
            # （クローズホールドで風に向かって進むため）
            wind_directions.append(bisector)
        
        if wind_directions:
            # 複数の推定値の平均
            wind_direction = self._calculate_mean_angle(wind_directions)
            
            # 信頼度は検出タック数に依存
            confidence = min(0.8, 0.4 + len(wind_directions) * 0.1)
            
            return wind_direction, confidence
        
        # 有効なタックが検出されない場合
        return 0.0, 0.2
    
    def _calculate_bisector(self, angle1: float, angle2: float) -> float:
        """
        2つの角度の二等分線を計算する（角度の循環性を考慮）
        
        Parameters:
        -----------
        angle1 : float
            1つ目の角度（度、0-360）
        angle2 : float
            2つ目の角度（度、0-360）
            
        Returns:
        --------
        float
            二等分線の角度（度、0-360）
        """
        # 正弦・余弦成分の平均を使用して二等分線を計算
        sin_avg = (np.sin(np.radians(angle1)) + np.sin(np.radians(angle2))) / 2
        cos_avg = (np.cos(np.radians(angle1)) + np.cos(np.radians(angle2))) / 2
        
        # 平均角度を計算
        bisector = np.degrees(np.arctan2(sin_avg, cos_avg)) % 360
        
        # 角度差が180度を超える場合（0度線を跨ぐ場合）の補正
        if abs(((angle1 - angle2) + 180) % 360 - 180) > 90:
            bisector = (bisector + 180) % 360
        
        return bisector

    def _estimate_from_speed_patterns(self, df):
        """
        速度パターン分析による風向推定
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ
            
        Returns:
        --------
        Tuple[float, float]
            (推定風向（風が吹いてくる方向）, 信頼度)
        """
        if df is None or len(df) < 10:
            return 0.0, 0.1  # データが少ない場合、低信頼度の風向を返す
        
        # 速度変化に基づく風向推定
        df_speed = df.copy()
        
        # 移動平均で速度をスムーズ化
        if len(df_speed) > 5:
            df_speed['smooth_speed'] = df_speed['speed'].rolling(window=5, center=True).mean()
            df_speed['smooth_speed'].fillna(df_speed['speed'], inplace=True)
        else:
            df_speed['smooth_speed'] = df_speed['speed']
        
        # 速度の変化に基づいて有効なインデックスを選択
        speed_threshold = df_speed['smooth_speed'].max() * 0.7
        valid_mask = df_speed['smooth_speed'] > speed_threshold
        
        # 修正: valid_indicesが空でないことを確認
        valid_indices = df_speed.index[valid_mask].tolist()
        
        if not valid_indices:  # リストが空の場合のチェックを追加
            return 0.0, 0.1  # データが不十分な場合のデフォルト値
        
        # インデックスが整数値だけを含むことを確保
        valid_indices = [int(i) for i in valid_indices if pd.notna(i)]
        
        # 再度空のリストチェック
        if not valid_indices:
            return 0.0, 0.1
    
        # 有効なインデックスがあるデータのみを使用
        valid_df = df.iloc[valid_indices].copy()
        
        # 方位角の分析
        bearings = valid_df['bearing'].tolist()
        
        # 風向グループを特定
        if len(bearings) >= 10:
            # 方位角を円周上に変換
            X = np.column_stack([
                np.cos(np.radians(bearings)),
                np.sin(np.radians(bearings))
            ])
            
            # K-meansクラスタリングで主要な方位グループを分類
            n_clusters = min(3, len(X) // 3)  # データ量に応じてクラスタ数を調整
            if n_clusters < 1:
                n_clusters = 1
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X)
            clusters = kmeans.labels_
            
            # クラスタごとの統計
            cluster_stats = []
            for i in range(n_clusters):
                cluster_mask = clusters == i
                if np.any(cluster_mask):
                    cluster_bearings = np.array(bearings)[cluster_mask]
                    cluster_speeds = valid_df['speed'].iloc[np.where(cluster_mask)[0]].values
                    
                    # クラスタの統計情報
                    cluster_stats.append({
                        'mean_bearing': self._calculate_mean_angle(cluster_bearings),
                        'std_bearing': np.std(cluster_bearings),
                        'mean_speed': np.mean(cluster_speeds),
                        'count': np.sum(cluster_mask),
                        'bearings': cluster_bearings
                    })
            
            # クラスタ統計があれば処理
            if cluster_stats:
                # 速度の最大・最小クラスタを特定
                max_speed_cluster = max(cluster_stats, key=lambda x: x['mean_speed'])
                min_speed_cluster = min(cluster_stats, key=lambda x: x['mean_speed'])
                
                # 風上風下の推定
                if max_speed_cluster['mean_speed'] > min_speed_cluster['mean_speed'] * 1.2:
                    # 十分な速度差がある場合
                    # 速い方が風下、遅い方が風上と推定
                    downwind_bearing = max_speed_cluster['mean_bearing']
                    upwind_bearing = min_speed_cluster['mean_bearing']
                    
                    # 修正: 風向 = 風上の方位（風が吹いてくる方向）
                    wind_direction = upwind_bearing
                    
                    # 風上風下が互いに反対方向に近いほど信頼度が高い
                    angle_diff = abs(((downwind_bearing - upwind_bearing + 180) % 360) - 180)
                    confidence = max(0.3, min(0.9, angle_diff / 180.0))
                    
                    return wind_direction, confidence
                
            # 単一クラスタまたは速度差が小さい場合    
            # 速度に基づく簡易推定
            fastest_indices = valid_df['speed'].nlargest(min(5, len(valid_df))).index
            fastest_bearings = valid_df.loc[fastest_indices, 'bearing'].values
            
            # 修正: 最速点の方位から風向推定（風が吹いてくる方向=艇の進行方向+180度）
            fastest_mean_bearing = self._calculate_mean_angle(fastest_bearings)
            wind_direction = (fastest_mean_bearing + 180) % 360  # 船の進行方向の反対が風向
            
            # 低い信頼度
            return wind_direction, 0.4
        
    else:
        # データが少ない場合、単純に平均風向と低信頼度を返す
        mean_bearing = self._calculate_mean_angle(bearings)
        wind_direction = (mean_bearing + 180) % 360  # 船の進行方向の反対が風向
        return wind_direction, 0.3
            
        else:
            # データが少ない場合、単純に平均風向と低信頼度を返す
            mean_bearing = self._calculate_mean_angle(bearings)
            wind_direction = (mean_bearing + 180) % 360  # 修正: 船の進行方向の反対が風向
            return wind_direction, 0.3
    
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
        ポーラー性能カーブとの比較に基づいて風向を推定
        
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
        # ポーラーデータの取得
        coefficients = self.boat_coefficients.get(boat_type.lower(), self.boat_coefficients['default'])
        upwind_vmg_angle = self._calculate_optimal_vmg_angle(boat_type, True)
        downwind_vmg_angle = self._calculate_optimal_vmg_angle(boat_type, False)
        
        # データを有効なエントリだけに絞り込み
        # dropna() を使う代わりに安全なマスキングを使用
        if 'bearing' not in df.columns or 'speed' not in df.columns:
            return 0.0, 0.3  # 有効なデータがない場合
            
        valid_mask = df['bearing'].notna() & df['speed'].notna()
        if not valid_mask.any():
            return 0.0, 0.3  # 有効なデータがない場合
        
        valid_indices = np.where(valid_mask)[0]
        valid_data = df.iloc[valid_indices].copy()
        
        # 十分なデータポイントがない場合
        if len(valid_data) < 10:
            return 0.0, 0.3
        
        # 速度の標準化（最大値で割る）
        max_speed = valid_data['speed'].max()
        if max_speed <= 0:
            return 0.0, 0.3
            
        valid_data['norm_speed'] = valid_data['speed'] / max_speed
        
        # 方位角をグループ化（10度間隔）
        bin_size = 10
        bins = np.arange(0, 361, bin_size)
        
        # 各方位ビンの標準化速度を計算
        bin_speeds = []
        bin_centers = []
        bin_counts = []
        
        for i in range(len(bins)-1):
            # ビンの下限と上限
            low = bins[i]
            high = bins[i+1]
            
            # このビンに含まれる方位のインデックスを安全に取得
            if i == len(bins)-2:  # 最後のビン（360度含む）
                bin_mask = (valid_data['bearing'] >= low) & (valid_data['bearing'] <= high)
            else:
                bin_mask = (valid_data['bearing'] >= low) & (valid_data['bearing'] < high)
            
            bin_indices = np.where(bin_mask)[0]
            if len(bin_indices) > 0:
                bin_data = valid_data.iloc[bin_indices]
                bin_speeds.append(bin_data['norm_speed'].mean())
                bin_centers.append((low + high) / 2)
                bin_counts.append(len(bin_data))
        
        if not bin_speeds:
            return 0.0, 0.3
        
        # 2つの主要方向クラスターを見つける（K-means）
        from sklearn.cluster import KMeans
        
        # 角度データを単位円上の点に変換（周期性を考慮）
        X = np.column_stack([
            np.cos(np.radians(bin_centers)) * np.array(bin_speeds),
            np.sin(np.radians(bin_centers)) * np.array(bin_speeds)
        ])
        
        # クラスタリング（K-means）
        n_clusters = min(2, len(X))
        if n_clusters < 2:
            return 0.0, 0.3  # クラスタリングに十分なデータがない
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        clusters = kmeans.labels_
        
        # クラスタごとの平均速度と方位
        cluster_data = []
        for i in range(n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            if len(cluster_indices) > 0:
                # このクラスタの方位と速度を取得
                cluster_bearings = [bin_centers[j] for j in cluster_indices]
                cluster_speeds = [bin_speeds[j] for j in cluster_indices]
                
                # 平均方位の計算（角度なので特別な処理）
                sin_sum = sum(np.sin(np.radians(b)) for b in cluster_bearings)
                cos_sum = sum(np.cos(np.radians(b)) for b in cluster_bearings)
                avg_bearing = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
                
                # 平均速度
                avg_speed = sum(cluster_speeds) / len(cluster_speeds)
                
                cluster_data.append({
                    'bearing': avg_bearing,
                    'speed': avg_speed,
                    'size': sum(bin_counts[j] for j in cluster_indices)
                })
        
        if len(cluster_data) < 2:
            return 0.0, 0.3  # 少なくとも2つのクラスターが必要
        
        # 速度に基づいてソート（遅い順）
        cluster_data.sort(key=lambda x: x['speed'])
        
        # 最も遅いクラスタは風上、2番目に遅いクラスタは風下と仮定
        upwind_cluster = cluster_data[0]
        downwind_cluster = cluster_data[-1]
        
        # 風上クラスタの方位から風向を推定
        upwind_bearing = upwind_cluster['bearing']
        
        # 風向の計算（風上の方位から）
        wind_direction = (upwind_bearing + 180) % 360
        
        # 風下方向と理論上の風下方向（風上の反対）の差
        expected_downwind = (upwind_bearing + 180) % 360
        actual_downwind = downwind_cluster['bearing']
        downwind_diff = abs(((actual_downwind - expected_downwind) + 180) % 360 - 180)
        
        # 信頼度の計算
        speed_diff = downwind_cluster['speed'] - upwind_cluster['speed']
        speed_clarity = min(1.0, speed_diff / 0.3)  # 速度差の明確さ
        direction_agreement = max(0, 1.0 - downwind_diff / 90.0)  # 方向の一致度
        
        confidence = 0.5 + (speed_clarity * 0.25) + (direction_agreement * 0.25)
        
        # ポーラー性能と一致するかチェック（オプション）
        # 実測速度とポーラー曲線の比較
        
        # 風上・風下VMG角度を用いた風向修正
        upwind_direction = (wind_direction + upwind_vmg_angle) % 360
        downwind_direction = (wind_direction + 180 - downwind_vmg_angle) % 360
        
        # 風上・風下のクラスタに近いかをチェック
        upwind_match = min(
            abs(((upwind_cluster['bearing'] - upwind_direction) + 180) % 360 - 180),
            abs(((upwind_cluster['bearing'] - (upwind_direction + 60)) + 180) % 360 - 180),
            abs(((upwind_cluster['bearing'] - (upwind_direction - 60)) + 180) % 360 - 180)
        )
        
        downwind_match = min(
            abs(((downwind_cluster['bearing'] - downwind_direction) + 180) % 360 - 180),
            abs(((downwind_cluster['bearing'] - (downwind_direction + 30)) + 180) % 360 - 180),
            abs(((downwind_cluster['bearing'] - (downwind_direction - 30)) + 180) % 360 - 180)
        )
        
        # マッチの良さに基づいて信頼度を調整
        polar_match = 1.0 - min(1.0, (upwind_match + downwind_match) / 90.0)
        final_confidence = 0.7 * confidence + 0.3 * polar_match
        
        return wind_direction, final_confidence

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
        
        艇の航行方位パターンからタック角度を分析し、風上方向と最適VMG角度を
        考慮して風向を推定します。タック時の方位変化は通常、最適風上角度の
        2倍程度になるため、これを利用して風向を逆算します。
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ
        boat_type : str
            艇種識別子
            
        Returns:
        --------
        Tuple[float, float]
            (推定風向（度、0-360、風が吹いてくる方向）, 信頼度)
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
                    # 二等分線を計算（風上方向）- ヘルパーメソッドを使用
                    bisector = self._calculate_bisector(top_cluster_angles[0], top_cluster_angles[1])
                    
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
        
        各推定手法からの風向（風が吹いてくる方向）と信頼度の情報を元に、
        重み付き平均による合意形成を行い、最終的な風向を決定します。
        
        Parameters:
        -----------
        results_dict : Dict[str, Tuple[float, float]]
            推定手法: (風向, 信頼度) の辞書
            風向は「風が吹いてくる方向」の角度（度、0-360）
            
        Returns:
        --------
        float
            合意形成された風向（度、0-360、風が吹いてくる方向）
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
        
        # 正弦・余弦成分の重み付き平均（角度の循環性を考慮）
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
        
        艇のGPSデータから方向転換（マニューバー）を検出し、風向を基に
        タック、ジャイブ、ベアアウェイ、ヘッドアップに分類します。
        タックは風上走行中に風向をまたいで方向転換する操作、ジャイブは
        風下走行中に風向をまたいで方向転換する操作です。
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPSデータ
        wind_direction : float
            推定風向（度、0-360、風が吹いてくる方向）
        min_angle_change : float
            方向転換と判定する最小角度変化
        window_size : int
            移動ウィンドウサイズ
                
        Returns:
        --------
        pd.DataFrame
            検出された方向転換とその分類（タック/ジャイブ/ベアアウェイ/ヘッドアップ）
        """
        if df is None or len(df) < window_size * 2:
            return pd.DataFrame()  # 十分なデータがない場合は空のデータフレームを返す
        
        # 風向のバリデーション
        if wind_direction is None or not (0 <= wind_direction < 360):
            wind_direction = 0.0  # デフォルト値を設定
        
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
        
        セーリングにおけるマニューバー（方向転換）の種類は、艇が風向に対して
        どのように位置を変えたかによって決まります。このメソッドは、転換前後の
        艇の方位と風向から、タック、ジャイブ、ベアアウェイ、ヘッドアップの
        いずれかに分類します。
        
        Parameters:
        -----------
        before_bearing : float
            転換前の艇の方位（度、0-360）
        after_bearing : float
            転換後の艇の方位（度、0-360）
        wind_direction : float
            風向（度、0-360、風が吹いてくる方向）
            
        Returns:
        --------
        str
            'tack': 風上走行中に風向をまたいで方向転換する
            'jibe': 風下走行中に風向をまたいで方向転換する
            'bear_away': 風上から風下へと艇を落とす
            'head_up': 風下から風上へと艇を上げる
        """
        # 風向に対する相対角度を計算
        rel_before = (before_bearing - wind_direction + 360) % 360
        rel_after = (after_bearing - wind_direction + 360) % 360
        
        # 風上側か風下側か判定
        # 相対角度が0-90度または270-360度の範囲にある場合、風上側と判断
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
        # 1. データの検証
        if gps_data is None or len(gps_data) < self.min_valid_points:
            warnings.warn(f"有効なデータが不足しています")
            return None
        
        # 2. 必要列の存在チェック
        required_cols = ['bearing', 'speed']
        if not all(col in gps_data.columns for col in required_cols):
            warnings.warn(f"必要な列がありません")
            return None
        
        # 3. タイムスタンプ処理 - 安全に期間を計算
        duration = 0
        if 'timestamp' in gps_data.columns:
            try:
                # タイムスタンプ列のデータ型を確認
                if pd.api.types.is_datetime64_any_dtype(gps_data['timestamp']):
                    # すでにdatetime型の場合
                    min_time = gps_data['timestamp'].min()
                    max_time = gps_data['timestamp'].max()
                    duration = (max_time - min_time).total_seconds()
                else:
                    # datetimeに変換
                    timestamps = pd.to_datetime(gps_data['timestamp'], errors='coerce')
                    if timestamps.notna().any():
                        min_time = timestamps.min()
                        max_time = timestamps.max()
                        duration = (max_time - min_time).total_seconds()
                    else:
                        # 変換失敗時はポイント数からの推定
                        duration = len(gps_data) * 5  # 5秒間隔を仮定
            except Exception as e:
                # エラー時はデータポイント数からの推定
                warnings.warn(f"タイムスタンプ処理エラー: {e}")
                duration = len(gps_data) * 5  # 5秒間隔を仮定
        else:
            # タイムスタンプ列がない場合
            duration = len(gps_data) * 5  # 仮定の間隔
        
        # 残りのメソッドは既存のコードと同様...
        
        # 有効データ期間のチェック
        if duration < self.min_valid_duration:
            warnings.warn(f"データ期間が短すぎます: {duration}秒 < {self.min_valid_duration}秒")
            if len(gps_data) > self.min_valid_points * 2:  # ポイント数が十分であれば継続
                warnings.warn("ポイント数は十分なので処理を継続します")
            else:
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

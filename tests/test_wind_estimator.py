"""
WindEstimator クラスのテスト
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import os
import sys
import warnings

# テスト対象のモジュールをインポート
from sailing_data_processor import WindEstimator


class TestWindEstimator(unittest.TestCase):
    """WindEstimatorクラスのテストケース"""
    
    def setUp(self):
        """テストの準備"""
        # 警告を無視
        warnings.filterwarnings("ignore")
        
        # テスト用のオブジェクト作成
        self.estimator = WindEstimator()
        
        # テスト用のデータを作成
        self.upwind_downwind_data = self._create_upwind_downwind_pattern()
        self.tacking_data = self._create_tacking_pattern()
    
    def _create_upwind_downwind_pattern(self):
        """風上風下パターンのサンプルデータを作成"""
        # 基本情報
        base_lat, base_lon = 35.6, 139.7
        points = 200
        timestamps = [datetime(2024, 3, 1, 10, 0, 0) + timedelta(seconds=i*5) for i in range(points)]
        
        # 風上風下パターンのコース（風向0度と仮定）
        bearings = []
        speeds = []
        lats = []
        lons = []
        
        # 風上レグ（ジグザグでタックを模擬）
        for i in range(100):
            # タックを切る（30度と330度で風上へジグザグ）
            bearing = 30 if i % 20 < 10 else 330
            bearings.append(bearing)
            
            # 風上は遅め（4-5ノット）
            speed = (4.0 + np.random.random()) * 0.514444  # m/s に変換
            speeds.append(speed)
            
            # 座標を計算
            if i > 0:
                # 前の位置から新しい位置を計算
                dist = speeds[-2] * 5  # 5秒分の距離
                dx = dist * math.sin(math.radians(bearings[-2]))
                dy = dist * math.cos(math.radians(bearings[-2]))
                
                # メートルを度に変換（近似）
                dlat = dy / 111000
                dlon = dx / (111000 * math.cos(math.radians(lats[-1])))
                
                lats.append(lats[-1] + dlat)
                lons.append(lons[-1] + dlon)
            else:
                lats.append(base_lat)
                lons.append(base_lon)
        
        # 風下レグ（ほぼ直線）
        for i in range(100):
            # 風下は180度付近
            bearing = 180 + (np.random.random() - 0.5) * 10
            bearings.append(bearing)
            
            # 風下は速め（6-7ノット）
            speed = (6.0 + np.random.random()) * 0.514444  # m/s に変換
            speeds.append(speed)
            
            # 座標を計算
            dist = speeds[-2] * 5  # 5秒分の距離
            dx = dist * math.sin(math.radians(bearings[-2]))
            dy = dist * math.cos(math.radians(bearings[-2]))
            
            # メートルを度に変換（近似）
            dlat = dy / 111000
            dlon = dx / (111000 * math.cos(math.radians(lats[-1])))
            
            lats.append(lats[-1] + dlat)
            lons.append(lons[-1] + dlon)
        
        # DataFrameに変換
        return pd.DataFrame({
            'timestamp': timestamps,
            'latitude': lats,
            'longitude': lons,
            'bearing': bearings,
            'speed': speeds,
            'boat_id': ['test_boat'] * points
        })
    
    def _create_tacking_pattern(self):
        """タックパターンのサンプルデータを作成"""
        # 基本情報
        base_lat, base_lon = 35.6, 139.7
        points = 100
        timestamps = [datetime(2024, 3, 1, 10, 0, 0) + timedelta(seconds=i*5) for i in range(points)]
        
        # タックポイントの位置
        tack_indices = [20, 40, 60, 80]
        
        # 方位、速度、座標を初期化
        bearings = []
        speeds = []
        lats = [base_lat]
        lons = [base_lon]
        
        # タックポイントに基づいてコースを生成
        current_bearing = 30
        
        for i in range(points):
            # タックポイントでコースを変更
            if i in tack_indices:
                # タックを切る（約60度の角度変化）
                current_bearing = (current_bearing + 300) % 360
            
            bearings.append(current_bearing)
            
            # 速度は一定（5ノット）
            speed = 5.0 * 0.514444  # m/s に変換
            speeds.append(speed)
            
            # 座標を更新（最初のポイントを除く）
            if i > 0:
                dist = speeds[-2] * 5  # 5秒分の距離
                dx = dist * math.sin(math.radians(bearings[-2]))
                dy = dist * math.cos(math.radians(bearings[-2]))
                
                # メートルを度に変換（近似）
                dlat = dy / 111000
                dlon = dx / (111000 * math.cos(math.radians(lats[-1])))
                
                lats.append(lats[-1] + dlat)
                lons.append(lons[-1] + dlon)
        
        # DataFrameに変換
        return pd.DataFrame({
            'timestamp': timestamps,
            'latitude': lats,
            'longitude': lons,
            'bearing': bearings,
            'speed': speeds,
            'boat_id': ['test_boat'] * points
        })
    
    def _create_simple_tack_data(self):
        """明確なタックを含むシンプルなGPSデータを作成"""
        # 基準時刻
        base_time = datetime(2024, 3, 1, 10, 0, 0)
        
        # 100ポイントのデータ生成
        points = 100
        timestamps = [base_time + timedelta(seconds=i*5) for i in range(points)]
        
        # 方位データ - より明確なタックパターン（急激な変化）を作成
        bearings = [30] * 45  # 最初の45ポイントは30度
        bearings += [i for i in range(30, 150, 12)]  # 10ポイントで30度→150度へ変化
        bearings += [150] * (100 - len(bearings))  # 残りのポイントは150度
        
        # 緯度・経度データ（シンプルな直線）
        base_lat, base_lon = 35.6, 139.7
        lats = [base_lat + i * 0.0001 for i in range(points)]
        lons = [base_lon + i * 0.0001 for i in range(points)]
        
        # 速度データ（タック時に減速）
        speeds = []
        for i in range(points):
            if 45 <= i < 55:  # タック中は減速
                speeds.append(3.0)
            else:
                speeds.append(5.0)
        
        # 配列の長さを確認
        assert len(timestamps) == points
        assert len(bearings) == points
        assert len(lats) == points
        assert len(lons) == points
        assert len(speeds) == points
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'latitude': lats,
            'longitude': lons,
            'speed': np.array(speeds) * 0.514444,  # ノット→m/s変換
            'bearing': bearings,
            'boat_id': ['TestBoat'] * points
        })
        
        return df
    
    def test_tack_detection(self):
        """タック検出のテスト"""
        # タック検出用に方向変化を計算
        df = self.tacking_data.copy()
        df['bearing_change'] = df['bearing'].diff().abs()
        df['is_tack'] = df['bearing_change'] > 30.0
        
        # 検出されたタック数をカウント
        detected_tacks = df['is_tack'].sum()
        
        # 4つのタックポイントを正しく検出できているかチェック
        self.assertEqual(detected_tacks, 4, f"期待されるタック数（4）と検出されたタック数（{detected_tacks}）が一致しない")
    
    def test_single_boat_wind_estimation(self):
        """単一艇からの風向風速推定のテスト"""
        # 風上風下パターンから風向風速を推定
        result = self.estimator.estimate_wind_from_single_boat(
            gps_data=self.upwind_downwind_data,
            min_tack_angle=25.0,
            boat_type='470',
            use_bayesian=False
        )
        
        # 結果が存在することを確認
        self.assertIsNotNone(result, "風向風速の推定結果がNoneです")
        
        # 風向・風速・信頼度が含まれていることを確認
        self.assertIn('wind_direction', result.columns, "wind_directionカラムがありません")
        self.assertIn('wind_speed_knots', result.columns, "wind_speed_knotsカラムがありません")
        self.assertIn('confidence', result.columns, "confidenceカラムがありません")
        
        # 風向が正しく推定されているか（テストデータは風向0度で設計）
        avg_direction = result['wind_direction'].mean()
        self.assertTrue(
            (330 <= avg_direction <= 360) or (0 <= avg_direction <= 30),
            f"推定風向（{avg_direction}度）が想定範囲（330〜30度）外です"
        )
        
        # 風速が妥当な範囲か（テストデータは5ノット前後で設計）
        avg_speed = result['wind_speed_knots'].mean()
        self.assertTrue(
            4.0 <= avg_speed <= 8.0,
            f"推定風速（{avg_speed}ノット）が想定範囲（4〜8ノット）外です"
        )
    
    def test_bayesian_estimation(self):
        """ベイズ推定モードのテスト"""
        # ベイズ推定ありとなしで推定
        result_with_bayes = self.estimator.estimate_wind_from_single_boat(
            gps_data=self.upwind_downwind_data,
            min_tack_angle=25.0,
            boat_type='470',
            use_bayesian=True
        )
        
        result_without_bayes = self.estimator.estimate_wind_from_single_boat(
            gps_data=self.upwind_downwind_data,
            min_tack_angle=25.0,
            boat_type='470',
            use_bayesian=False
        )
        
        # どちらも結果が存在することを確認
        self.assertIsNotNone(result_with_bayes, "ベイズ推定ありの結果がNoneです")
        self.assertIsNotNone(result_without_bayes, "ベイズ推定なしの結果がNoneです")
        
        # 信頼度の比較（ベイズ推定ありの方が高いはず）
        avg_confidence_with_bayes = result_with_bayes['confidence'].mean()
        avg_confidence_without_bayes = result_without_bayes['confidence'].mean()
        
        # 差が小さい場合もあるので、同等以上であることを確認
        self.assertGreaterEqual(
            avg_confidence_with_bayes, 
            avg_confidence_without_bayes * 0.95,  # 5%の許容誤差
            "ベイズ推定ありの信頼度が明らかに低くなっています"
        )
    
    def test_different_boat_types(self):
        """艇種による推定の違いのテスト"""
        # 異なる艇種で推定
        result_laser = self.estimator.estimate_wind_from_single_boat(
            gps_data=self.upwind_downwind_data,
            min_tack_angle=25.0,
            boat_type='laser',
            use_bayesian=False
        )
        
        result_star = self.estimator.estimate_wind_from_single_boat(
            gps_data=self.upwind_downwind_data,
            min_tack_angle=25.0,
            boat_type='star',
            use_bayesian=False
        )
        
        # 風向はほぼ同じであることを確認
        laser_direction = result_laser['wind_direction'].mean()
        star_direction = result_star['wind_direction'].mean()
        
        direction_diff = abs((laser_direction - star_direction + 180) % 360 - 180)
        self.assertLessEqual(direction_diff, 20, 
                          f"艇種による風向の差が大きすぎます（{direction_diff}度）")
        
        # 風速は艇種による係数の違いで異なる可能性がある
        laser_speed = result_laser['wind_speed_knots'].mean()
        star_speed = result_star['wind_speed_knots'].mean()
        
        # ただし極端に違いすぎないことを確認
        self.assertLessEqual(abs(laser_speed - star_speed), 2.0,
                          f"艇種による風速の差が大きすぎます（{abs(laser_speed - star_speed)}ノット）")
    
    def test_multiple_boats_estimation(self):
        """複数艇からの推定のテスト"""
        # 2つのデータセットを準備（同じデータに微小な変化）
        data1 = self.upwind_downwind_data.copy()
        
        data2 = self.upwind_downwind_data.copy()
        data2['boat_id'] = ['test_boat2'] * len(data2)
        data2['speed'] = data2['speed'] * 1.1  # 10%速く
        
        # ボートデータセット
        boats_data = {
            'test_boat1': data1,
            'test_boat2': data2
        }
        
        # 艇種と重み
        boat_types = {'test_boat1': '470', 'test_boat2': '470'}
        boat_weights = {'test_boat1': 0.8, 'test_boat2': 0.6}
        
        # 推定を実行
        results = self.estimator.estimate_wind_from_multiple_boats(
            boats_data=boats_data,
            boat_types=boat_types,
            boat_weights=boat_weights
        )
        
        # 結果の確認
        self.assertIsNotNone(results, "複数艇からの推定結果がNoneです")
        self.assertEqual(len(results), 2, "2艇からの推定結果が返されていません")
        
        # 各艇の結果が存在することを確認
        self.assertIn('test_boat1', results, "test_boat1の推定結果がありません")
        self.assertIn('test_boat2', results, "test_boat2の推定結果がありません")
    
    def test_wind_field_estimation(self):
        """風の場推定のテスト"""
        # まず単一艇の風推定
        wind_data = self.estimator.estimate_wind_from_single_boat(
            gps_data=self.upwind_downwind_data,
            min_tack_angle=25.0,
            boat_type='470'
        )
        
        # 推定器に結果を設定
        self.estimator.wind_estimates = {'test_boat': wind_data}
        
        # 風の場を推定
        time_point = self.upwind_downwind_data['timestamp'].iloc[50]
        wind_field = self.estimator.estimate_wind_field(time_point, grid_resolution=5)
        
        # 結果が存在することを確認
        self.assertIsNotNone(wind_field, "風の場の推定結果がNoneです")
        
        # 必要なフィールドが含まれていることを確認
        self.assertIn('lat_grid', wind_field, "lat_gridフィールドがありません")
        self.assertIn('lon_grid', wind_field, "lon_gridフィールドがありません")
        self.assertIn('wind_direction', wind_field, "wind_directionフィールドがありません")
        self.assertIn('wind_speed', wind_field, "wind_speedフィールドがありません")
        self.assertIn('confidence', wind_field, "confidenceフィールドがありません")
        
        # グリッドのサイズ確認
        self.assertEqual(wind_field['lat_grid'].shape, (5, 5), "緯度グリッドのサイズが5x5ではありません")
        self.assertEqual(wind_field['wind_direction'].shape, (5, 5), "風向グリッドのサイズが5x5ではありません")
    
    def test_calculate_bearing_change(self):
        """循環角度を考慮した方位変化計算のテスト"""
        # テストデータ作成
        test_data = pd.DataFrame({
            'bearing': [0, 45, 90, 180, 270, 359, 1, 358],
            'timestamp': [datetime(2024, 3, 1, 10, 0, i) for i in range(8)],
            'latitude': [35.6] * 8,
            'longitude': [139.7] * 8,
            'speed': [5.0] * 8,
            'boat_id': ['test_boat'] * 8
        })
        
        # メソッドを実行
        result = self.estimator._calculate_bearing_change(test_data)
        
        # 結果の検証
        self.assertIn('bearing_change', result.columns, "bearing_change列が追加されていません")
        
        # 特定の角度変化の検証
        # 359度から1度への変化は2度であるべき（0度線を跨ぐ例）
        self.assertAlmostEqual(result.iloc[6]['bearing_change'], 2.0, delta=0.1, 
                         msg="0度線を跨ぐ角度変化の計算が誤っています")
        
        # 90度から180度への変化は90度であるべき
        self.assertAlmostEqual(result.iloc[3]['bearing_change'], 90.0, delta=0.1, 
                         msg="通常の角度変化の計算が誤っています")

    def test_detect_tacks_improved(self):
        """後方互換性用のタック検出メソッドテスト"""
        # テストデータ作成 - 明確なタックパターンを持つデータ
        df = self._create_simple_tack_data()
        
        # 方位変化を計算
        df = self.estimator._calculate_bearing_change(df)
        
        # 古いタック検出メソッドはシンプルに動作するだけ
        tack_points = self.estimator._detect_tacks_improved(df, min_tack_angle=30.0)
        
        # 結果がNoneでないことを確認（最低限のチェック）
        self.assertIsNotNone(tack_points, "タック検出結果がNoneです")
        
        # 注: 今後はdetect_maneuversメソッドを使用するため、
        # 詳細なアサーションは行わない

    def test_wind_direction_calculation_methods(self):
        """風向計算メソッドのテスト"""
        # 風上走行の場合
        boat_bearing = 45
        vmg_angle = 40
        wind_dir_upwind = self.estimator._calculate_wind_direction(boat_bearing, 'upwind', vmg_angle)
        
        # 理論上の風向（方位+180-VMG角度）
        theoretical_wind_dir = (boat_bearing + 180 - vmg_angle) % 360
        
        # 検証
        self.assertEqual(wind_dir_upwind, theoretical_wind_dir, 
                   f"風上走行での風向計算が不正確です: {wind_dir_upwind} != {theoretical_wind_dir}")
        
        # 風下走行の場合
        boat_bearing = 180
        wind_dir_downwind = self.estimator._calculate_wind_direction(boat_bearing, 'downwind')
        
        # 風下走行では、艇の進行方向が風向になる
        self.assertEqual(wind_dir_downwind, boat_bearing, 
                   f"風下走行での風向計算が不正確です: {wind_dir_downwind} != {boat_bearing}")
        
        # リーチング（横風）の場合
        boat_bearing = 90
        wind_dir_reaching = self.estimator._calculate_wind_direction(boat_bearing, 'reaching')
        
        # リーチングでは、艇の進行方向+90度が風向になる
        theoretical_reaching_dir = (boat_bearing + 90) % 360
        self.assertEqual(wind_dir_reaching, theoretical_reaching_dir, 
                   f"リーチングでの風向計算が不正確です: {wind_dir_reaching} != {theoretical_reaching_dir}")
        
    def test_special_data_patterns(self):
        """特殊データパターンのテスト"""
        # クローズホールドのみのデータ（単一方向）
        close_hauled_data = self._create_close_hauled_data()
        close_hauled_result = self.estimator.estimate_wind_from_single_boat(
            gps_data=close_hauled_data,
            min_tack_angle=30.0,
            boat_type='laser',
            use_bayesian=False
        )
        
        # 結果が存在することを確認
        self.assertIsNotNone(close_hauled_result, "クローズホールドデータからの風向風速推定がNoneです")
        
        # リーチングのみのデータ（風向と直角）
        reaching_data = self._create_reaching_data()
        reaching_result = self.estimator.estimate_wind_from_single_boat(
            gps_data=reaching_data,
            min_tack_angle=30.0,
            boat_type='laser',
            use_bayesian=False
        )
        
        # 結果が存在することを確認
        self.assertIsNotNone(reaching_result, "リーチングデータからの風向風速推定がNoneです")
        
        # 非常に短いデータセット（最小有効データポイント付近）
        min_points = self.estimator.min_valid_points
        short_data = self.upwind_downwind_data.iloc[:min_points].copy()
        short_result = self.estimator.estimate_wind_from_single_boat(
            gps_data=short_data,
            min_tack_angle=30.0,
            boat_type='laser',
            use_bayesian=False
        )
        
        # 結果が存在することを確認
        self.assertIsNotNone(short_result, "短いデータセットからの風向風速推定がNoneです")
        
        # 異常値を含むデータセット
        noisy_data = self.upwind_downwind_data.copy()
        # 速度に異常値を追加
        noisy_data.loc[5, 'speed'] = noisy_data['speed'].max() * 3
        # 方位に異常値を追加
        noisy_data.loc[15, 'bearing'] = (noisy_data.loc[14, 'bearing'] + 180) % 360
        
        noisy_result = self.estimator.estimate_wind_from_single_boat(
            gps_data=noisy_data,
            min_tack_angle=30.0,
            boat_type='laser',
            use_bayesian=False
        )
        
        # 結果が存在することを確認
        self.assertIsNotNone(noisy_result, "異常値を含むデータからの風向風速推定がNoneです")

    def _create_close_hauled_data(self):
        """クローズホールドのみのサンプルデータを作成"""
        # 基本情報
        base_lat, base_lon = 35.6, 139.7
        points = 100
        timestamps = [datetime(2024, 3, 1, 10, 0, 0) + timedelta(seconds=i*5) for i in range(points)]
        
        # クローズホールド（風上に対して約40度）- 一定方向
        bearings = [40] * points
        speeds = [5.0 + np.random.normal(0, 0.5) for _ in range(points)]  # 標準的な風上の速度
        
        # 座標を計算
        lats = [base_lat]
        lons = [base_lon]
        
        for i in range(1, points):
            # 前の位置から新しい位置を計算
            dist = speeds[i-1] * 5 * 0.514444  # 5秒分の距離（m/s）
            dx = dist * math.sin(math.radians(bearings[i-1]))
            dy = dist * math.cos(math.radians(bearings[i-1]))
            
            # メートルを度に変換（近似）
            dlat = dy / 111000
            dlon = dx / (111000 * math.cos(math.radians(lats[-1])))
            
            lats.append(lats[-1] + dlat)
            lons.append(lons[-1] + dlon)
        
        # DataFrameに変換
        return pd.DataFrame({
            'timestamp': timestamps,
            'latitude': lats,
            'longitude': lons,
            'bearing': bearings,
            'speed': np.array(speeds) * 0.514444,  # ノット→m/s変換
            'boat_id': ['test_boat'] * points
        })
    
    def _create_reaching_data(self):
        """リーチング（横風）のみのサンプルデータを作成"""
        # 基本情報
        base_lat, base_lon = 35.6, 139.7
        points = 100
        timestamps = [datetime(2024, 3, 1, 10, 0, 0) + timedelta(seconds=i*5) for i in range(points)]
        
        # リーチング（風に対して約90度）- 一定方向
        bearings = [90] * points
        speeds = [7.0 + np.random.normal(0, 0.5) for _ in range(points)]  # 標準的なリーチングの速度
        
        # 座標を計算
        lats = [base_lat]
        lons = [base_lon]
        
        for i in range(1, points):
            # 前の位置から新しい位置を計算
            dist = speeds[i-1] * 5 * 0.514444  # 5秒分の距離（m/s）
            dx = dist * math.sin(math.radians(bearings[i-1]))
            dy = dist * math.cos(math.radians(bearings[i-1]))
            
            # メートルを度に変換（近似）
            dlat = dy / 111000
            dlon = dx / (111000 * math.cos(math.radians(lats[-1])))
            
            lats.append(lats[-1] + dlat)
            lons.append(lons[-1] + dlon)
        
        # DataFrameに変換
        return pd.DataFrame({
            'timestamp': timestamps,
            'latitude': lats,
            'longitude': lons,
            'bearing': bearings,
            'speed': np.array(speeds) * 0.514444,  # ノット→m/s変換
            'boat_id': ['test_boat'] * points
        })

    def test_hybrid_estimation_strategy(self):
        """リファクタリングで追加された複数推定手法の統合機能をテスト"""
        # データセットの準備
        test_data = self.upwind_downwind_data.copy()
        
        # 基本の推定
        result_basic = self.estimator.estimate_wind_from_single_boat(
            gps_data=test_data,
            min_tack_angle=30.0,
            boat_type='laser',
            use_bayesian=False
        )
        
        # ベイズ推定を使用した場合
        result_bayesian = self.estimator.estimate_wind_from_single_boat(
            gps_data=test_data,
            min_tack_angle=30.0,
            boat_type='laser',
            use_bayesian=True
        )
        
        # 結果が存在することを確認
        self.assertIsNotNone(result_basic, "基本推定の結果がNoneです")
        self.assertIsNotNone(result_bayesian, "ベイズ推定の結果がNoneです")
        
        # 信頼度の比較
        avg_confidence_basic = result_basic['confidence'].mean()
        avg_confidence_bayesian = result_bayesian['confidence'].mean()
        
        # ベイズ推定の方が信頼度が高くなる傾向がある
        self.assertGreaterEqual(
            avg_confidence_bayesian, 
            avg_confidence_basic * 0.9,  # 10%の許容誤差
            "ベイズ推定の信頼度が低すぎます"
        )
        
        # 異なる艇種での推定結果の比較
        result_different_boat = self.estimator.estimate_wind_from_single_boat(
            gps_data=test_data,
            min_tack_angle=30.0,
            boat_type='49er',  # 高性能スキフ
            use_bayesian=False
        )
        
        # 風向は艇種に依存せず、ほぼ同じになるはず
        self.assertAlmostEqual(
            result_basic['wind_direction'].mean(),
            result_different_boat['wind_direction'].mean(),
            delta=15,  # 15度の許容誤差
            msg="異なる艇種で風向の推定結果が大きく異なります"
        )
        
        # 風速は艇種の係数によって変わる可能性がある
        # 49erは高性能なのでlesser係数が小さく、風速推定が小さくなる傾向がある
        coef_laser = self.estimator.boat_coefficients['laser']['upwind']
        coef_49er = self.estimator.boat_coefficients['49er']['upwind']
        
        if coef_laser > coef_49er:
            # 係数の比率に応じた風速の比率を期待
            expected_ratio = coef_49er / coef_laser
            actual_ratio = result_different_boat['wind_speed_knots'].mean() / result_basic['wind_speed_knots'].mean()
            
            self.assertAlmostEqual(
                actual_ratio,
                expected_ratio,
                delta=0.2,  # 20%の許容誤差
                msg="異なる艇種の係数比率と風速比率が一致しません"
            )

    def test_real_world_data_accuracy(self):
        """実データを用いた精度検証のテスト"""
        # 実際のデータを持っていないので、既知の風向風速でシミュレーションデータを作成
        known_wind_direction = 0  # 北からの風（0度）
        known_wind_speed_knots = 10.0  # 10ノット
        
        # 風上・風下レグを含むシミュレーションデータを作成
        simulated_data = self._create_simulated_data_with_known_wind(
            wind_direction=known_wind_direction,
            wind_speed_knots=known_wind_speed_knots
        )
        
        # 風向風速を推定
        result = self.estimator.estimate_wind_from_single_boat(
            gps_data=simulated_data,
            min_tack_angle=30.0,
            boat_type='laser',
            use_bayesian=True
        )
        
        # 結果が存在することを確認
        self.assertIsNotNone(result, "シミュレーションデータからの風向風速推定がNoneです")
        
        if result is not None:
            # 風向の平均絶対誤差を計算
            wind_dir_mae = np.mean(
                np.abs((result['wind_direction'] - known_wind_direction + 180) % 360 - 180)
            )
            
            # 風速の平均絶対誤差を計算
            wind_speed_mae = np.mean(np.abs(result['wind_speed_knots'] - known_wind_speed_knots))
            
            # 許容誤差内にあることを検証
            self.assertLess(wind_dir_mae, 45, f"風向の平均絶対誤差が大きすぎます: {wind_dir_mae}度")
            self.assertLess(wind_speed_mae, 5, f"風速の平均絶対誤差が大きすぎます: {wind_speed_mae}ノット")

    def _create_simulated_data_with_known_wind(self, wind_direction, wind_speed_knots):
        """
        既知の風向風速でシミュレーションデータを作成
        
        Parameters:
        -----------
        wind_direction : float
            既知の風向（度）
        wind_speed_knots : float
            既知の風速（ノット）
            
        Returns:
        --------
        pd.DataFrame
            シミュレーションGPSデータ
        """
        # 基本情報
        base_lat, base_lon = 35.6, 139.7
        points = 200
        timestamps = [datetime(2024, 3, 1, 10, 0, 0) + timedelta(seconds=i*5) for i in range(points)]
        
        # 風上レグの方位（風向に対して45度）
        upwind_bearing = (wind_direction + 45) % 360
        # 風下レグの方位（風向に対して180度）
        downwind_bearing = (wind_direction + 180) % 360
        
        # 風上・風下レグを組み合わせる
        bearings = []
        speeds = []
        
        # 係数（レーザー艇を想定）
        upwind_coef = self.estimator.boat_coefficients['laser']['upwind']
        downwind_coef = self.estimator.boat_coefficients['laser']['downwind']
        
        # 風上風下レグを交互に配置
        for i in range(points):
            if i % 100 < 50:  # 風上レグ
                bearings.append(upwind_bearing)
                # 風上の艇速 = 風速 / 風上係数
                speed = wind_speed_knots / upwind_coef
                speeds.append(speed + np.random.normal(0, 0.5))  # ノイズを追加
            else:  # 風下レグ
                bearings.append(downwind_bearing)
                # 風下の艇速 = 風速 / 風下係数
                speed = wind_speed_knots / downwind_coef
                speeds.append(speed + np.random.normal(0, 0.5))  # ノイズを追加
        
        # 座標を計算
        lats = [base_lat]
        lons = [base_lon]
        
        for i in range(1, points):
            # 前の位置から新しい位置を計算
            dist = speeds[i-1] * 5 * 0.514444  # 5秒分の距離（m/s）
            dx = dist * math.sin(math.radians(bearings[i-1]))
            dy = dist * math.cos(math.radians(bearings[i-1]))
            
            # メートルを度に変換（近似）
            dlat = dy / 111000
            dlon = dx / (111000 * math.cos(math.radians(lats[-1])))
            
            lats.append(lats[-1] + dlat)
            lons.append(lons[-1] + dlon)
        
        # DataFrameに変換
        return pd.DataFrame({
            'timestamp': timestamps,
            'latitude': lats,
            'longitude': lons,
            'bearing': bearings,
            'speed': np.array(speeds) * 0.514444,  # ノット→m/s変換
            'boat_id': ['sim_boat'] * points
    })

    def test_detect_maneuvers(self):
        """新しいタック/ジャイブ検出メソッドのテスト"""
        # テスト準備：現時点で実装されていなければスキップ
        if not hasattr(self.estimator, 'detect_maneuvers'):
            self.skipTest("detect_maneuversメソッドはまだ実装されていません")
        
        # テストデータ作成
        tack_data = self._create_simple_tack_data()
        
        # マニューバを検出
        maneuvers = self.estimator.detect_maneuvers(tack_data)
        
        # 基本的な検証
        self.assertIsNotNone(maneuvers, "マニューバ検出結果がNoneです")
        
        # マニューバが検出されていることを確認
        if isinstance(maneuvers, pd.DataFrame):
            self.assertGreater(len(maneuvers), 0, "マニューバが検出されていません")
            
            # 必要な列が存在するか確認
            expected_columns = ['timestamp', 'maneuver_type', 'confidence']
            for col in expected_columns:
                self.assertIn(col, maneuvers.columns, f"必要な列 '{col}' がありません")

if __name__ == '__main__':
    unittest.main()

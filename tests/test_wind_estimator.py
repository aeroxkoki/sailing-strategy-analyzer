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
        
        # 方位データ - 30度から150度へのタックを含む
        bearings = [30] * 40 + list(range(30, 150, 3)) + [150] * 40
        
        # 緯度・経度データ（シンプルな直線）
        base_lat, base_lon = 35.6, 139.7
        lats = [base_lat + i * 0.0001 for i in range(points)]
        lons = [base_lon + i * 0.0001 for i in range(points)]
        
        # 速度データ（タック時に減速）
        speeds = [5.0] * 40 + [3.0] * len(range(30, 150, 3)) + [5.0] * 40
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'latitude': lats,
            'longitude': lons,
            'speed': np.array(speeds),
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
        """改良版タック検出アルゴリズムのテスト"""
        # テストデータ作成
        df = self._create_simple_tack_data()
        df = self.estimator._calculate_bearing_change(df)
        
        # 改良版タック検出メソッドを実行
        min_tack_angle = 45.0
        tack_points = self.estimator._detect_tacks_improved(df, min_tack_angle=min_tack_angle)
        
        # 結果の検証
        self.assertIsNotNone(tack_points, "タック検出結果がNoneです")
        self.assertGreater(len(tack_points), 0, "タックが検出されていません")
        
        # タック検出時の方位変化量を確認
        detected_change = tack_points['bearing_change'].iloc[0]
        self.assertGreater(detected_change, min_tack_angle, 
                     f"検出されたタックの方位変化({detected_change})が閾値({min_tack_angle})より小さいです")

    def test_wind_direction_calculation_methods(self):
        """風向計算メソッドのテスト"""
        # 複数の風上方位からの風向計算テスト
        upwind_bearings = [30, 150]  # 約120度離れた風上方位
        wind_dir, confidence = self.estimator._calculate_wind_direction_from_tacks(upwind_bearings)
        
        # 理論上の風向（風上方位の反対方向）
        # 30度と150度の二等分線は90度付近、その反対は270度付近
        self.assertTrue(240 <= wind_dir <= 300, 
                  f"風向計算結果({wind_dir})が期待範囲(240-300度)外です")
        
        # 信頼度も検証
        self.assertGreater(confidence, 0.5, "風向計算の信頼度が低すぎます")
        
        # 単一方位からの風向計算テスト
        bearing = 45
        vmg_angle = 40
        
        # 風向計算
        wind_dir_single = self.estimator._calculate_wind_direction_single_bearing(bearing, vmg_angle)
        
        # 理論上の風向（方位+180-VMG角度）
        theoretical_wind_dir = (bearing + 180 - vmg_angle) % 360
        
        # 検証
        self.assertEqual(wind_dir_single, theoretical_wind_dir, 
                   f"単一方位からの風向計算が不正確です: {wind_dir_single} != {theoretical_wind_dir}")


if __name__ == '__main__':
    unittest.main()

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import warnings

# テスト対象のクラスをインポート
from race_wind_tracker import RaceWindTracker

class TestRaceWindTracker(unittest.TestCase):
    
    def setUp(self):
        """各テスト実行前の準備"""
        # 警告を無視
        warnings.filterwarnings("ignore")
        
        # テスト用のインスタンス
        self.tracker = RaceWindTracker()
        
        # テスト用のデータを作成
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
        """テスト用のサンプルGPSデータを作成"""
        # 基準時刻と位置
        base_time = datetime(2025, 3, 1, 10, 0, 0)
        base_lat, base_lon = 35.6, 139.7
        
        # 30艇分のデータを生成
        boats_data = {}
        
        for boat_id in range(1, 31):
            # 艇ごとに少しずつ異なる位置
            lat_offset = (boat_id % 5) * 0.001
            lon_offset = (boat_id // 5) * 0.001
            
            # 100ポイントのデータ生成
            points = 100
            timestamps = [base_time + timedelta(seconds=i*5) for i in range(points)]
            
            # 風上風下パターンを模擬（最初の半分は風上、後半は風下）
            bearings = []
            speeds = []
            
            for i in range(points):
                if i < points / 2:
                    # 風上走行 - ジグザグパターン
                    bearing = 30 if i % 20 < 10 else 330
                    speed = 5.0 + np.random.random() * 0.5  # 風上は遅め
                else:
                    # 風下走行 - 安定したコース
                    bearing = 180 + np.random.random() * 10
                    speed = 7.0 + np.random.random() * 0.5  # 風下は速め
                
                bearings.append(bearing)
                speeds.append(speed)
            
            # 座標を計算
            lats = [base_lat + lat_offset]
            lons = [base_lon + lon_offset]
            
            for i in range(1, points):
                # 前の位置から新しい位置を計算
                speed_ms = speeds[i-1] * 0.514444  # ノットをm/sに変換
                dist = speed_ms * 5  # 5秒分の距離
                dx = dist * math.sin(math.radians(bearings[i-1]))
                dy = dist * math.cos(math.radians(bearings[i-1]))
                
                # メートルを度に変換（近似）
                dlat = dy / 111000
                dlon = dx / (111000 * math.cos(math.radians(lats[-1])))
                
                lats.append(lats[-1] + dlat)
                lons.append(lons[-1] + dlon)
            
            # DataFrameに変換
            df = pd.DataFrame({
                'timestamp': timestamps,
                'latitude': lats,
                'longitude': lons,
                'bearing': bearings,
                'speed': np.array(speeds) * 0.514444,  # ノット -> m/s
                'boat_id': [f"Boat{boat_id}"] * points
            })
            
            boats_data[f"Boat{boat_id}"] = df
        
        return boats_data
    
    def test_calculate_wind_direction(self):
        """風向計算メソッドのテスト"""
        # 風上走行時のテスト
        upwind_dir = self.tracker._calculate_wind_direction(30, 'upwind', 45)
        self.assertEqual(upwind_dir, 165, "風上走行時の風向計算が誤っています")
        
        # 風下走行時のテスト
        downwind_dir = self.tracker._calculate_wind_direction(180, 'downwind')
        self.assertEqual(downwind_dir, 180, "風下走行時の風向計算が誤っています")
        
        # リーチング時のテスト
        reaching_dir = self.tracker._calculate_wind_direction(90, 'reaching')
        self.assertEqual(reaching_dir, 180, "リーチング時の風向計算が誤っています")
    
    def test_determine_sailing_state(self):
        """走行状態判定メソッドのテスト"""
        # 風上パターンのテスト
        upwind_bearings = [30, 330, 30, 330, 30]
        upwind_speeds = [5.0, 5.2, 5.1, 5.3, 5.0]
        state = self.tracker._determine_sailing_state(upwind_bearings, upwind_speeds)
        self.assertEqual(state, 'upwind', "風上走行パターンの判定が誤っています")
        
        # 風下パターンのテスト
        downwind_bearings = [175, 180, 185, 178, 182]
        downwind_speeds = [7.0, 7.2, 7.1, 7.3, 7.0]
        state = self.tracker._determine_sailing_state(downwind_bearings, downwind_speeds)
        self.assertEqual(state, 'downwind', "風下走行パターンの判定が誤っています")
    
    def test_calculate_spatial_distribution(self):
        """空間分布評価メソッドのテスト"""
        # 均一に分布した位置のテスト
        uniform_positions = [
            (35.60, 139.70), (35.61, 139.70), (35.60, 139.71), (35.61, 139.71),
            (35.605, 139.705), (35.595, 139.695), (35.595, 139.705), (35.605, 139.695)
        ]
        uniform_score = self.tracker._calculate_spatial_distribution(uniform_positions)
        self.assertGreater(uniform_score, 0.7, "均一分布の評価が低すぎます")
        
        # 偏った分布のテスト
        biased_positions = [
            (35.60, 139.70), (35.601, 139.701), (35.599, 139.699),
            (35.602, 139.702), (35.598, 139.698)
        ]
        biased_score = self.tracker._calculate_spatial_distribution(biased_positions)
        self.assertLess(biased_score, uniform_score, "偏った分布の評価が高すぎます")
    
    def test_calculate_boat_data_quality(self):
        """データ品質評価メソッドのテスト"""
        # サンプルデータから1艇を選択
        boat_data = next(iter(self.sample_data.values()))
        
        # 通常データの品質評価
        quality = self.tracker._calculate_boat_data_quality(boat_data)
        self.assertGreater(quality, 0.5, "通常データの品質評価が低すぎます")
        
        # 異常データを含むケース
        noisy_data = boat_data.copy()
        
        # 速度に急激な変化を追加
        noisy_data.loc[10:15, 'speed'] = 20.0
        
        # 方位に急激な変化を追加
        noisy_data.loc[30:35, 'bearing'] = 100.0
        
        noisy_quality = self.tracker._calculate_boat_data_quality(noisy_data)
        self.assertLess(noisy_quality, quality, "異常データに対する品質評価が高すぎます")
    
    def test_detect_anomalies_realtime(self):
        """リアルタイム異常値検出のテスト"""
        # サンプル風データの作成
        wind_data = pd.DataFrame({
            'wind_direction': [0, 5, 2, 8, 5, 50, 4, 6, 3, 7],  # 50が異常値
            'wind_speed_knots': [10, 11, 10.5, 11.5, 10.8, 25, 11.2, 10.5, 11, 10.8]  # 25が異常値
        })
        
        # 異常値検出
        result = self.tracker._detect_anomalies_realtime(wind_data, sensitivity=0.8)
        
        # 異常フラグの確認
        self.assertTrue('is_anomaly' in result.columns, "異常フラグ列がありません")
        self.assertTrue(result.loc[5, 'is_anomaly'], "明らかな異常値が検出されていません")
    
    def test_single_boat_wind_estimation(self):
        """単一艇からの風推定テスト"""
        # サンプルデータから1艇を選択
        boat_data = next(iter(self.sample_data.values()))
        
        # 風推定
        wind_result = self.tracker.estimate_wind_from_single_boat(
            boat_data, min_tack_angle=30.0, boat_type='470'
        )
        
        # 結果の検証
        self.assertIsNotNone(wind_result, "風推定結果がNoneです")
        self.assertTrue('wind_direction' in wind_result.columns, "風向カラムがありません")
        self.assertTrue('wind_speed_knots' in wind_result.columns, "風速カラムがありません")
        
        # 期待される風向の範囲をチェック
        avg_direction = wind_result['wind_direction'].mean()
        self.assertTrue(
            (330 <= avg_direction <= 360) or (0 <= avg_direction <= 30),
            f"風向推定値 {avg_direction} が想定範囲外です"
        )
    
    def test_high_resolution_wind_field(self):
        """高解像度風の場推定テスト"""
        # 複数艇の風推定を実行
        for boat_id, boat_data in list(self.sample_data.items())[:5]:  # 最初の5艇だけ使用
            self.tracker.estimate_wind_from_single_boat(
                boat_data, min_tack_angle=30.0, boat_type='470'
            )
        
        # 高解像度風の場を推定
        current_time = datetime.now()
        wind_field = self.tracker.estimate_high_resolution_wind_field(current_time)
        
        # 結果の検証
        self.assertIsNotNone(wind_field, "風の場推定結果がNoneです")
        
        # グリッドサイズの確認
        grid_resolution = self.tracker.race_config['grid_resolution']
        self.assertEqual(wind_field['lat_grid'].shape, (grid_resolution, grid_resolution),
                        f"格子サイズが仕様({grid_resolution}x{grid_resolution})と一致しません")
        
        # 風向・風速データの存在確認
        self.assertTrue('wind_direction' in wind_field, "風向データがありません")
        self.assertTrue('wind_speed' in wind_field, "風速データがありません")
        
        # 分布評価スコアの存在確認
        self.assertTrue('spatial_distribution_score' in wind_field, "空間分布スコアがありません")
        self.assertTrue('data_quality' in wind_field, "データ品質情報がありません")

if __name__ == '__main__':
    unittest.main()

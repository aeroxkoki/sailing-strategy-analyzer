"""
OptimalVMGCalculator クラスのテスト
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import json
import warnings
import matplotlib.pyplot as plt

# テスト対象のモジュールをインポート
from sailing_data_processor.optimal_vmg_calculator import OptimalVMGCalculator


class TestOptimalVMGCalculator(unittest.TestCase):
    """OptimalVMGCalculatorクラスのテストケース"""
    
    def setUp(self):
        """各テストケース実行前のセットアップ"""
        # 警告を無視
        warnings.filterwarnings("ignore")
        
        # テスト用のオブジェクト作成
        self.calculator = OptimalVMGCalculator()
        
        # テスト用の風向風速データを読み込む
        self.wind_field = self._load_test_wind_field()
        
        # 風向風速データを設定
        if self.wind_field:
            self.calculator.set_wind_field(self.wind_field)
        
    def _load_test_wind_field(self):
        """テスト用の風向風速データを読み込む"""
        # モジュールのディレクトリを基準にテストデータディレクトリを特定
        module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_data_path = os.path.join(module_dir, 'tests', 'data', 'test_wind_field.json')
        
        try:
            with open(test_data_path, 'r') as f:
                wind_data = json.load(f)
                
            # JSON データを風向風速フィールド形式に変換
            lat_vals = np.linspace(wind_data['bounds']['min_lat'], 
                                  wind_data['bounds']['max_lat'], 
                                  wind_data['grid_size'][0])
            lon_vals = np.linspace(wind_data['bounds']['min_lon'], 
                                  wind_data['bounds']['max_lon'], 
                                  wind_data['grid_size'][1])
            
            grid_lats, grid_lons = np.meshgrid(lat_vals, lon_vals)
            
            # 風向と風速の空のグリッドを作成
            wind_dir = np.zeros(grid_lats.shape)
            wind_speed = np.zeros(grid_lats.shape)
            confidence = np.ones(grid_lats.shape) * 0.9
            
            # データポイントから格子点を設定
            for point in wind_data['data_points']:
                # 最も近い格子点を見つける
                lat_idx = np.abs(lat_vals - point['lat']).argmin()
                lon_idx = np.abs(lon_vals - point['lon']).argmin()
                
                wind_dir[lon_idx, lat_idx] = point['wind_direction']
                wind_speed[lon_idx, lat_idx] = point['wind_speed']
                confidence[lon_idx, lat_idx] = point.get('confidence', 0.9)
            
            # その他の格子点を補完（単純な線形補間）
            from scipy.interpolate import griddata
            
            points = [(lon_vals[j], lat_vals[i]) 
                     for i in range(len(lat_vals)) 
                     for j in range(len(lon_vals))
                     if wind_dir[j, i] != 0]
            
            values_dir = [wind_dir[j, i] 
                        for i in range(len(lat_vals)) 
                        for j in range(len(lon_vals))
                        if wind_dir[j, i] != 0]
            
            values_speed = [wind_speed[j, i] 
                          for i in range(len(lat_vals)) 
                          for j in range(len(lon_vals))
                          if wind_dir[j, i] != 0]
            
            if points and values_dir and values_speed:
                xi = np.array([(lon_vals[j], lat_vals[i]) 
                             for i in range(len(lat_vals)) 
                             for j in range(len(lon_vals))])
                
                # 補間
                interp_dir = griddata(points, values_dir, xi, method='linear')
                interp_speed = griddata(points, values_speed, xi, method='linear')
                
                # 結果を再度グリッドに変換
                for idx, (j, i) in enumerate([(j, i) 
                                           for i in range(len(lat_vals)) 
                                           for j in range(len(lon_vals))]):
                    if not np.isnan(interp_dir[idx]):
                        wind_dir[j, i] = interp_dir[idx]
                    if not np.isnan(interp_speed[idx]):
                        wind_speed[j, i] = interp_speed[idx]
            
            # 風の場データ構造を作成
            return {
                'lat_grid': grid_lats,
                'lon_grid': grid_lons,
                'wind_direction': wind_dir,
                'wind_speed': wind_speed,
                'confidence': confidence,
                'time': datetime.strptime(wind_data['timestamp'], "%Y-%m-%dT%H:%M:%S")
            }
            
        except Exception as e:
            print(f"テスト用風向風速データの読み込みエラー: {e}")
            return None
    
    def _load_test_waypoints(self):
        """テスト用のウェイポイントデータを読み込む"""
        # モジュールのディレクトリを基準にテストデータディレクトリを特定
        module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_data_path = os.path.join(module_dir, 'tests', 'data', 'test_waypoints.json')
        
        try:
            with open(test_data_path, 'r') as f:
                waypoints_data = json.load(f)
            return waypoints_data.get('course', [])
        except Exception as e:
            print(f"テスト用ウェイポイントデータの読み込みエラー: {e}")
            return []
    
    def test_load_polar_data(self):
        """ポーラーデータの読み込みテスト"""
        # モジュールのディレクトリを基準にテストデータディレクトリを特定
        module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_data_path = os.path.join(module_dir, 'tests', 'data', 'test_boat_polar.csv')
        
        # ポーラーデータの読み込み
        success = self.calculator.load_polar_data('test_boat', test_data_path)
        
        # 検証
        self.assertTrue(success, "ポーラーデータの読み込みに失敗")
        self.assertIn('test_boat', self.calculator.boat_types, "test_boatが艇種リストに追加されていない")
        
        # ポーラーデータの内容確認
        boat_data = self.calculator.boat_types['test_boat']
        self.assertIn('polar_data', boat_data, "polar_dataがない")
        self.assertIn('upwind_optimal', boat_data, "upwind_optimalがない")
        self.assertIn('downwind_optimal', boat_data, "downwind_optimalがない")
    
    def test_get_boat_performance(self):
        """艇性能計算のテスト"""
        # 標準艇種に対するテスト
        for boat_type in ['laser', '470', '49er']:
            if boat_type in self.calculator.boat_types:
                # 風上での性能を確認
                upwind_speed = self.calculator.get_boat_performance(boat_type, 10.0, 45.0)
                self.assertGreater(upwind_speed, 0, f"{boat_type}の風上性能が0以下")
                
                # 風下での性能を確認
                downwind_speed = self.calculator.get_boat_performance(boat_type, 10.0, 150.0)
                self.assertGreater(downwind_speed, 0, f"{boat_type}の風下性能が0以下")
                
                # 風上より風下の方が通常速い
                self.assertGreaterEqual(downwind_speed, upwind_speed * 0.8, 
                                      f"{boat_type}の風下性能が風上性能の80%未満")
    
    def test_find_optimal_twa(self):
        """最適風向角算出のテスト"""
        # 標準艇種に対するテスト
        for boat_type in ['laser', '470', '49er']:
            if boat_type in self.calculator.boat_types:
                # 風上での最適角度
                upwind_angle, upwind_vmg = self.calculator.find_optimal_twa(boat_type, 10.0, upwind=True)
                self.assertGreater(upwind_angle, 0, f"{boat_type}の風上最適角度が0以下")
                self.assertLess(upwind_angle, 90, f"{boat_type}の風上最適角度が90度以上")
                
                # 風下での最適角度
                downwind_angle, downwind_vmg = self.calculator.find_optimal_twa(boat_type, 10.0, upwind=False)
                self.assertGreater(downwind_angle, 90, f"{boat_type}の風下最適角度が90度以下")
                self.assertLess(downwind_angle, 180, f"{boat_type}の風下最適角度が180度以上")
    
    def test_calculate_optimal_vmg(self):
        """最適VMG計算のテスト"""
        if not self.wind_field:
            self.skipTest("風向風速データが使用できないためテストをスキップ")
        
        # 風向風速データのグリッド中心付近の緯度経度
        center_lat = np.mean(self.wind_field['lat_grid'])
        center_lon = np.mean(self.wind_field['lon_grid'])
        
        # 東に1キロメートル移動した地点の緯度経度（概算）
        target_lon = center_lon + 0.01  # 約1km
        target_lat = center_lat
        
        # 最適VMGを計算
        result = self.calculator.calculate_optimal_vmg('laser', center_lat, center_lon, target_lat, target_lon)
        
        # 結果の検証
        self.assertIsNotNone(result, "最適VMG計算結果がNone")
        self.assertIn('optimal_course', result, "optimal_courseがない")
        self.assertIn('boat_speed', result, "boat_speedがない")
        self.assertIn('vmg', result, "vmgがない")
        
        # 速度は0より大きいはず
        self.assertGreater(result['boat_speed'], 0, "艇速が0以下")
        self.assertGreaterEqual(result['vmg'], 0, "VMGが0未満")
    
    def test_find_optimal_path(self):
        """最適パス計算のテスト"""
        if not self.wind_field:
            self.skipTest("風向風速データが使用できないためテストをスキップ")
        
        # 風向風速データのグリッド中心付近の緯度経度
        center_lat = np.mean(self.wind_field['lat_grid'])
        center_lon = np.mean(self.wind_field['lon_grid'])
        
        # 北東に2キロメートル移動した地点の緯度経度（概算）
        target_lat = center_lat + 0.01  # 約1km
        target_lon = center_lon + 0.01  # 約1km
        
        # 最適パスを計算
        result = self.calculator.find_optimal_path('laser', center_lat, center_lon, target_lat, target_lon)
        
        # 結果の検証
        self.assertIsNotNone(result, "最適パス計算結果がNone")
        self.assertIn('path_points', result, "path_pointsがない")
        self.assertIn('total_distance', result, "total_distanceがない")
        self.assertIn('total_time', result, "total_timeがない")
        
        # パスポイントがあるはず
        self.assertGreater(len(result['path_points']), 0, "パスポイントがない")
        
        # 距離と時間は0より大きいはず
        self.assertGreater(result['total_distance'], 0, "総距離が0以下")
        self.assertGreater(result['total_time'], 0, "総時間が0以下")
    
    def test_calculate_optimal_route_for_course(self):
        """コース最適戦略計算のテスト"""
        if not self.wind_field:
            self.skipTest("風向風速データが使用できないためテストをスキップ")
        
        # テスト用のウェイポイントデータを読み込む
        waypoints = self._load_test_waypoints()
        if not waypoints:
            self.skipTest("ウェイポイントデータが使用できないためテストをスキップ")
        
        # コース最適戦略を計算
        result = self.calculator.calculate_optimal_route_for_course('laser', waypoints)
        
        # 結果の検証
        self.assertIsNotNone(result, "コース最適戦略計算結果がNone")
        self.assertIn('legs', result, "legsがない")
        self.assertIn('total_time', result, "total_timeがない")
        self.assertIn('total_distance', result, "total_distanceがない")
        
        # レッグ数はウェイポイント数-1
        self.assertEqual(len(result['legs']), len(waypoints) - 1, "レッグ数がウェイポイント数-1と一致しない")
        
        # 各レッグの検証
        for leg in result['legs']:
            self.assertIn('path', leg, "レッグにpathがない")
            self.assertIn('leg_type', leg, "レッグにleg_typeがない")
    
    def test_visualize_optimal_path(self):
        """最適パスの可視化テスト"""
        if not self.wind_field:
            self.skipTest("風向風速データが使用できないためテストをスキップ")
        
        # 風向風速データのグリッド中心付近の緯度経度
        center_lat = np.mean(self.wind_field['lat_grid'])
        center_lon = np.mean(self.wind_field['lon_grid'])
        
        # 北東に2キロメートル移動した地点の緯度経度（概算）
        target_lat = center_lat + 0.01  # 約1km
        target_lon = center_lon + 0.01  # 約1km
        
        # 最適パスを計算
        path_data = self.calculator.find_optimal_path('laser', center_lat, center_lon, target_lat, target_lon)
        
        if not path_data or 'path_points' not in path_data or not path_data['path_points']:
            self.skipTest("パスデータが使用できないためテストをスキップ")
        
        # 可視化
        fig = self.calculator.visualize_optimal_path(path_data, show_wind=True)
        
        # 結果の検証
        self.assertIsNotNone(fig, "可視化結果がNone")
        
        # プロットをクローズ
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()

# test_sailing_visualizer.py

import unittest
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, '/content/drive/MyDrive/sailing-project')

from visualization.sailing_visualizer import SailingVisualizer

class TestSailingVisualizer(unittest.TestCase):
    def test_init(self):
        visualizer = SailingVisualizer()
        self.assertIsNone(visualizer.data_processor)
        self.assertEqual(visualizer.boats_data, {})
        self.assertIsNone(visualizer.course_data)
        self.assertIsNone(visualizer.map_object)
        self.assertIsNotNone(visualizer.colors)
    
    def test_set_data_processor(self):
        """データ処理モジュール設定のテスト"""
        visualizer = SailingVisualizer()
        
        # モックのデータプロセッサを作成
        class MockDataProcessor:
            pass
        
        mock_processor = MockDataProcessor()
        
        # データプロセッサを設定
        visualizer.set_data_processor(mock_processor)
        
        # 設定されたことを確認
        self.assertEqual(visualizer.data_processor, mock_processor)
    
    def test_load_boat_data(self):
        """ボートデータ読み込みのテスト"""
        visualizer = SailingVisualizer()
        
        # テスト用のデータフレームを作成
        test_data = pd.DataFrame({
            'latitude': [35.1, 35.2, 35.3],
            'longitude': [139.1, 139.2, 139.3],
            'timestamp': pd.to_datetime(['2025-03-01 10:00:00', '2025-03-01 10:05:00', '2025-03-01 10:10:00']),
            'speed': [5.1, 5.2, 5.3]
        })
        
        # データの読み込み
        result = visualizer.load_boat_data('test_boat', data=test_data)
        
        # 読み込みが成功したことを確認
        self.assertTrue(result)
        self.assertIn('test_boat', visualizer.boats_data)
        self.assertTrue(visualizer.boats_data['test_boat'].equals(test_data))
    
    def test_load_course_data(self):
        """コースデータ読み込みのテスト"""
        visualizer = SailingVisualizer()
        
        # テスト用のコースデータを作成
        test_course_data = {
            'mark1': {'latitude': 35.1, 'longitude': 139.1},
            'mark2': {'latitude': 35.2, 'longitude': 139.2}
        }
        
        # データの読み込み
        result = visualizer.load_course_data(course_data=test_course_data)
        
        # 読み込みが成功したことを確認
        self.assertTrue(result)
        self.assertEqual(visualizer.course_data, test_course_data)
    
    def test_create_base_map(self):
        """ベースマップ作成のテスト"""
        visualizer = SailingVisualizer()
        
        # マップの作成
        map_object = visualizer.create_base_map()
        
        # マップが作成されたことを確認
        self.assertIsNotNone(map_object)
        self.assertEqual(visualizer.map_object, map_object)
    
    def test_visualize_single_boat(self):
        """単一ボート表示のテスト"""
        visualizer = SailingVisualizer()
        
        # テスト用のデータを作成
        test_data = pd.DataFrame({
            'latitude': [35.1, 35.2, 35.3],
            'longitude': [139.1, 139.2, 139.3],
            'timestamp': pd.to_datetime(['2025-03-01 10:00:00', '2025-03-01 10:05:00', '2025-03-01 10:10:00']),
            'speed': [5.1, 5.2, 5.3]
        })
        
        # データの読み込み
        visualizer.load_boat_data('test_boat', data=test_data)
        
        # 単一ボート表示
        map_object = visualizer.visualize_single_boat('test_boat')
        
        # マップが作成されたことを確認
        self.assertIsNotNone(map_object)
    
    def test_visualize_all_boats(self):
        """全ボート表示のテスト"""
        visualizer = SailingVisualizer()
        
        # テスト用の2つのボートデータを作成
        boat1_data = pd.DataFrame({
            'latitude': [35.1, 35.2, 35.3],
            'longitude': [139.1, 139.2, 139.3],
            'timestamp': pd.to_datetime(['2025-03-01 10:00:00', '2025-03-01 10:05:00', '2025-03-01 10:10:00']),
            'speed': [5.1, 5.2, 5.3]
        })
        
        boat2_data = pd.DataFrame({
            'latitude': [35.2, 35.3, 35.4],
            'longitude': [139.2, 139.3, 139.4],
            'timestamp': pd.to_datetime(['2025-03-01 10:02:00', '2025-03-01 10:07:00', '2025-03-01 10:12:00']),
            'speed': [6.1, 6.2, 6.3]
        })
        
        # ボートデータの追加
        visualizer.load_boat_data('boat1', data=boat1_data)
        visualizer.load_boat_data('boat2', data=boat2_data)
        
        # 全ボート表示
        map_object = visualizer.visualize_all_boats()
        
        # マップが作成されたことを確認
        self.assertIsNotNone(map_object)
    
    def test_save_map(self):
        """地図保存のテスト"""
        visualizer = SailingVisualizer()
        
        # マップが存在しない場合は保存に失敗する
        result = visualizer.save_map()
        self.assertFalse(result)
        
        # マップを作成して保存
        visualizer.create_base_map()
        
        # 実際の保存はテスト環境によって異なるので、ここではモックします
        # 実際のテストでは一時ファイルを使用するか、モックライブラリを使用します
    
    def test_visualize_multiple_boats(self):
        """複数艇表示機能のテスト"""
        visualizer = SailingVisualizer()
        
        # テスト用の2つのボートデータを作成
        boat1_data = pd.DataFrame({
            'latitude': [35.1, 35.2, 35.3],
            'longitude': [139.1, 139.2, 139.3],
            'timestamp': pd.to_datetime(['2025-03-01 10:00:00', '2025-03-01 10:05:00', '2025-03-01 10:10:00']),
            'speed': [5.1, 5.2, 5.3]
        })
        
        boat2_data = pd.DataFrame({
            'latitude': [35.2, 35.3, 35.4],
            'longitude': [139.2, 139.3, 139.4],
            'timestamp': pd.to_datetime(['2025-03-01 10:02:00', '2025-03-01 10:07:00', '2025-03-01 10:12:00']),
            'speed': [6.1, 6.2, 6.3]
        })
        
        # ボートデータの追加
        visualizer.load_boat_data('boat1', data=boat1_data)
        visualizer.load_boat_data('boat2', data=boat2_data)
        
        # 複数艇表示（ラベルなし、時間同期なし）
        map1 = visualizer.visualize_multiple_boats(['boat1', 'boat2'], show_labels=False, sync_time=False)
        self.assertIsNotNone(map1)
        
        # 複数艇表示（ラベルあり、時間同期なし）
        map2 = visualizer.visualize_multiple_boats(['boat1', 'boat2'], show_labels=True, sync_time=False)
        self.assertIsNotNone(map2)
        
        # 複数艇表示（ラベルあり、時間同期あり）
        map3 = visualizer.visualize_multiple_boats(['boat1', 'boat2'], show_labels=True, sync_time=True)
        self.assertIsNotNone(map3)
    
    def test_create_performance_summary(self):
        """パフォーマンス指標サマリー作成のテスト"""
        visualizer = SailingVisualizer()
        
        # テスト用のボートデータを作成
        boat_data = pd.DataFrame({
            'latitude': [35.1, 35.2, 35.3],
            'longitude': [139.1, 139.2, 139.3],
            'timestamp': pd.to_datetime(['2025-03-01 10:00:00', '2025-03-01 10:05:00', '2025-03-01 10:10:00']),
            'speed': [5.1, 5.2, 5.3],
            'course': [45, 50, 130],  # 3番目の値を大きく変えてタックを検出できるようにする
            'wind_direction': [90, 90, 90]
        })
        
        # ボートデータの追加
        visualizer.load_boat_data('test_boat', data=boat_data)
        
        # パフォーマンスサマリーの作成
        summary = visualizer.create_performance_summary('test_boat')
        
        # サマリーが作成されたことを確認
        self.assertIsNotNone(summary)
        
        # 基本的な統計情報が含まれていることを確認
        self.assertIn('speed', summary)
        
        # タックが検出されていることを確認
        self.assertIn('tack_count', summary)
        self.assertEqual(summary['tack_count'], 1)  # 1回のタックが検出されるはず

if __name__ == '__main__':
    unittest.main()

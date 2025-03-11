# test_map_display.py

import unittest
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, '/content/drive/MyDrive/sailing-project')

from visualization.map_display import SailingMapDisplay

class TestSailingMapDisplay(unittest.TestCase):
    def test_init(self):
        """初期化のテスト"""
        map_display = SailingMapDisplay()
        self.assertIsNone(map_display.map_object)
        self.assertIsNotNone(map_display.colors)
        self.assertIsNotNone(map_display.available_tiles)
        self.assertEqual(map_display.default_tile, "CartoDB positron")
    
    def test_create_map(self):
        """地図作成のテスト"""
        map_display = SailingMapDisplay()
        
        # デフォルト設定での地図作成
        map_object = map_display.create_map()
        self.assertIsNotNone(map_object)
        self.assertIsInstance(map_object, folium.Map)
        
        # カスタム設定での地図作成
        custom_center = (34.5, 138.5)
        custom_zoom = 10
        custom_tile = "オープンストリートマップ"
        
        map_object = map_display.create_map(
            center=custom_center,
            zoom_start=custom_zoom,
            tile=custom_tile
        )
        
        self.assertIsNotNone(map_object)
        self.assertEqual(map_object.location, custom_center)
        self.assertEqual(map_object.zoom_start, custom_zoom)
    
    def test_add_track(self):
        """航跡追加のテスト"""
        map_display = SailingMapDisplay()
        
        # 地図の作成
        map_object = map_display.create_map()
        
        # テスト用のデータフレーム
        test_data = pd.DataFrame({
            'latitude': [35.1, 35.2, 35.3],
            'longitude': [139.1, 139.2, 139.3],
            'timestamp': pd.to_datetime(['2025-03-01 10:00:00', '2025-03-01 10:05:00', '2025-03-01 10:10:00']),
            'speed': [5.1, 5.2, 5.3]
        })
        
        # 航跡の追加
        result_map = map_display.add_track(test_data, "テストボート")
        
        # 戻り値が正しいことを確認
        self.assertIsNotNone(result_map)
        self.assertIsInstance(result_map, folium.Map)
        
        # 指定した色での航跡の追加
        custom_color = "purple"
        result_map = map_display.add_track(test_data, "テストボート2", color=custom_color)
        self.assertIsNotNone(result_map)
    
    def test_add_track_without_map(self):
        """地図作成前の航跡追加テスト（例外発生を期待）"""
        map_display = SailingMapDisplay()
        
        # テスト用のデータフレーム
        test_data = pd.DataFrame({
            'latitude': [35.1, 35.2, 35.3],
            'longitude': [139.1, 139.2, 139.3]
        })
        
        # 地図が作成されていない状態で航跡を追加しようとすると例外が発生するはず
        with self.assertRaises(ValueError):
            map_display.add_track(test_data, "テストボート")
    
    def test_add_course_marks(self):
        """コースマーク追加のテスト"""
        map_display = SailingMapDisplay()
        
        # 地図の作成
        map_object = map_display.create_map()
        
        # テスト用のマークデータ（DataFrame）
        marks_df = pd.DataFrame({
            'name': ['Mark1', 'Mark2'],
            'latitude': [35.1, 35.2],
            'longitude': [139.1, 139.2]
        })
        
        # マークの追加（DataFrame）
        result_map = map_display.add_course_marks(marks_df)
        self.assertIsNotNone(result_map)
        
        # テスト用のマークデータ（辞書）
        marks_dict = {
            'Mark3': {'latitude': 35.3, 'longitude': 139.3},
            'Mark4': {'latitude': 35.4, 'longitude': 139.4}
        }
        
        # マークの追加（辞書）
        result_map = map_display.add_course_marks(marks_dict)
        self.assertIsNotNone(result_map)
    
    def test_add_start_finish_line(self):
        """スタート/フィニッシュライン追加のテスト"""
        map_display = SailingMapDisplay()
        
        # 地図の作成
        map_object = map_display.create_map()
        
        # テスト用のライン座標
        start_line = [(35.1, 139.1), (35.15, 139.15)]
        finish_line = [(35.2, 139.2), (35.25, 139.25)]
        
        # スタートラインのみ追加
        result_map = map_display.add_start_finish_line(start_line=start_line)
        self.assertIsNotNone(result_map)
        
        # フィニッシュラインのみ追加
        result_map = map_display.add_start_finish_line(finish_line=finish_line)
        self.assertIsNotNone(result_map)
        
        # 両方追加
        result_map = map_display.add_start_finish_line(start_line=start_line, finish_line=finish_line)
        self.assertIsNotNone(result_map)
    
    def test_add_wind_direction(self):
        """風向矢印追加のテスト"""
        map_display = SailingMapDisplay()
        
        # 地図の作成
        map_object = map_display.create_map()
        
        # 風向の追加
        result_map = map_display.add_wind_direction(35.1, 139.1, 45)
        self.assertIsNotNone(result_map)
        
        # 風速付きの風向追加
        result_map = map_display.add_wind_direction(35.2, 139.2, 90, 15)
        self.assertIsNotNone(result_map)
    
    def test_add_speed_heatmap(self):
        """速度ヒートマップ追加のテスト"""
        map_display = SailingMapDisplay()
        
        # 地図の作成
        map_object = map_display.create_map()
        
        # テスト用のデータフレーム
        test_data = pd.DataFrame({
            'latitude': [35.1, 35.2, 35.3, 35.4],
            'longitude': [139.1, 139.2, 139.3, 139.4],
            'speed': [5.1, 7.2, 6.3, 8.4]
        })
        
        # ヒートマップの追加
        result_map = map_display.add_speed_heatmap(test_data)
        self.assertIsNotNone(result_map)
    
    def test_add_speed_heatmap_missing_columns(self):
        """必要な列がない場合のヒートマップテスト"""
        map_display = SailingMapDisplay()
        
        # 地図の作成
        map_object = map_display.create_map()
        
        # 速度列が欠けたデータフレーム
        invalid_data = pd.DataFrame({
            'latitude': [35.1, 35.2],
            'longitude': [139.1, 139.2]
            # 'speed' 列が欠けている
        })
        
        # 例外が発生するはず
        with self.assertRaises(ValueError):
            map_display.add_speed_heatmap(invalid_data)

if __name__ == '__main__':
    unittest.main()

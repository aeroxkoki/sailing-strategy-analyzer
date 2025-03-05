import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import os
import sys
import tempfile
import warnings

# モジュールのインポートパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sailing_data_processor import SailingDataProcessor

class TestSailingDataProcessor(unittest.TestCase):
    """SailingDataProcessorクラスのテストケース"""
    
    def setUp(self):
        """各テストケース実行前のセットアップ"""
        # 警告を無視
        warnings.filterwarnings("ignore")
        
        # テスト用のデータプロセッサインスタンス
        self.processor = SailingDataProcessor()
        
        # テスト用のデータを作成
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
        """テスト用のサンプルGPSデータを作成"""
        # 3艇分のデータを作成
        data = {}
        
        for boat_id in range(1, 4):
            # 基本の座標
            base_lat, base_lon = 35.6, 139.7
            
            # 100ポイントのデータ生成
            points = 100
            timestamps = [datetime(2024, 3, 1, 10, 0, 0) + timedelta(seconds=i*5) for i in range(points)]
            
            lats = [base_lat + i * 0.001 + boat_id * 0.0001 for i in range(points)]
            lons = [base_lon + i * 0.0005 + boat_id * 0.0002 for i in range(points)]
            speeds = [5.0 + boat_id * 0.5 + np.sin(i/10) for i in range(points)]
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                'latitude': lats,
                'longitude': lons,
                'speed': np.array(speeds) * 0.514444,  # ノット -> m/s
                'boat_id': [f"Boat{boat_id}"] * points
            })
            
            data[f"Boat{boat_id}"] = df
        
        return data
    
    def _generate_csv_content(self, df):
        """DataFrameをCSVコンテンツに変換"""
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue().encode('utf-8')
    
    def test_load_csv_files(self):
        """CSVファイルの読み込みテスト"""
        # テスト用のファイルコンテンツを準備
        file_contents = []
        
        for boat_id, df in self.sample_data.items():
            csv_content = self._generate_csv_content(df)
            file_contents.append((f"{boat_id}.csv", csv_content, 'csv'))
        
        # ファイル読み込み
        result = self.processor.load_multiple_files(file_contents, auto_id=True)
        
        # 検証
        self.assertEqual(len(result), 3, "3つのファイルを読み込めていない")
        self.assertIn("Boat1", result, "Boat1のデータが見つからない")
        self.assertIn("Boat2", result, "Boat2のデータが見つからない")
        self.assertIn("Boat3", result, "Boat3のデータが見つからない")
        
        # データの内容確認
        for boat_id, df in result.items():
            self.assertIn('timestamp', df.columns, f"{boat_id}: timestampカラムがない")
            self.assertIn('latitude', df.columns, f"{boat_id}: latitudeカラムがない")
            self.assertIn('longitude', df.columns, f"{boat_id}: longitudeカラムがない")
            self.assertIn('speed', df.columns, f"{boat_id}: speedカラムがない")
    
    def test_manual_id_assignment(self):
        """手動ID割り当てのテスト"""
        # テスト用のファイルコンテンツを準備
        file_contents = []
        
        for boat_id, df in list(self.sample_data.items())[:2]:  # 2艇分のみ使用
            csv_content = self._generate_csv_content(df)
            file_contents.append((f"{boat_id}.csv", csv_content, 'csv'))
        
        # 手動ID
        manual_ids = ["TeamA", "TeamB"]
        
        # ファイル読み込み
        result = self.processor.load_multiple_files(file_contents, auto_id=False, manual_ids=manual_ids)
        
        # 検証
        self.assertEqual(len(result), 2, "2つのファイルを読み込めていない")
        self.assertIn("TeamA", result, "TeamAのデータが見つからない")
        self.assertIn("TeamB", result, "TeamBのデータが見つからない")
    
    def test_time_synchronization(self):
        """時間同期機能のテスト"""
        # 艇データの設定
        self.processor.boat_data = self.sample_data
        
        # 同期（1秒間隔）
        result = self.processor.synchronize_time(target_freq='1s')
        
        # 検証
        self.assertEqual(len(result), 3, "3艇のデータが同期されていない")
        
        # 全ての艇で同じタイムスタンプになっているか
        timestamps = {}
        for boat_id, df in result.items():
            timestamps[boat_id] = set(df['timestamp'].astype(str).tolist())
        
        # 全艇で共通のタイムスタンプセットを持つ
        common_timestamps = set.intersection(*map(set, timestamps.values()))
        self.assertGreater(len(common_timestamps), 0, "共通のタイムスタンプがない")
        
        # 各艇のタイムスタンプは共通セットと一致
        for boat_id, ts_set in timestamps.items():
            self.assertEqual(ts_set, common_timestamps, f"{boat_id}のタイムスタンプが他と異なる")
    
    def test_anomaly_detection(self):
        """異常値検出と修正のテスト"""
        # サンプルデータに異常値を追加
        df = self.sample_data["Boat1"].copy()
        
        # 異常な速度を設定（20m/sを超える）
        df.loc[10, 'speed'] = 25.0  # 約48.6ノット
        
        # 処理用データセット
        self.processor.boat_data = {"TestBoat": df}
        
        # 異常値検出・修正
        result = self.processor.detect_and_fix_gps_anomalies("TestBoat", max_speed_knots=30.0)
        
        # 検証
        self.assertLess(result.loc[10, 'speed'], 25.0, "異常な速度が修正されていない")
    
    def test_data_quality_report(self):
        """データ品質レポートのテスト"""
        # 艇データの設定
        self.processor.boat_data = self.sample_data
        
        # 品質レポート生成
        report = self.processor.get_data_quality_report()
        
        # 検証
        self.assertEqual(len(report), 3, "3艇分のレポートがない")
        
        for boat_id, boat_report in report.items():
            self.assertIn('quality_score', boat_report, f"{boat_id}の品質スコアがない")
            self.assertIn('quality_rating', boat_report, f"{boat_id}の品質評価がない")
            self.assertIn('total_points', boat_report, f"{boat_id}のポイント数がない")
    
    def test_export_data(self):
        """データエクスポート機能のテスト"""
        # 艇データの設定
        self.processor.synced_data = self.sample_data
        
        # CSVエクスポート
        csv_data = self.processor.export_processed_data("Boat1", format_type='csv')
        
        # 検証
        self.assertIsNotNone(csv_data, "CSVデータがエクスポートされていない")
        self.assertGreater(len(csv_data), 0, "CSVデータが空")
        
        # JSONエクスポート
        json_data = self.processor.export_processed_data("Boat2", format_type='json')
        
        # 検証
        self.assertIsNotNone(json_data, "JSONデータがエクスポートされていない")
        self.assertGreater(len(json_data), 0, "JSONデータが空")
    
    def test_common_timeframe(self):
        """共通時間枠検出のテスト"""
        # 各艇の時間範囲をずらす
        boat1_df = self.sample_data["Boat1"].copy()
        boat2_df = self.sample_data["Boat2"].copy()
        boat3_df = self.sample_data["Boat3"].copy()
        
        # Boat2は少し遅く始まる
        boat2_df['timestamp'] = boat2_df['timestamp'] + timedelta(minutes=2)
        
        # Boat3は少し早く終わる
        boat3_df = boat3_df.iloc[:-20]
        
        # 処理用データセット
        self.processor.boat_data = {
            "Boat1": boat1_df,
            "Boat2": boat2_df,
            "Boat3": boat3_df
        }
        
        # 共通時間枠の検出
        start_time, end_time = self.processor.get_common_timeframe()
        
        # 検証
        self.assertIsNotNone(start_time, "開始時刻がNone")
        self.assertIsNotNone(end_time, "終了時刻がNone")
        
        # 開始時刻はBoat2の開始時刻と一致
        self.assertEqual(start_time, boat2_df['timestamp'].min(), "共通開始時刻が不正")
        
        # 終了時刻はBoat3の終了時刻と一致
        self.assertEqual(end_time, boat3_df['timestamp'].max(), "共通終了時刻が不正")

if __name__ == '__main__':
    unittest.main()

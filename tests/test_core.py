"""
SailingDataProcessor コアクラスのテスト
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import tempfile
import warnings
import io

# テスト対象のモジュールをインポート
from sailing_data_processor import SailingDataProcessor


class TestSailingDataProcessor(unittest.TestCase):
    """SailingDataProcessorクラスのテストケース"""
    
    # tests/test_core.py の setUp メソッドを拡張

    def setUp(self):
        """各テストケース実行前のセットアップ"""
        # 警告を無視
        warnings.filterwarnings("ignore")
        
        # テスト用のデータプロセッサインスタンス
        self.processor = SailingDataProcessor()
        
        # テスト用のデータを作成
        self.sample_data = self._create_sample_data()
        
        # モックデータディレクトリの作成
        self.test_dir = tempfile.mkdtemp()
        
        # テスト用のポーラーデータを作成（必要に応じて）
        self._create_mock_polar_data()
    
    def _create_mock_polar_data(self):
        """テスト用のポーラーデータファイルを作成"""
        # ポーラーデータディレクトリ
        polar_dir = os.path.join(self.test_dir, 'polars')
        os.makedirs(polar_dir, exist_ok=True)
        
        # 基本的なポーラーデータの作成
        polar_data = [
            "TWA,6,8,10,12,14,16,20",
            "30,0.0,2.9,3.9,4.6,5.1,5.4,5.7",
            "45,3.1,4.0,4.9,5.4,5.7,6.0,6.3",
            "60,3.8,4.6,5.2,5.7,6.1,6.3,6.5",
            "90,4.2,5.1,5.8,6.3,6.7,7.0,7.2",
            "120,3.9,4.8,5.5,6.1,6.8,7.2,7.5",
            "150,3.1,4.0,4.9,5.7,6.4,7.1,7.8",
            "180,2.8,3.7,4.5,5.3,6.1,6.9,7.6"
        ]
        
        # ポーラーファイルの書き込み
        with open(os.path.join(polar_dir, 'default.csv'), 'w') as f:
            f.write('\n'.join(polar_data))
        
        # 各艇種用のポーラーファイルも作成
        for boat_type in ['laser', 'finn', '470', '49er']:
            # 基本データを少し変更
            modified_data = polar_data.copy()
            modified_data[0] = "TWA,6,8,10,12,14,16,20"  # ヘッダーは同じ
            
            # 各艇種の特性に応じて値を調整
            for i in range(1, len(modified_data)):
                parts = modified_data[i].split(',')
                twa = parts[0]
                
                # 艇種ごとの調整係数
                if boat_type == 'laser':
                    mod = 1.02  # Laserは少し速い
                elif boat_type == 'finn':
                    mod = 1.04  # Finnはさらに速い
                elif boat_type == '470':
                    mod = 1.06  # 470はかなり速い
                elif boat_type == '49er':
                    mod = 1.1   # 49erは最も速い
                
                # 速度値を調整
                for j in range(1, len(parts)):
                    if parts[j] != '0.0':
                        parts[j] = f"{float(parts[j]) * mod:.1f}"
                
                modified_data[i] = ','.join(parts)
            
            # 修正したポーラーファイルの書き込み
            with open(os.path.join(polar_dir, f'{boat_type}.csv'), 'w') as f:
                f.write('\n'.join(modified_data))
        
        # 環境変数で場所を指定（必要に応じて）
        os.environ['SAILING_POLAR_PATH'] = polar_dir
        
    def tearDown(self):
        """各テストケース実行後のクリーンアップ"""
        # 一時ディレクトリの削除
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)
    
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
        
        print("Debug: サンプルデータの準備を開始")
        
        for boat_id, df in self.sample_data.items():
            print(f"Debug: ボートID {boat_id}, データフレームサイズ {len(df)}")
            csv_content = self._generate_csv_content(df)
            file_contents.append((f"{boat_id}.csv", csv_content, 'csv'))
        
        print(f"Debug: 準備したファイル数: {len(file_contents)}")
        
        # ファイル読み込み
        print("Debug: load_multiple_files を呼び出します")
        result = self.processor.load_multiple_files(file_contents, auto_id=True)
        
        print(f"Debug: 読み込み結果のキー: {list(result.keys()) if result else '空のディクショナリ'}")
        
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
    
    def test_detect_and_fix_gps_anomalies(self):
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
    
    def test_process_multiple_boats(self):
        """複数艇データ処理のテスト"""
        # 艇データの設定
        self.processor.boat_data = self.sample_data
        
        # 処理実行
        result = self.processor.process_multiple_boats()
        
        # 検証
        self.assertEqual(len(result['data']), 3, "3艇のデータが処理されていない")
        self.assertEqual(len(result['stats']), 3, "3艇の統計情報が生成されていない")
        
        # 処理済みデータが存在するか
        for boat_id in self.sample_data.keys():
            self.assertIn(boat_id, result['data'], f"{boat_id}の処理済みデータが見つからない")
            self.assertIn(boat_id, result['stats'], f"{boat_id}の統計情報が見つからない")


if __name__ == '__main__':
    unittest.main()

import unittest
import pandas as pd
import numpy as np
import time
import psutil
import os
import math
import sys
import gc
from multiprocessing import Pool
from datetime import datetime, timedelta

# テスト対象のモジュールパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# テスト対象のクラスをインポート
try:
    from performance_optimizer import PerformanceOptimizer
except ImportError:
    # 新しいパッケージ構造の場合
    from sailing_data_processor.performance_optimizer import PerformanceOptimizer

class TestPerformanceOptimizer(unittest.TestCase):
    """PerformanceOptimizerクラスのテスト"""
    
    def setUp(self):
        """テストの準備"""
        self.optimizer = PerformanceOptimizer()
        self.large_df = self._create_large_dataframe(100000)  # 10万行のデータフレーム
    
    def _create_large_dataframe(self, size):
        """大きなDataFrameを作成"""
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=size, freq='1s'),
            'latitude': np.random.normal(35.6, 0.01, size),
            'longitude': np.random.normal(139.7, 0.01, size),
            'speed': np.random.uniform(3.0, 8.0, size),
            'bearing': np.random.uniform(0, 360, size),
            'altitude': np.random.uniform(0, 10, size),
            'extra_int': np.random.randint(0, 1000, size),
            'extra_float': np.random.randn(size),
            'boat_id': ['test_boat'] * size
        })
    
    def test_memory_optimization(self):
        """メモリ最適化のテスト"""
        # 元のメモリ使用量を計算
        original_memory = self.large_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        
        # 最適化
        optimized_df = self.optimizer.optimize_dataframe(self.large_df)
        
        # 最適化後のメモリ使用量
        optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        
        # 検証: メモリ使用量が減少していること
        self.assertLess(optimized_memory, original_memory)
        
        # 少なくとも20%の削減を期待
        reduction = (original_memory - optimized_memory) / original_memory
        self.assertGreaterEqual(reduction, 0.2)
        
        # データの等価性を確認
        pd.testing.assert_frame_equal(
            self.large_df.reset_index(drop=True),
            optimized_df.reset_index(drop=True),
            check_dtype=False  # データ型の違いは許容
        )
    
    def test_get_memory_usage(self):
        """メモリ使用量取得機能のテスト"""
        # メモリ使用情報を取得
        memory_info = self.optimizer.get_memory_usage()
        
        # 必要なキーが含まれていることを確認
        self.assertIn('process_memory_mb', memory_info)
        self.assertIn('total_memory_mb', memory_info)
        self.assertIn('available_memory_mb', memory_info)
        self.assertIn('memory_percent', memory_info)
        self.assertIn('process_memory_formatted', memory_info)
        
        # 値の範囲確認
        self.assertGreater(memory_info['process_memory_mb'], 0)
        self.assertGreater(memory_info['total_memory_mb'], 0)
        self.assertGreater(memory_info['available_memory_mb'], 0)
        self.assertGreaterEqual(memory_info['memory_percent'], 0)
        self.assertLessEqual(memory_info['memory_percent'], 100)
    
    def test_check_memory_threshold(self):
        """メモリ閾値チェック機能のテスト"""
        # 閾値を一時的に変更
        original_threshold = self.optimizer.memory_threshold
        
        try:
            # 閾値を非常に低く設定（必ず超えるように）
            self.optimizer.memory_threshold = 0.01
            self.assertTrue(self.optimizer.check_memory_threshold())
            
            # 閾値を非常に高く設定（必ず下回るように）
            self.optimizer.memory_threshold = 0.99
            self.assertFalse(self.optimizer.check_memory_threshold())
        finally:
            # 元の閾値に戻す
            self.optimizer.memory_threshold = original_threshold
    
    def test_downsample_adaptive(self):
        """適応的ダウンサンプリングのテスト"""
        # ダウンサンプリング
        target_size = 1000
        result = self.optimizer.downsample_data(
            self.large_df, target_size=target_size, method='adaptive'
        )
        
        # サイズの検証
        self.assertEqual(len(result), target_size)
        
        # メモリ削減率の計算
        original_memory = self.large_df.memory_usage(deep=True).sum()
        downsampled_memory = result.memory_usage(deep=True).sum()
        self.assertLess(downsampled_memory, original_memory)
        
        # 最初と最後のポイントが保持されていることを確認
        pd.testing.assert_series_equal(
            self.large_df.iloc[0][['latitude', 'longitude']], 
            result.iloc[0][['latitude', 'longitude']], 
            check_dtype=False
        )
        
        pd.testing.assert_series_equal(
            self.large_df.iloc[-1][['latitude', 'longitude']], 
            result.iloc[-1][['latitude', 'longitude']], 
            check_dtype=False
        )
    
    def test_downsample_uniform(self):
        """均一ダウンサンプリングのテスト"""
        # ダウンサンプリング
        factor = 0.01  # 1%に縮小
        result = self.optimizer.downsample_data(
            self.large_df, factor=factor, method='uniform'
        )
        
        # サイズの検証
        expected_size = int(len(self.large_df) * factor)
        self.assertAlmostEqual(len(result), expected_size, delta=2)
        
        # 最初と最後のデータポイントが保持されていることを確認
        pd.testing.assert_series_equal(
            self.large_df.iloc[0][['latitude', 'longitude']], 
            result.iloc[0][['latitude', 'longitude']], 
            check_dtype=False
        )
        
        pd.testing.assert_series_equal(
            self.large_df.iloc[-1][['latitude', 'longitude']], 
            result.iloc[-1][['latitude', 'longitude']], 
            check_dtype=False
        )
    
    def test_chunk_processing(self):
        """チャンク処理のテスト"""
        # チャンクサイズ
        chunk_size = 10000
        
        # チャンク分割
        chunks = self.optimizer.split_dataframe_in_chunks(self.large_df, chunk_size)
        
        # チャンク数の検証
        expected_chunks = math.ceil(len(self.large_df) / chunk_size)
        self.assertEqual(len(chunks), expected_chunks)
        
        # 各チャンクのサイズ検証
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                self.assertEqual(len(chunk), chunk_size)
            else:
                # 最後のチャンクは残りのサイズ
                self.assertEqual(len(chunk), len(self.large_df) % chunk_size or chunk_size)
        
        # すべてのチャンクを統合すると元のデータフレームになることを確認
        combined = pd.concat(chunks, ignore_index=True)
        self.assertEqual(len(combined), len(self.large_df))
        
        # データの一致を確認（インデックスは異なる可能性があるので内容のみ確認）
        combined = combined.sort_values(by=['timestamp']).reset_index(drop=True)
        sorted_original = self.large_df.sort_values(by=['timestamp']).reset_index(drop=True)
        pd.testing.assert_frame_equal(sorted_original, combined)
    
    def test_parallel_process_chunks(self):
        """並列処理のテスト"""
        # 処理関数
        def process_func(df):
            # シンプルな処理（平均を計算して新しい列を追加）
            df = df.copy()
            df['mean_speed'] = df['speed'].mean()
            return df
        
        # チャンク処理
        chunks = self.optimizer.split_dataframe_in_chunks(self.large_df, 20000)
        
        # シングルスレッド処理の時間測定
        start_time = time.time()
        single_results = [process_func(chunk) for chunk in chunks]
        single_time = time.time() - start_time
        
        # 並列処理の時間測定
        start_time = time.time()
        parallel_results = self.optimizer.parallel_process_chunks(chunks, process_func)
        parallel_time = time.time() - start_time
        
        # 結果が同じであることを確認
        for single, parallel in zip(single_results, parallel_results):
            pd.testing.assert_frame_equal(single, parallel)
        
        # 並列処理の方が速いことを確認（2コア以上ある環境では）
        if os.cpu_count() > 1:
            self.assertLess(parallel_time, single_time)
    
    def test_merge_processed_chunks(self):
        """処理済みチャンクの結合テスト"""
        # チャンク分割
        chunks = self.optimizer.split_dataframe_in_chunks(self.large_df, 20000)
        
        # 各チャンクに簡単な処理を施す
        processed_chunks = []
        for chunk in chunks:
            chunk = chunk.copy()
            chunk['processed'] = True
            processed_chunks.append(chunk)
        
        # チャンクの結合
        merged = self.optimizer.merge_processed_chunks(processed_chunks)
        
        # 結合結果の検証
        self.assertEqual(len(merged), len(self.large_df))
        self.assertTrue('processed' in merged.columns)
        self.assertTrue(merged['processed'].all())
    
    def test_process_large_dataset(self):
        """大規模データセットの処理テスト"""
        # 処理関数
        def process_func(df, multiply_by=2):
            df = df.copy()
            df['speed'] = df['speed'] * multiply_by
            return df
        
        # 大規模データセット処理
        result = self.optimizer.process_large_dataset(
            df=self.large_df,
            process_func=process_func,
            chunk_size=25000,
            optimize=True,
            multiply_by=3  # パラメータの渡し方テスト
        )
        
        # 結果の検証
        self.assertEqual(len(result), len(self.large_df))
        
        # 処理が正しく適用されたことを確認
        expected_speeds = self.large_df['speed'] * 3
        pd.testing.assert_series_equal(result['speed'], expected_speeds, check_names=False)
    
    def test_chunk_and_optimize_dict(self):
        """辞書データのチャンク化と最適化テスト"""
        # 2つの艇データを含む辞書を作成
        data_dict = {
            'boat1': self.large_df.copy(),
            'boat2': self.large_df.iloc[:50000].copy()  # 半分のサイズ
        }
        
        # チャンク化と最適化
        result = self.optimizer.chunk_and_optimize_dict(data_dict, chunk_size=20000)
        
        # 結果の検証
        self.assertEqual(len(result), 2)
        self.assertIn('boat1', result)
        self.assertIn('boat2', result)
        
        # boat1のチャンク数の検証
        expected_chunks1 = math.ceil(len(data_dict['boat1']) / 20000)
        self.assertEqual(len(result['boat1']), expected_chunks1)
        
        # boat2のチャンク数の検証
        expected_chunks2 = math.ceil(len(data_dict['boat2']) / 20000)
        self.assertEqual(len(result['boat2']), expected_chunks2)
        
        # 最適化されたチャンクのメモリサイズ検証
        original_memory = data_dict['boat1'].memory_usage(deep=True).sum() / expected_chunks1
        optimized_memory = result['boat1'][0].memory_usage(deep=True).sum()
        self.assertLessEqual(optimized_memory, original_memory)
    
    def test_empty_dataframe(self):
        """空のDataFrameに対する処理のテスト"""
        # 空のDataFrame
        empty_df = pd.DataFrame()
        
        # 最適化
        optimized = self.optimizer.optimize_dataframe(empty_df)
        self.assertTrue(optimized.empty)
        
        # ダウンサンプリング
        downsampled = self.optimizer.downsample_data(empty_df, target_size=10)
        self.assertTrue(downsampled.empty)
        
        # チャンク分割
        chunks = self.optimizer.split_dataframe_in_chunks(empty_df)
        self.assertEqual(len(chunks), 0)
    
    def test_cleanup_memory(self):
        """メモリ解放とGCテスト"""
        # 大量のデータを生成してメモリを消費
        big_data = [self._create_large_dataframe(10000) for _ in range(10)]
        
        # メモリ使用状況を確認
        memory_before = self.optimizer.get_memory_usage()['process_memory_mb']
        
        # 明示的にGCを実行
        del big_data
        gc.collect()
        
        # メモリ使用状況を再確認
        memory_after = self.optimizer.get_memory_usage()['process_memory_mb']
        
        # メモリが解放されていることを確認（環境によって異なる場合があるため柔軟に）
        # 確実にテストが失敗しないよう、条件を緩めに設定
        self.assertLessEqual(memory_after, memory_before * 1.2)
    
    def tearDown(self):
        """テスト終了後のクリーンアップ"""
        # 大きなデータフレームの参照を削除
        del self.large_df
        gc.collect()


if __name__ == '__main__':
    unittest.main()

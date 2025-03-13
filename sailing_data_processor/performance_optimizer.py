import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import gc
import psutil
import os
import time
import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class PerformanceOptimizer:
    """
    セーリングデータの処理パフォーマンスを最適化するためのクラス
    メモリ使用量の削減、大規模データセットの効率的な処理、
    ダウンサンプリングなどのユーティリティを提供します
    """
    
    def __init__(self):
        """初期化"""
        self.process = psutil.Process(os.getpid())
        self.memory_threshold = 0.8  # メモリ使用率の警告閾値 (80%)
        self.default_chunk_size = 10000  # デフォルトのチャンクサイズ
        self.max_workers = max(1, psutil.cpu_count(logical=True) - 1)  # 使用するCPUコア数（1つは常にメイン処理用に確保）
    
    def get_memory_usage(self) -> Dict[str, Union[float, str]]:
        """
        現在のメモリ使用状況を取得
        
        Returns:
        --------
        Dict[str, Union[float, str]]
            メモリ使用状況の情報
        """
        # 現在のプロセスのメモリ使用量
        process_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        
        # システム全体のメモリ情報
        system_memory = psutil.virtual_memory()
        total_memory = system_memory.total / (1024 * 1024)  # MB
        available_memory = system_memory.available / (1024 * 1024)  # MB
        used_memory = system_memory.used / (1024 * 1024)  # MB
        memory_percent = system_memory.percent
        
        return {
            'process_memory_mb': process_memory,
            'total_memory_mb': total_memory,
            'available_memory_mb': available_memory,
            'used_memory_mb': used_memory,
            'memory_percent': memory_percent,
            'process_memory_formatted': f"{process_memory:.1f} MB",
            'total_memory_formatted': f"{total_memory:.1f} MB",
            'available_memory_formatted': f"{available_memory:.1f} MB"
        }
    
    def check_memory_threshold(self) -> bool:
        """
        メモリ使用率が閾値を超えているか確認
        
        Returns:
        --------
        bool
            閾値を超えている場合True
        """
        memory_info = self.get_memory_usage()
        return memory_info['memory_percent'] > self.memory_threshold * 100
    
    def optimize_dataframe(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        DataFrameのメモリ使用量を最適化
        
        Parameters:
        -----------
        df : pd.DataFrame
            最適化するDataFrame
        verbose : bool
            詳細情報を出力するかどうか
            
        Returns:
        --------
        pd.DataFrame
            メモリ使用量が最適化されたDataFrame
        """
        if df is None or df.empty:
            return df
            
        start_mem = df.memory_usage().sum() / 1024**2
        if verbose:
            print(f"最適化前のメモリ使用量: {start_mem:.2f} MB")
        
        # コピーの作成
        df_optimized = df.copy()
        
        # 数値型の最適化
        for col in df_optimized.select_dtypes(include=['int']).columns:
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            # 符号なし整数型の使用
            if c_min >= 0:
                if c_max < 255:
                    df_optimized[col] = df_optimized[col].astype(np.uint8)
                elif c_max < 65535:
                    df_optimized[col] = df_optimized[col].astype(np.uint16)
                elif c_max < 4294967295:
                    df_optimized[col] = df_optimized[col].astype(np.uint32)
                else:
                    df_optimized[col] = df_optimized[col].astype(np.uint64)
            else:
                if c_min > -128 and c_max < 127:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif c_min > -32768 and c_max < 32767:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif c_min > -2147483648 and c_max < 2147483647:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
                else:
                    df_optimized[col] = df_optimized[col].astype(np.int64)
        
        # 浮動小数点型の最適化
        for col in df_optimized.select_dtypes(include=['float']).columns:
            # 値の範囲を調べて適切なデータ型を選択
            df_optimized[col] = df_optimized[col].astype(np.float32)
        
        # カテゴリ型の最適化
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if df_optimized[col].nunique() < len(df_optimized) * 0.5:  # ユニーク値が多すぎない場合
                df_optimized[col] = df_optimized[col].astype('category')
        
        end_mem = df_optimized.memory_usage().sum() / 1024**2
        reduction = 100 * (start_mem - end_mem) / start_mem
        
        if verbose:
            print(f"最適化後のメモリ使用量: {end_mem:.2f} MB")
            print(f"削減率: {reduction:.1f}%")
        
        return df_optimized
    
    def downsample_data(self, df: pd.DataFrame, target_size: int = None, 
                     factor: float = None, method: str = 'uniform') -> pd.DataFrame:
        """
        データをダウンサンプリング
        
        Parameters:
        -----------
        df : pd.DataFrame
            ダウンサンプリングするDataFrame
        target_size : int, optional
            目標サイズ（行数）
        factor : float, optional
            削減率（0-1の間、例: 0.5は元のサイズの半分に削減）
        method : str
            サンプリング方法（'uniform'または'adaptive'）
            'uniform': 均等間隔でサンプリング
            'adaptive': データ変化率に応じた適応的サンプリング
            
        Returns:
        --------
        pd.DataFrame
            ダウンサンプリングされたDataFrame
        """
        if df is None or df.empty:
            return df
            
        if target_size is None and factor is None:
            # デフォルトは半分のサイズ
            factor = 0.5
            
        if factor is not None:
            # 係数に基づいてターゲットサイズを計算
            target_size = max(2, int(len(df) * factor))
        
        # 目標サイズが現在のサイズより大きい場合は変更なし
        if target_size >= len(df):
            return df
        
        if method == 'uniform':
            # 均等間隔でサンプリング
            # 最初と最後の行は必ず含める
            if target_size <= 2:
                return df.iloc[[0, -1]]
                
            indices = np.linspace(0, len(df) - 1, target_size).astype(int)
            return df.iloc[indices]
            
        elif method == 'adaptive':
            # データの変化率に基づいた適応的サンプリング
            # 基本指標として速度と方位の変化を使用
            
            # 変化率の計算
            change_rates = []
            
            for col in ['speed', 'bearing']:
                if col in df.columns:
                    # 差分の絶対値を計算
                    # 方位は角度の循環性を考慮
                    if col == 'bearing':
                        # 方位の差分計算（循環性を考慮）
                        diffs = np.abs(
                            ((df[col].diff() + 180) % 360) - 180
                        )
                    else:
                        # 通常の差分
                        diffs = np.abs(df[col].diff())
                    
                    # NaNを0に置換
                    diffs = diffs.fillna(0)
                    
                    # 変化率を追加
                    change_rates.append(diffs)
            
            if not change_rates:
                # 変化率が計算できない場合は均等サンプリングにフォールバック
                return self.downsample_data(df, target_size=target_size, method='uniform')
            
            # 複数の変化率がある場合は合計
            if len(change_rates) > 1:
                total_change = sum(change_rates)
            else:
                total_change = change_rates[0]
            
            # 累積変化量を計算
            cumulative_change = total_change.cumsum()
            
            # 総変化量を計算
            total_change_sum = cumulative_change.iloc[-1]
            
            if total_change_sum <= 0:
                # 変化がない場合は均等サンプリングにフォールバック
                return self.downsample_data(df, target_size=target_size, method='uniform')
            
            # 目標のサンプリングポイントを計算
            # 常に最初と最後の点を含める
            target_points = [0]
            
            # 変化量に基づいて残りのポイントを選択
            change_step = total_change_sum / (target_size - 2)
            current_sum = 0
            
            for i in range(1, len(df) - 1):
                current_sum += total_change[i]
                if current_sum >= change_step:
                    target_points.append(i)
                    current_sum = 0
            
            # 最後の点を追加
            target_points.append(len(df) - 1)
            
            # 結果が目標サイズより小さい場合、均等サンプリングで補完
            if len(target_points) < target_size:
                remaining = target_size - len(target_points)
                # 既に選択した点を除外
                remaining_indices = [i for i in range(len(df)) if i not in target_points]
                
                if remaining_indices:
                    # 残りのインデックスから均等にサンプリング
                    additional_indices = np.linspace(0, len(remaining_indices) - 1, remaining).astype(int)
                    additional_points = [remaining_indices[i] for i in additional_indices]
                    target_points.extend(additional_points)
                    target_points.sort()
            
            # 目標サイズより大きい場合、一部を間引く
            if len(target_points) > target_size:
                # 最初と最後の点は保持
                middle_points = target_points[1:-1]
                # 中間点から均等にサンプリング
                selected_middle = np.linspace(0, len(middle_points) - 1, target_size - 2).astype(int)
                selected_indices = [0] + [middle_points[i] for i in selected_middle] + [target_points[-1]]
                target_points = selected_indices
            
            return df.iloc[target_points]
            
        else:
            raise ValueError(f"未対応のサンプリング方法: {method}")
    
    def split_dataframe_in_chunks(self, df: pd.DataFrame, chunk_size: int = None) -> List[pd.DataFrame]:
        """
        DataFrameを複数のチャンクに分割
        
        Parameters:
        -----------
        df : pd.DataFrame
            分割するDataFrame
        chunk_size : int, optional
            チャンクサイズ（指定がなければメモリ状況に応じて自動決定）
            
        Returns:
        --------
        List[pd.DataFrame]
            分割されたDataFrameのリスト
        """
        if df is None or df.empty:
            return []
            
        # メモリ状況に基づいてチャンクサイズを決定
        if chunk_size is None:
            mem_info = self.get_memory_usage()
            available_mb = mem_info['available_memory_mb']
            
            # 1行あたりのメモリ使用量を推定（バイト単位）
            row_size = df.memory_usage(index=True, deep=True).sum() / len(df)
            
            # 利用可能メモリの50%を使用するチャンクサイズを計算
            chunk_size = max(1000, int((available_mb * 0.5 * 1024 * 1024) / row_size))
            
            # デフォルトのチャンクサイズを上限とする
            chunk_size = min(chunk_size, self.default_chunk_size)
        
        # チャンク数の計算
        n_chunks = math.ceil(len(df) / chunk_size)
        
        # デ​​ータフレームを分割
        return [df.iloc[i*chunk_size:(i+1)*chunk_size].copy() for i in range(n_chunks)]
    
    def parallel_process_chunks(self, chunks: List[pd.DataFrame], process_func: callable, 
                             **kwargs) -> List[pd.DataFrame]:
        """
        データチャンクを並列処理
        
        Parameters:
        -----------
        chunks : List[pd.DataFrame]
            処理するDataFrameのチャンクリスト
        process_func : callable
            各チャンクに適用する処理関数 (DataFrame -> DataFrame)
        **kwargs
            処理関数に渡す追加引数
            
        Returns:
        --------
        List[pd.DataFrame]
            処理されたチャンクのリスト
        """
        if not chunks:
            return []
            
        # 並列処理用の部分関数を作成
        func = partial(process_func, **kwargs)
        
        # スレッドプールに変更
        # 使用するワーカー数はチャンク数と最大ワーカー数の小さい方
        n_workers = min(len(chunks), self.max_workers)
        
        results = []
        if n_workers > 1:
            # 並列処理の実行
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(func, chunk) for chunk in chunks]
                results = [future.result() for future in futures]
        else:
            # 並列処理しない場合
            results = [func(chunk) for chunk in chunks]
        
        return results
    
    def merge_processed_chunks(self, processed_chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """
        処理されたチャンクを結合
        
        Parameters:
        -----------
        processed_chunks : List[pd.DataFrame]
            処理されたDataFrameのチャンクリスト
            
        Returns:
        --------
        pd.DataFrame
            結合されたDataFrame
        """
        if not processed_chunks:
            return pd.DataFrame()
            
        # チャンクを結合
        result = pd.concat(processed_chunks, ignore_index=True)
        
        # メモリを解放
        del processed_chunks
        gc.collect()
        
        return result
    
    def process_large_dataset(self, df: pd.DataFrame, process_func: callable, 
                           chunk_size: int = None, optimize: bool = True, 
                           **kwargs) -> pd.DataFrame:
        """
        大規模データセットを分割・並列処理・結合
        
        Parameters:
        -----------
        df : pd.DataFrame
            処理するDataFrame
        process_func : callable
            適用する処理関数
        chunk_size : int, optional
            チャンクサイズ
        optimize : bool
            結果を最適化するかどうか
        **kwargs
            処理関数に渡す追加引数
            
        Returns:
        --------
        pd.DataFrame
            処理結果
        """
        if df is None or df.empty:
            return df
            
        # データフレームを分割
        chunks = self.split_dataframe_in_chunks(df, chunk_size)
        
        # 分割したチャンクを並列処理
        processed_chunks = self.parallel_process_chunks(chunks, process_func, **kwargs)
        
        # 処理したチャンクを結合
        result = self.merge_processed_chunks(processed_chunks)
        
        # 結果の最適化
        if optimize and not result.empty:
            result = self.optimize_dataframe(result)
        
        return result
    
    def chunk_and_optimize_dict(self, data_dict: Dict[str, pd.DataFrame], 
                             chunk_size: int = None) -> Dict[str, List[pd.DataFrame]]:
        """
        複数艇のデータを含む辞書を最適化＆チャンク分割
        
        Parameters:
        -----------
        data_dict : Dict[str, pd.DataFrame]
            艇ID:DataFrameの辞書
        chunk_size : int, optional
            チャンクサイズ
            
        Returns:
        --------
        Dict[str, List[pd.DataFrame]]
            艇ID:最適化・分割されたチャンクリストの辞書
        """
        if not data_dict:
            return {}
            
        result = {}
        
        for boat_id, df in data_dict.items():
            # まずデータフレームを最適化
            optimized_df = self.optimize_dataframe(df)
            
            # 次にチャンクに分割
            chunks = self.split_dataframe_in_chunks(optimized_df, chunk_size)
            
            result[boat_id] = chunks
        
        return result

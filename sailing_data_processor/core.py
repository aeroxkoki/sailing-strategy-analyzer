# sailing_data_processor/core.py
"""
セーリング戦略分析システム - コアモジュール

SailingDataProcessorクラスを提供し、データ処理の中心的な役割を担う
"""

import pandas as pd
import numpy as np
import gpxpy
import io
import math
from geopy.distance import geodesic
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# 内部モジュールのインポート
from .wind_estimator import WindEstimator
from .performance_optimizer import PerformanceOptimizer
from .boat_data_fusion import BoatDataFusionModel
from .wind_field_interpolator import WindFieldInterpolator

class SailingDataProcessor:
    """セーリングデータ処理クラス - GPSデータの読み込み、前処理、分析を担当"""
    
    def __init__(self):
        """初期化"""
        self.boat_data = {}  # 艇ID: DataFrameの辞書
        self.processed_data = {}  # 処理済みデータ
        self.synced_data = {}  # 時間同期済みデータ
        self.max_boats = 100  # 最大処理可能艇数
        
        # 風推定器と最適化ユーティリティを初期化
        self._wind_estimator = WindEstimator()
        self._optimizer = PerformanceOptimizer()
        
        # 必要に応じて融合モデルと補間器を初期化（遅延初期化）
        self._fusion_model = None
        self._interpolator = None
        
        # 推定された風向風速データ
        self.wind_estimates = {}  # 艇ID: 風推定DataFrameの辞書
        self.wind_field_data = {}  # 時間: 風の場データの辞書
        
        # パフォーマンス統計情報
        self.performance_stats = {
            'load_time': 0,  # データ読み込み時間
            'process_time': 0,  # 処理時間
            'sync_time': 0,  # 同期時間
            'wind_estimation_time': 0,  # 風推定時間
            'total_points_processed': 0,  # 処理されたデータポイント総数
            'memory_usage': []  # メモリ使用量の履歴
        }
        
        # コンフィグ
        self.config = {
            'auto_optimize': True,  # 自動最適化の有効/無効
            'use_parallel': True,  # 並列処理の有効/無効
            'downsample_threshold': 1000000,  # ダウンサンプリングの閾値（ポイント数）
            'downsample_target': 0.5,  # ダウンサンプリング目標（元のサイズに対する比率）
            'chunking_threshold': 50000,  # チャンク分割の閾値（ポイント数）
            'log_performance': False,  # パフォーマンスログの有効/無効
            'auto_gc': True,  # 自動ガベージコレクションの有効/無効
            'default_tack_angle': 30.0,  # デフォルトのタック検出角度
            'wind_estimate_confidence_threshold': 0.6  # 風推定の信頼度閾値
        }
    
    # プロパティによる遅延初期化
    @property
    def wind_estimator(self):
        """風推定器へのアクセス"""
        return self._wind_estimator
    
    @property
    def optimizer(self):
        """最適化ユーティリティへのアクセス"""
        return self._optimizer
    
    @property
    def fusion_model(self):
        """データ融合モデルの遅延初期化"""
        if self._fusion_model is None:
            self._fusion_model = BoatDataFusionModel()
        return self._fusion_model
    
    @property
    def interpolator(self):
        """風の場補間器の遅延初期化"""
        if self._interpolator is None:
            self._interpolator = WindFieldInterpolator()
        return self._interpolator
    
    def update_config(self, **kwargs):
        """
        設定を更新
        
        Parameters:
        -----------
        **kwargs
            更新する設定のキーと値
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            else:
                warnings.warn(f"未知の設定キー: {key}")
    
    def _log_performance_step(self, step_name: str):
        """
        パフォーマンスステップをログに記録
        
        Parameters:
        -----------
        step_name : str
            ステップ名
        """
        if not self.config['log_performance']:
            return
            
        # メモリ使用量を記録
        memory_info = self.optimizer.get_memory_usage()
        memory_entry = {
            'step': step_name,
            'timestamp': datetime.now().isoformat(),
            'memory_mb': memory_info['process_memory_mb'],
            'system_memory_percent': memory_info['memory_percent']
        }
        self.performance_stats['memory_usage'].append(memory_entry)

    def _ensure_columns(self, df: pd.DataFrame, required_cols: List[str], 
                        optional_cols: List[str] = None, boat_id: str = "Unknown") -> pd.DataFrame:
        """
        DataFrameに必要なカラムが存在するか確認し、欠けているオプションカラムを追加します
        
        Parameters:
        -----------
        df : pd.DataFrame
            確認するDataFrame
        required_cols : List[str]
            必須カラムのリスト
        optional_cols : List[str], optional
            オプショナルカラムのリスト（存在しない場合は空または0で初期化）
        boat_id : str
            ログメッセージ用の艇ID
            
        Returns:
        --------
        pd.DataFrame
            必要に応じて追加カラムが付加されたDataFrame
        
        Raises:
        -------
        ValueError
            必須カラムが存在しない場合
        """
        if df is None:
            return None
            
        # 必須カラムのチェック
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            error_msg = f"{boat_id}: 必須カラムがありません: {missing_cols}"
            warnings.warn(error_msg)
            raise ValueError(error_msg)
        
        # オプショナルカラムの追加
        df_modified = df.copy()
        if optional_cols:
            for col in optional_cols:
                if col not in df.columns:
                    # カラムの型に応じた初期化
                    if col in ['speed', 'bearing', 'distance', 'time_diff', 'acceleration']:
                        df_modified[col] = 0.0
                    elif col in ['timestamp']:
                        # timestamp列が必須なら、この分岐は実行されないはず
                        pass
                    else:
                        df_modified[col] = None
        
        return df_modified
    
    def load_multiple_files(self, file_contents: List[Tuple[str, bytes, str]], 
                           auto_id: bool = True, manual_ids: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        複数のGPXまたはCSVファイルを一度に読み込む
        
        Parameters:
        -----------
        file_contents : List[Tuple[str, bytes, str]]
            (ファイル名, ファイル内容, ファイル形式)のリスト
        auto_id : bool
            True: ファイル名からIDを自動生成、False: manual_idsを使用
        manual_ids : List[str], optional
            手動で指定する艇ID（auto_id=Falseの場合に使用）
                
        Returns:
        --------
        Dict[str, pd.DataFrame]
            艇IDをキーとするDataFrameの辞書
        """
        if not auto_id and (manual_ids is None or len(manual_ids) != len(file_contents)):
            raise ValueError("手動ID指定モードの場合、ファイル数と同じ数のIDが必要です")
    
        # ここにデバッグ出力を追加
        print(f"Debug: load_multiple_files が呼び出されました。ファイル数: {len(file_contents)}")
        
        # 最大艇数をチェック
        if len(file_contents) > self.max_boats:
            warnings.warn(f"ファイル数が最大処理可能艇数({self.max_boats})を超えています。最初の{self.max_boats}ファイルのみ処理します。")
            file_contents = file_contents[:self.max_boats]
            if not auto_id and manual_ids is not None:
                manual_ids = manual_ids[:self.max_boats]
        
        # 開始時間を記録
        start_time = time.time()
        self._log_performance_step("load_multiple_start")
        
        # 並列処理を使用するかどうか
        # テスト時は並列処理を無効化
        use_parallel = False  # 並列処理の問題を排除するため一時的に無効化
        print(f"Debug: 並列処理モード: {use_parallel}")
        
        # 処理結果を格納する辞書を初期化
        results_dict = {}
        
        if use_parallel:
            # 並列処理用にタスクを準備
            tasks = []
            for idx, (filename, content, filetype) in enumerate(file_contents):
                boat_id = manual_ids[idx] if not auto_id and manual_ids else filename.split('.')[0]
                tasks.append((filename, content, filetype, boat_id, self.config['auto_optimize']))
            
            # マルチプロセッシングのためのワーカー数
            n_workers = min(len(tasks), self.optimizer.max_workers)
            
            # 並列処理の実行
            results = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # 部分関数を作成
                func = partial(self._parallel_load_file)
                # プールで実行
                futures = [executor.submit(func, task) for task in tasks]
                results = [future.result() for future in futures if future.result() is not None]
            
            # 結果を統合
            for result in results:
                if result is not None:
                    boat_id, df = result
                    if df is not None:
                        self.boat_data[boat_id] = df
                        results_dict[boat_id] = df  # 結果も格納
        else:
            # 逐次処理
            for idx, (filename, content, filetype) in enumerate(file_contents):
                # 重要な修正: boat_idをここで明示的に設定
                boat_id = manual_ids[idx] if not auto_id and manual_ids else filename.split('.')[0]
                print(f"Debug: ファイル {filename} 読み込み開始 (boat_id: {boat_id})")
                
                df = self._load_file(filename, content, filetype, boat_id, self.config['auto_optimize'])
                if df is not None:
                    print(f"Debug: ファイル {filename} の読み込み成功 (行数: {len(df)}, boat_id: {boat_id})")
                    # アルゴリズムの改善：タイムスタンプでソート
                    if 'timestamp' in df.columns:
                        df = df.sort_values('timestamp').reset_index(drop=True)
                    self.boat_data[boat_id] = df
                    results_dict[boat_id] = df  # 結果も格納
                else:
                    print(f"Debug: ファイル {filename} の読み込み失敗")
                         
        # パフォーマンス統計を更新
        elapsed = time.time() - start_time
        self.performance_stats['load_time'] += elapsed
        for boat_id, df in self.boat_data.items():
            self.performance_stats['total_points_processed'] += len(df)
        
        self._log_performance_step("load_multiple_end")
    
        # ここにデバッグ出力を追加
        print(f"Debug: 読み込み完了。艇データ数: {len(self.boat_data)}")
        if self.boat_data:
            print(f"Debug: 艇データキー: {list(self.boat_data.keys())}")
        
        # メモリ状況の確認と必要に応じてガベージコレクション
        if self.config['auto_gc'] and self.optimizer.check_memory_threshold():
            gc.collect()
        
        # 結果を直接返す（boat_dataではなくresults_dictを返す）
        print(f"Debug: 戻り値の艇データ数: {len(results_dict)}")
        if results_dict:
            print(f"Debug: 読み込み結果のキー: {list(results_dict.keys())}")
        return results_dict

    def _load_file(self, filename: str, content: bytes, filetype: str, 
         boat_id: str = None, auto_optimize: bool = None) -> Optional[pd.DataFrame]:
        """
        単一ファイルを読み込む（内部メソッド）
        """
        if auto_optimize is None:
            auto_optimize = self.config['auto_optimize']
        
        # 艇IDの決定
        if boat_id is None:
            boat_id = filename.split('.')[0]
        
        # 同じIDが存在する場合は連番を付加
        base_id = boat_id
        counter = 1
        while boat_id in self.boat_data:
            boat_id = f"{base_id}_{counter}"
            counter += 1
        
        # ファイルタイプに応じた読み込み
        df = None
        try:
            if filetype.lower() == 'gpx':
                df = self._load_gpx(content.decode('utf-8'), boat_id)
            elif filetype.lower() == 'csv':
                # content_sizeにバイトサイズを格納してロギング
                content_size = len(content) if content else 0
                print(f"Debug: CSVファイル読み込み開始 ({filename}, サイズ: {content_size}バイト)")
                df = self._load_csv(content, boat_id)
                if df is not None:
                    print(f"Debug: CSVファイル読み込み成功 ({len(df)}行, {list(df.columns)})")
                else:
                    print(f"Debug: CSVファイル読み込み失敗")
            else:
                warnings.warn(f"未対応のファイル形式です: {filetype}")
                return None
            
            if df is None:
                return None
            
            # データサイズの確認と自動ダウンサンプリング
            if auto_optimize and len(df) > self.config['downsample_threshold']:
                # 大きなデータセットを自動的にダウンサンプリング
                target_size = int(len(df) * self.config['downsample_target'])
                df = self.optimizer.downsample_data(df, target_size=target_size, method='adaptive')
                warnings.warn(f"{boat_id}: 大規模データセットを {len(df)} ポイントにダウンサンプリングしました")
            
            # メモリ最適化
            if auto_optimize:
                df = self.optimizer.optimize_dataframe(df)
            
            # 前処理（速度、方位などの計算）
            df = self._preprocess_gps_data(df)
            
            # 処理が成功したら明示的にログ出力
            if df is not None:
                print(f"Debug: ファイル {filename} の処理完了 - 行数: {len(df)}")
            
            return df
        except Exception as e:
            print(f"Debug: ファイル読み込みエラー ({filename}): {str(e)}")
            warnings.warn(f"ファイル読み込みエラー ({filename}): {str(e)}")
            return None
    
    def _parallel_load_file(self, args):
        """
        並列処理のためのファイル読み込み関数
        
        Parameters:
        -----------
        args : tuple
            (filename, content, filetype, boat_id, auto_optimize)のタプル
            
        Returns:
        --------
        tuple or None
            (boat_id, DataFrame)のタプル、失敗時はNone
        """
        filename, content, filetype, boat_id, auto_optimize = args
        
        try:
            df = self._load_file(filename, content, filetype, boat_id, auto_optimize)
            if df is not None:
                return (boat_id, df)
            return None
        except Exception as e:
            warnings.warn(f"並列ファイル読み込みエラー ({filename}): {e}")
            return None
    
    def _load_gpx(self, gpx_content: str, boat_id: str) -> Optional[pd.DataFrame]:
        """
        GPXデータを読み込み、DataFrameに変換
        
        Parameters:
        -----------
        gpx_content : str
            GPXファイルの内容
        boat_id : str
            艇ID
            
        Returns:
        --------
        Optional[pd.DataFrame]
            解析済みGPSデータ、読み込み失敗時はNone
        """
        try:
            # GPXデータを解析
            gpx = gpxpy.parse(gpx_content)
                
            # データポイントを格納するリスト
            points = []
            
            # GPXファイルからトラックポイントを抽出
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        points.append({
                            'timestamp': point.time,
                            'latitude': point.latitude,
                            'longitude': point.longitude,
                            'elevation': point.elevation if point.elevation is not None else 0,
                            'boat_id': boat_id
                        })
            
            # 十分なポイントがない場合
            if len(points) < 10:
                warnings.warn(f"{boat_id}: GPXファイルに十分なトラックポイントがありません")
                return None
            
            # DataFrameに変換
            df = pd.DataFrame(points)
            
            # タイムスタンプを日時型に変換
            if df['timestamp'].dtype != 'datetime64[ns]':
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            warnings.warn(f"{boat_id}: GPXファイルの読み込みエラー: {str(e)}")
            return None

    def _load_csv(self, csv_content: bytes, boat_id: str) -> Optional[pd.DataFrame]:
        """
        CSVデータを読み込み、DataFrameに変換
        """
        try:
            # デコードの際にエラーが発生する場合の対策
            try:
                decoded_content = csv_content.decode('utf-8')
            except UnicodeDecodeError:
                # UTF-8以外のエンコーディングを試す
                try:
                    decoded_content = csv_content.decode('shift-jis')
                except UnicodeDecodeError:
                    decoded_content = csv_content.decode('latin-1')  # 最終手段
            
            # CSVファイルを読み込み
            # StringIOではなく直接BytesIOを使うように修正
            df = pd.read_csv(io.BytesIO(csv_content))
            
            # boat_id列を追加
            df['boat_id'] = boat_id
            
            # 必要な列があるか確認
            required_cols = ['timestamp', 'latitude', 'longitude']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                warnings.warn(f"{boat_id}: CSVファイルに必要な列がありません: {missing_cols}")
                return None
            
            # タイムスタンプを日時型に変換
            if 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception as e:
                    warnings.warn(f"{boat_id}: タイムスタンプの変換に失敗しました: {e}")
                    # 失敗しても処理を続ける
            
            return df
                
        except Exception as e:
            warnings.warn(f"{boat_id}: CSVファイルの読み込みエラー: {str(e)}")
            return None

    def _preprocess_gps_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        GPSデータの前処理（速度、方位計算、ノイズ除去など）
        
        Parameters:
        -----------
        df : pd.DataFrame
            生のGPSデータ
            
        Returns:
        --------
        pd.DataFrame
            前処理済みデータ
        """
        if df is None or df.empty:
            return df
            
        # Categoricalデータ型をチェックし、必要に応じて変換する
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype(object)  # Categoricalを一般的なオブジェクト型に変換
                
        # 必須カラムをチェック
        required_cols = ['timestamp', 'latitude', 'longitude']
        optional_cols = ['speed', 'bearing', 'distance', 'time_diff']
        
        try:
            df = self._ensure_columns(df, required_cols, optional_cols)
        except ValueError:
            # 必須カラムがない場合は処理不可
            return None
        
        # タイムスタンプのソート
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 時間差分を計算（秒単位）
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        
        # 異常な時間差（負の値や極端に大きい値）を検出
        time_mask = (df['time_diff'] > 0) & (df['time_diff'] < 60)  # 1分以内を有効とする
        if not time_mask.all() and len(df) > 1:
            # 最初の行はNaNになるので除外
            invalid_times = df.index[~time_mask & (df.index > 0)].tolist()
            if len(invalid_times) > 0:
                # 異常値の処理（補間または除外）
                for idx in invalid_times:
                    if idx > 0 and idx < len(df) - 1:
                        # 前後の有効な値から線形補間
                        prev_idx = max([i for i in range(idx) if i not in invalid_times and i < idx])
                        next_idx = min([i for i in range(idx + 1, len(df)) if i not in invalid_times])
                        
                        # 時間の補間
                        total_time = (df.loc[next_idx, 'timestamp'] - df.loc[prev_idx, 'timestamp']).total_seconds()
                        points = next_idx - prev_idx
                        if points > 0:
                            df.loc[idx, 'time_diff'] = total_time / points
        
        # 距離計算（メートル単位）
        # 初期化を "0.0" にする前に、df['distance']がCategoricalでないか確認する
        if 'distance' in df.columns and pd.api.types.is_categorical_dtype(df['distance']):
            df['distance'] = df['distance'].astype(float)
        else:
            df['distance'] = 0.0
            
        for i in range(1, len(df)):
            df.loc[i, 'distance'] = geodesic(
                (df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']),
                (df.loc[i, 'latitude'], df.loc[i, 'longitude'])
            ).meters
        
        # 速度計算（メートル/秒）- 既に速度がある場合は上書きしない
        if 'speed' not in df.columns or df['speed'].isnull().all():
            # speedカラムがCategoricalでないことを確認
            if 'speed' in df.columns and pd.api.types.is_categorical_dtype(df['speed']):
                df['speed'] = df['speed'].astype(float)
            df['speed'] = np.where(df['time_diff'] > 0, df['distance'] / df['time_diff'], 0)
        
        # 異常値検出（速度ベースのアウトライアー検出）
        if len(df) > 10:  # 十分なデータポイントがある場合
            speed_mean = df['speed'].mean()
            speed_std = df['speed'].std()
            if speed_std > 0:  # 標準偏差が0でない場合のみ
                speed_threshold = speed_mean + 3 * speed_std  # 3シグマルール
                
                # 極端に速い速度を検出し、補正
                speed_outliers = df['speed'] > speed_threshold
                if speed_outliers.any():
                    # 異常値を検出した場合、中央値で置換または補間
                    df.loc[speed_outliers, 'speed'] = df['speed'].median()
                    # 距離も補正
                    for idx in df.index[speed_outliers]:
                        if idx > 0 and df.loc[idx, 'time_diff'] > 0:
                            df.loc[idx, 'distance'] = df.loc[idx, 'speed'] * df.loc[idx, 'time_diff']
        
        # 進行方向（ベアリング）の計算 - 既に方位がある場合は上書きしない
        if 'bearing' not in df.columns or df['bearing'].isnull().all():
            # bearingカラムがCategoricalでないことを確認
            if 'bearing' in df.columns and pd.api.types.is_categorical_dtype(df['bearing']):
                df['bearing'] = df['bearing'].astype(float)
            else:
                df['bearing'] = 0.0
                
            for i in range(1, len(df)):
                lat1, lon1 = math.radians(df.loc[i-1, 'latitude']), math.radians(df.loc[i-1, 'longitude'])
                lat2, lon2 = math.radians(df.loc[i, 'latitude']), math.radians(df.loc[i, 'longitude'])
                
                # ベアリング計算
                y = math.sin(lon2 - lon1) * math.cos(lat2)
                x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
                bearing = math.degrees(math.atan2(y, x))
                
                # 0-360度の範囲に正規化
                bearing = (bearing + 360) % 360
                
                df.loc[i, 'bearing'] = bearing
        
        # NaN値を処理
        df = df.fillna(0)
        
        return df
    
    # ==== 風向風速推定メソッド ====
    
    def estimate_wind_from_boat(self, boat_id: str, min_tack_angle: float = None, 
                              boat_type: str = 'default', use_bayesian: bool = True) -> Optional[pd.DataFrame]:
        """
        単一艇のGPSデータから風向風速を推定
        
        Parameters:
        -----------
        boat_id : str
            推定対象の艇ID
        min_tack_angle : float, optional
            タックと認識する最小の方向転換角度（指定がない場合は設定から取得）
        boat_type : str
            艇種識別子（速度係数に影響）
        use_bayesian : bool
            ベイズ推定を使用するかどうか
            
        Returns:
        --------
        pd.DataFrame or None
            推定された風向風速情報を含むDataFrame、推定失敗時はNone
        """
        if boat_id not in self.boat_data and boat_id not in self.processed_data:
            warnings.warn(f"艇 {boat_id} のデータが見つかりません")
            return None
        
        # 使用するデータの決定
        gps_data = self.processed_data.get(boat_id, self.boat_data.get(boat_id))
        
        if min_tack_angle is None:
            min_tack_angle = self.config['default_tack_angle']
        
        # 開始時間を記録
        start_time = time.time()
        self._log_performance_step(f"wind_estimation_start_{boat_id}")
        
        # WindEstimatorを使用して風向風速を推定
        wind_estimates = self.wind_estimator.estimate_wind_from_single_boat(
            gps_data=gps_data,
            min_tack_angle=min_tack_angle,
            boat_type=boat_type,
            use_bayesian=use_bayesian
        )
        
        # 推定結果を保存
        if wind_estimates is not None:
            self.wind_estimates[boat_id] = wind_estimates
        
        # パフォーマンス統計を更新
        elapsed = time.time() - start_time
        self.performance_stats['wind_estimation_time'] += elapsed
        
        self._log_performance_step(f"wind_estimation_end_{boat_id}")
        
        return wind_estimates
    
    def estimate_wind_from_all_boats(self, boat_types: Dict[str, str] = None, 
                                   boat_weights: Dict[str, float] = None) -> Dict[str, pd.DataFrame]:
        """
        全艇のGPSデータから風向風速を推定
        
        Parameters:
        -----------
        boat_types : Dict[str, str], optional
            艇ID:艇種の辞書
        boat_weights : Dict[str, float], optional
            艇ID:重み係数の辞書（技術レベルに基づく重み付け）
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            艇ID:風推定データの辞書
        """
        # 使用するデータの決定
        boats_data = self.processed_data if self.processed_data else self.boat_data
        
        if not boats_data:
            warnings.warn("処理対象の艇データがありません")
            return {}
        
        # デフォルト値の設定
        if boat_types is None:
            boat_types = {boat_id: 'default' for boat_id in boats_data.keys()}
            
        if boat_weights is None:
            boat_weights = {boat_id: 1.0 for boat_id in boats_data.keys()}
        
        # 開始時間を記録
        start_time = time.time()
        self._log_performance_step("wind_estimation_all_start")
        
        # 並列処理を使用するかどうか
        use_parallel = self.config['use_parallel'] and len(boats_data) > 1
        
        if use_parallel:
            # 並列処理用にタスクを準備
            tasks = []
            for boat_id, df in boats_data.items():
                boat_type = boat_types.get(boat_id, 'default')
                tasks.append((boat_id, df, self.config['default_tack_angle'], boat_type, True))
            
            # マルチプロセッシングのためのワーカー数
            n_workers = min(len(tasks), self.optimizer.max_workers)
            
            # 並列処理の実行
            results = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # 部分関数を作成
                func = partial(self._parallel_estimate_wind)
                # プールで実行
                futures = [executor.submit(func, task) for task in tasks]
                results = [future.result() for future in futures if future.result() is not None]
            
            # 結果を統合
            for result in results:
                if result is not None:
                    boat_id, wind_df = result
                    if wind_df is not None:
                        self.wind_estimates[boat_id] = wind_df
        else:
            # 逐次処理
            for boat_id, df in boats_data.items():
                boat_type = boat_types.get(boat_id, 'default')
                self.estimate_wind_from_boat(boat_id, self.config['default_tack_angle'], boat_type, True)
        
        # 複数艇のデータを融合して統合風推定を作成
        if len(self.wind_estimates) > 1:
            self._create_integrated_wind_estimate(boat_weights)
        
        # パフォーマンス統計を更新
        elapsed = time.time() - start_time
        self.performance_stats['wind_estimation_time'] += elapsed
        
        self._log_performance_step("wind_estimation_all_end")
        
        # メモリ状況の確認と必要に応じてガベージコレクション
        if self.config['auto_gc'] and self.optimizer.check_memory_threshold():
            gc.collect()
        
        return self.wind_estimates
    
    def _parallel_estimate_wind(self, args):
        """
        並列処理のための風推定関数
        
        Parameters:
        -----------
        args : tuple
            (boat_id, df, min_tack_angle, boat_type, use_bayesian)のタプル
            
        Returns:
        --------
        tuple or None
            (boat_id, wind_df)のタプル、失敗時はNone
        """
        boat_id, df, min_tack_angle, boat_type, use_bayesian = args
        
        try:
            # 新しいWindEstimatorインスタンスを作成（並列処理用）
            estimator = WindEstimator()
            
            # 風向風速を推定
            wind_df = estimator.estimate_wind_from_single_boat(
                gps_data=df,
                min_tack_angle=min_tack_angle,
                boat_type=boat_type,
                use_bayesian=use_bayesian
            )
            
            if wind_df is not None:
                return (boat_id, wind_df)
            return None
            
        except Exception as e:
            warnings.warn(f"並列風推定エラー ({boat_id}): {e}")
            return None
    
    def _create_integrated_wind_estimate(self, boat_weights: Dict[str, float] = None):
        """
        複数艇からの風推定を統合して、より信頼性の高い統合推定を作成
        
        Parameters:
        -----------
        boat_weights : Dict[str, float], optional
            艇ID:重み係数の辞書（技術レベルに基づく重み付け）
        """
        if len(self.wind_estimates) < 2:
            return  # 統合のための十分なデータがない
            
        # BoatDataFusionModelを使用して風向風速データを融合
        fused_estimates = self.fusion_model.fuse_wind_estimates(
            boats_estimates=self.wind_estimates,
            time_point=None  # 全時間点での融合
        )
        
        if fused_estimates:
            self.wind_estimates['integrated'] = fused_estimates
    
    def fuse_wind_estimates(self, time_point: datetime = None) -> Optional[Dict[str, Any]]:
        """
        複数艇からの風向風速推定を融合
        
        Parameters:
        -----------
        time_point : datetime, optional
            対象時間点（指定がない場合は最新の共通時間）
            
        Returns:
        --------
        Dict[str, Any] or None
            融合された風向風速データと信頼度
        """
        if not self.wind_estimates:
            warnings.warn("風向風速の推定データがありません")
            return None
        
        # BoatDataFusionModelを使用して風向風速データを融合
        return self.fusion_model.fuse_wind_estimates(
            boats_estimates=self.wind_estimates,
            time_point=time_point
        )
    
    def estimate_wind_field(self, time_point: datetime, 
                         grid_resolution: int = 20) -> Optional[Dict[str, Any]]:
        """
        特定時点での風の場を推定
        
        Parameters:
        -----------
        time_point : datetime
            風推定を行いたい時点
        grid_resolution : int
            出力グリッドの解像度
            
        Returns:
        --------
        Dict[str, Any] or None
            風の場データ
        """
        # 風向風速推定データがあるか確認
        if not self.wind_estimates:
            # データがなければ全艇から風推定を実行
            self.estimate_wind_from_all_boats()
            
        if not self.wind_estimates:
            warnings.warn("風向風速の推定データがありません")
            return None
        
        # WindEstimatorを使用して風の場を推定
        wind_field = self.wind_estimator.estimate_wind_field(time_point, grid_resolution)
        
        if wind_field is not None:
            # 風の場データをキャッシュ
            self.wind_field_data[time_point] = wind_field
            
            # WindFieldInterpolatorにも追加
            self.interpolator.add_wind_field(time_point, wind_field)
        
        return wind_field
    
    def interpolate_wind_field(self, target_time: datetime, 
                             resolution: int = None, 
                             method: str = None) -> Optional[Dict[str, Any]]:
        """
        指定時間の風の場を補間
        
        Parameters:
        -----------
        target_time : datetime
            対象時間
        resolution : int, optional
            出力解像度（指定がない場合は元データと同解像度）
        method : str, optional
            補間方法（'gp', 'rbf', 'idw'）、指定がない場合はデフォルト設定を使用
            
        Returns:
        --------
        Dict[str, Any] or None
            補間された風の場
        """
        # WindFieldInterpolatorを使用して風の場を補間
        return self.interpolator.interpolate_wind_field(
            target_time=target_time,
            resolution=resolution,
            method=method
        )
    
    def create_wind_field_animation(self, start_time: datetime, end_time: datetime, 
                                  time_steps: int = 10, 
                                  resolution: int = 20) -> List[Dict[str, Any]]:
        """
        時間経過とともに変化する風の場アニメーションを作成
        
        Parameters:
        -----------
        start_time : datetime
            開始時間
        end_time : datetime
            終了時間
        time_steps : int
            時間ステップ数
        resolution : int
            空間解像度
            
        Returns:
        --------
        List[Dict[str, Any]]
            各時間ステップでの風の場データ
        """
        # WindFieldInterpolatorを使用して風の場アニメーションを作成
        return self.interpolator.create_wind_field_animation(
            start_time=start_time,
            end_time=end_time,
            time_steps=time_steps,
            resolution=resolution
        )
    
    def detect_and_fix_gps_anomalies(self, boat_id: str, max_speed_knots: float = 30.0, 
                                   max_accel: float = 2.0) -> pd.DataFrame:
        """
        GPSデータの異常値を検出して修正
        
        Parameters:
        -----------
        boat_id : str
            処理する艇のID
        max_speed_knots : float
            最大想定速度（ノット）- この値を超える速度を異常とみなす
        max_accel : float
            最大想定加速度（m/s^2）- この値を超える加速度を異常とみなす
            
        Returns:
        --------
        pd.DataFrame
            修正されたGPSデータ
        """
        if boat_id not in self.boat_data:
            warnings.warn(f"艇 {boat_id} のデータが見つかりません")
            return None
            
        # 開始時間を記録
        start_time = time.time()
        self._log_performance_step(f"anomaly_detection_start_{boat_id}")
        
        df = self.boat_data[boat_id].copy()
        
        # ノットをm/sに変換 (1ノット = 0.514444 m/s)
        max_speed_ms = max_speed_knots * 0.514444
        
        # 異常な速度を検出
        speed_mask = df['speed'] > max_speed_ms

        # タイムスタンプの差分を秒単位で計算
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
                                       
        # 異常な加速度を検出
        df['acceleration'] = df['speed'].diff() / df['time_diff'].replace(0, np.nan)
        accel_mask = abs(df['acceleration']) > max_accel
        
        # NaNを除外
        accel_mask = accel_mask.fillna(False)
        
        # 異常値のインデックスを取得
        anomaly_indices = df.index[speed_mask | accel_mask].tolist()
        
        # 異常値を処理
        if anomaly_indices:
            for idx in anomaly_indices:
                # 前後の正常なポイントを見つける
                prev_indices = [i for i in range(idx) if i not in anomaly_indices]
                next_indices = [i for i in range(idx + 1, len(df)) if i not in anomaly_indices]
                
                prev_normal_idx = max(prev_indices) if prev_indices else None
                next_normal_idx = min(next_indices) if next_indices else None
                
                if prev_normal_idx is not None and next_normal_idx is not None:
                    # 両側に正常値がある場合は線形補間
                    for col in ['latitude', 'longitude', 'speed']:
                        # 補間する値の計算
                        weight = (idx - prev_normal_idx) / (next_normal_idx - prev_normal_idx)
                        interpolated_value = df.loc[prev_normal_idx, col] + \
                            (df.loc[next_normal_idx, col] - df.loc[prev_normal_idx, col]) * weight
                            
                        df.loc[idx, col] = interpolated_value
                            
                    # ベアリングの補間（円環データなので特殊処理）- bearingカラムが存在する場合のみ
                    if 'bearing' in df.columns:
                        bearing1 = df.loc[prev_normal_idx, 'bearing']
                        bearing2 = df.loc[next_normal_idx, 'bearing']
                        
                        # 角度差を計算（0-360度の循環を考慮）
                        angle_diff = (bearing2 - bearing1 + 180) % 360 - 180
                        
                        # 補間値を計算
                        interp_bearing = (bearing1 + angle_diff * (idx - prev_normal_idx) / 
                                         (next_normal_idx - prev_normal_idx)) % 360
                                         
                        df.loc[idx, 'bearing'] = interp_bearing
                                    
                    elif prev_normal_idx is not None:
                        # 前の値のみがある場合は前方補完
                        cols_to_fill = ['latitude', 'longitude', 'speed']
                        if 'bearing' in df.columns:
                            cols_to_fill.append('bearing')
                        for col in cols_to_fill:
                            df.loc[idx, col] = df.loc[prev_normal_idx, col]
                                    
                    elif next_normal_idx is not None:
                        # 次の値のみがある場合は後方補完
                        cols_to_fill = ['latitude', 'longitude', 'speed']
                        if 'bearing' in df.columns:
                            cols_to_fill.append('bearing')
                        for col in cols_to_fill:
                            df.loc[idx, col] = df.loc[next_normal_idx, col]
        
        # 不要な列を削除
        if 'acceleration' in df.columns:
            df = df.drop(columns=['acceleration'])
        
        # 処理済みデータを保存
        self.processed_data[boat_id] = df
        
        # パフォーマンス統計を更新
        elapsed = time.time() - start_time
        self.performance_stats['process_time'] += elapsed
        
        self._log_performance_step(f"anomaly_detection_end_{boat_id}")
        
        # メモリ最適化
        if self.config['auto_optimize']:
            df = self.optimizer.optimize_dataframe(df)
        
        return df
    
    def process_multiple_boats(self, max_boats: int = None) -> Dict[str, Any]:
        """
        複数艇データの処理（パフォーマンス最適化済み）
        
        Parameters:
        -----------
        max_boats : int, optional
            処理する最大艇数
                
        Returns:
        --------
        Dict[str, Any]
            処理結果と統計情報
        """
        print(f"Debug [PMB]: Starting with {len(self.boat_data)} boats in boat_data. Keys: {list(self.boat_data.keys())}")
        if not self.boat_data:
            print("Debug [PMB]: boat_data is empty, returning empty dict")
            return {}
        
        if max_boats is None:
            max_boats = self.max_boats
        
        # 開始時間を記録
        start_time = time.time()
        self._log_performance_step("process_multiple_start")
        
        # 処理結果を格納する辞書を初期化
        processed_data = {}  # 明示的に処理結果辞書を初期化
        print(f"Debug [PMB]: Initialized empty processed_data dict")
        
        # 艇数制限（メモリ保護）
        boat_ids = list(self.boat_data.keys())[:min(max_boats, len(self.boat_data))]
        print(f"Debug [PMB]: Selected boat_ids for processing: {boat_ids}")
        
        # テスト環境では並列処理を強制的に無効化
        use_parallel = False  # テスト中は常に逐次処理を使用
        print(f"Debug [PMB]: Parallel processing is forcibly disabled for testing")
        
        # 逐次処理（テスト用に必ず実行される）
        print(f"Debug [PMB]: Using sequential processing for {len(boat_ids)} boats")
        for boat_id in boat_ids:
            print(f"Debug [PMB]: Processing boat_id: {boat_id}")
            try:
                # detect_and_fix_gps_anomaliesの結果を変数に保存
                processed_df = self.detect_and_fix_gps_anomalies(boat_id, 30.0, 2.0)
                print(f"Debug [PMB]: detect_and_fix_gps_anomalies returned df with {len(processed_df) if processed_df is not None else 'None'} rows")
                
                if processed_df is not None:
                    # both self.processed_data and processed_data
                    self.processed_data[boat_id] = processed_df
                    processed_data[boat_id] = processed_df
                    print(f"Debug [PMB]: Added processed data for {boat_id}, processed_data now has {len(processed_data)} items")
                else:
                    print(f"Debug [PMB]: Skipping boat_id {boat_id} as processed_df is None")
            except Exception as e:
                print(f"Debug [PMB]: Exception while processing boat_id {boat_id}: {str(e)}")
        
        # 処理後の状態確認
        print(f"Debug [PMB]: After processing all boats, processed_data has {len(processed_data)} items with keys: {list(processed_data.keys())}")
        
        # パフォーマンス統計を更新
        elapsed = time.time() - start_time
        self.performance_stats['process_time'] += elapsed
        
        self._log_performance_step("process_multiple_end")
        
        # 統計情報を作成
        boat_stats = {}
        print(f"Debug [PMB]: Creating stats for {len(processed_data)} boats")
        
        # 処理されたデータを使用して統計情報を作成
        for boat_id, df in processed_data.items():
            try:
                print(f"Debug [PMB]: Creating stats for boat_id: {boat_id}, df size: {len(df)}")
                stats = {
                    'duration_seconds': (df['timestamp'].max() - df['timestamp'].min()).total_seconds(),
                    'avg_speed_knots': (df['speed'].mean() * 1.94384),  # m/s → ノット
                    'max_speed_knots': (df['speed'].max() * 1.94384),
                    'distance_meters': df['distance'].sum() if 'distance' in df.columns else 0,
                    'points_count': len(df)
                }
                boat_stats[boat_id] = stats
                print(f"Debug [PMB]: Created stats for boat_id: {boat_id}")
            except Exception as e:
                print(f"Debug [PMB]: Error creating stats for boat_id {boat_id}: {str(e)}")
        
        # メモリ状況の確認と必要に応じてガベージコレクション
        if self.config['auto_gc'] and self.optimizer.check_memory_threshold():
            gc.collect()
        
        # 明示的に処理データと統計情報を返す
        result = {
            'data': processed_data,  # self.processed_dataではなくprocessed_dataを返す
            'stats': boat_stats
        }
        
        print(f"Debug [PMB]: Returning result with data keys: {list(result['data'].keys())}, stats keys: {list(result['stats'].keys())}")
        return result
    
    def _parallel_process_anomalies(self, args):
        """
        並列処理のための異常値検出・修正関数
        
        Parameters:
        -----------
        args : tuple
            (boat_id, df, max_speed_knots, max_accel)のタプル
            
        Returns:
        --------
        tuple or None
            (boat_id, processed_df)のタプル、失敗時はNone
        """
        boat_id, df, max_speed_knots, max_accel = args
        
        try:
            # ノットをm/sに変換
            max_speed_ms = max_speed_knots * 0.514444
            
            # 処理用にデータをコピー
            df_copy = df.copy()
            
            # 異常な速度を検出
            speed_mask = df_copy['speed'] > max_speed_ms
            
            # 異常な加速度を検出
            df_copy['acceleration'] = df_copy['speed'].diff() / df_copy['time_diff'].replace(0, np.nan)
            accel_mask = abs(df_copy['acceleration']) > max_accel
            
            # NaNを除外
            accel_mask = accel_mask.fillna(False)
            
            # 異常値のインデックスを取得
            anomaly_indices = df_copy.index[speed_mask | accel_mask].tolist()
            
            # 異常値を処理
            if anomaly_indices:
                for idx in anomaly_indices:
                    # 前後の正常なポイントを見つける
                    prev_indices = [i for i in range(idx) if i not in anomaly_indices]
                    next_indices = [i for i in range(idx + 1, len(df_copy)) if i not in anomaly_indices]
                    
                    prev_normal_idx = max(prev_indices) if prev_indices else None
                    next_normal_idx = min(next_indices) if next_indices else None
                    
                    if prev_normal_idx is not None and next_normal_idx is not None:
                        # 両側に正常値がある場合は線形補間
                        for col in ['latitude', 'longitude', 'speed']:
                            # 補間する値の計算
                            weight = (idx - prev_normal_idx) / (next_normal_idx - prev_normal_idx)
                            interpolated_value = df_copy.loc[prev_normal_idx, col] + \
                                (df_copy.loc[next_normal_idx, col] - df_copy.loc[prev_normal_idx, col]) * weight
                                
                            df_copy.loc[idx, col] = interpolated_value
                                
                        # ベアリングの補間（円環データなので特殊処理）
                        bearing1 = df_copy.loc[prev_normal_idx, 'bearing']
                        bearing2 = df_copy.loc[next_normal_idx, 'bearing']
                        
                        # 角度差を計算（0-360度の循環を考慮）
                        angle_diff = (bearing2 - bearing1 + 180) % 360 - 180
                        
                        # 補間値を計算
                        interp_bearing = (bearing1 + angle_diff * (idx - prev_normal_idx) / 
                                         (next_normal_idx - prev_normal_idx)) % 360
                                         
                        df_copy.loc[idx, 'bearing'] = interp_bearing
                    
                    elif prev_normal_idx is not None:
                        # 前の値のみがある場合は前方補完
                        for col in ['latitude', 'longitude', 'speed', 'bearing']:
                            df_copy.loc[idx, col] = df_copy.loc[prev_normal_idx, col]
                    
                    elif next_normal_idx is not None:
                        # 次の値のみがある場合は後方補完
                        for col in ['latitude', 'longitude', 'speed', 'bearing']:
                            df_copy.loc[idx, col] = df_copy.loc[next_normal_idx, col]
            
            # 不要な列を削除
            if 'acceleration' in df_copy.columns:
                df_copy = df_copy.drop(columns=['acceleration'])
            
            # 最適化
            optimizer = PerformanceOptimizer()  # 並列処理用に新しいインスタンス
            df_copy = optimizer.optimize_dataframe(df_copy)
            
            return (boat_id, df_copy)
            
        except Exception as e:
            warnings.warn(f"並列異常値処理エラー ({boat_id}): {e}")
            return None
    
    def synchronize_time(self, target_freq: str = '1s', start_time: Optional[datetime] = None, 
                        end_time: Optional[datetime] = None, min_overlap: float = 0.5) -> Dict[str, pd.DataFrame]:
        """
        複数艇のデータを共通の時間軸に同期（最適化済み）
        
        Parameters:
        -----------
        target_freq : str
            リサンプリング頻度（例: '1s'は1秒間隔）
        start_time : datetime, optional
            開始時刻（指定がない場合は全艇の最も遅い開始時刻）
        end_time : datetime, optional
            終了時刻（指定がない場合は全艇の最も早い終了時刻）
        min_overlap : float
            最小オーバーラップ比率（0〜1）
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            艇ID:時間同期済みDataFrameの辞書
        """
        input_data = self.processed_data if self.processed_data else self.boat_data
        
        if not input_data:
            return {}
            
        # 開始時間を記録
        start_proc_time = time.time()
        self._log_performance_step("sync_start")
        
        # 各艇の開始・終了時刻を取得
        boat_times = {}
        for boat_id, df in input_data.items():
            if 'timestamp' not in df.columns:
                warnings.warn(f"艇 {boat_id} のデータにはタイムスタンプ列がありません")
                continue
                
            boat_times[boat_id] = {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max(),
                'duration': (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
            }
        
        if not boat_times:
            warnings.warn("有効なタイムスタンプを持つ艇データがありません")
            return {}
        
        # 共通の開始・終了時刻を決定
        if start_time is None:
            # デフォルトは全艇の中で最も遅い開始時刻
            start_time = max([times['start'] for times in boat_times.values()])
        
        if end_time is None:
            # デフォルトは全艇の中で最も早い終了時刻
            end_time = min([times['end'] for times in boat_times.values()])
        
        # 時間範囲の妥当性チェック
        if start_time >= end_time:
            warnings.warn("開始時刻が終了時刻以降です。有効な共通時間枠がありません。")
            return {}
        
        # 共通時間枠の長さ
        common_duration = (end_time - start_time).total_seconds()
        
        # 各艇のオーバーラップを計算し、基準を満たす艇を選択
        valid_boats = []
        for boat_id, times in boat_times.items():
            # 艇の時間範囲と共通時間枠のオーバーラップを計算
            overlap_start = max(times['start'], start_time)
            overlap_end = min(times['end'], end_time)
            
            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                overlap_ratio = overlap_duration / common_duration
                
                if overlap_ratio >= min_overlap:
                    valid_boats.append(boat_id)
                else:
                    warnings.warn(f"艇 {boat_id} は共通時間枠との重複が少なすぎます（{overlap_ratio:.1%}）。同期から除外します。")
        
        if not valid_boats:
            warnings.warn("共通時間枠で有効な艇データがありません")
            return {}
        
        # 共通の時間インデックスを作成
        # メモリ最適化のため、長すぎる場合はダウンサンプリング
        target_freq_seconds = pd.Timedelta(target_freq).total_seconds()
        expected_points = int(common_duration / target_freq_seconds) + 1
        
        if expected_points > self.config['downsample_threshold']:
            # 時間間隔を調整してポイント数を削減
            adjusted_freq = f"{int(common_duration / self.config['downsample_threshold'])}s"
            warnings.warn(f"共通時間枠が大きすぎます。時間間隔を {target_freq} から {adjusted_freq} に調整します。")
            target_freq = adjusted_freq
        
        common_timeindex = pd.date_range(start=start_time, end=end_time, freq=target_freq)
        
        # 各艇のデータを共通時間軸に再サンプリング
        synced_data = {}
        
        # 並列処理を使用するかどうか
        use_parallel = self.config['use_parallel'] and len(valid_boats) > 3  # 3艇以上で並列化
        
        if use_parallel:
            # 並列処理用にタスクを準備
            tasks = []
            for boat_id in valid_boats:
                tasks.append((boat_id, input_data[boat_id], common_timeindex, self.config['auto_optimize']))
            
            # マルチプロセッシングのためのワーカー数
            n_workers = min(len(tasks), self.optimizer.max_workers)
            
            # 並列処理の実行
            parallel_results = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # 部分関数を作成
                func = partial(self._parallel_resample_boat)
                # プールで実行
                futures = [executor.submit(func, task) for task in tasks]
                parallel_results = [future.result() for future in futures if future.result() is not None]
            
            # 結果を統合
            for result in parallel_results:
                if result is not None:
                    boat_id, synced_df = result
                    if synced_df is not None:
                        synced_data[boat_id] = synced_df
        else:
            # 逐次処理
            for boat_id in valid_boats:
                df = input_data[boat_id]
                
                # 必須カラムとオプションカラムをチェック
                df = self._ensure_columns(
                    df,
                    required_cols=['timestamp', 'latitude', 'longitude'],
                    optional_cols=['speed', 'bearing'],
                    boat_id=boat_id
                )
                # 必要なカラムだけを抽出
                df_subset = df[['timestamp', 'latitude', 'longitude', 'speed', 'bearing']].copy()
                
                # タイムスタンプの重複を検出して処理
                if df_subset['timestamp'].duplicated().any():
                    warnings.warn(f"艇 {boat_id} に重複したタイムスタンプが存在します。微小オフセットを追加します。")
                    
                    # 重複インデックスを取得
                    dup_mask = df_subset['timestamp'].duplicated(keep=False)
                    
                    # 重複があるタイムスタンプ値を取得
                    dup_times = df_subset.loc[dup_mask, 'timestamp'].unique()
                    
                    # 各重複タイムスタンプに対応
                    for dup_time in dup_times:
                        # 同じタイムスタンプを持つ行のインデックスを取得
                        indices = df_subset[df_subset['timestamp'] == dup_time].index.tolist()
                        
                        # 微小オフセット（マイクロ秒）を追加
                        for i, idx in enumerate(indices):
                            # 最初の重複インデックス以外に微小オフセットを適用
                            if i > 0:
                                offset = i * 1000  # マイクロ秒単位のオフセット
                                df_subset.loc[idx, 'timestamp'] += pd.Timedelta(microseconds=offset)
                
                # タイムスタンプをインデックスに設定
                df_subset = df_subset.set_index('timestamp')
                
                # 共通時間軸の範囲内のデータを抽出
                mask = (df_subset.index >= start_time) & (df_subset.index <= end_time)
                if mask.any():
                    df_subset = df_subset[mask]
                    
                    # 線形補間で再サンプリング
                    df_resampled = df_subset.reindex(
                        common_timeindex,
                        method=None  # 補間なし（後で各列に適した補間を適用）
                    )
                    
                    # 各列に適した補間方法を適用
                    # 緯度・経度・速度は線形補間
                    for col in ['latitude', 'longitude', 'speed']:
                        df_resampled[col] = df_resampled[col].interpolate(method='linear', limit=10)
                    
                    # 方位（bearing）は円環データなので特殊な補間が必要
                    # シンプルな線形補間でベアリングを補間
                    if 'bearing' in df_resampled.columns:
                        # 単純な角度の差分が180度を超える場合に問題が生じるため
                        # sin/cosに分解して補間
                        bearings = df_subset['bearing'].values
                        sin_bearings = np.sin(np.radians(bearings))
                        cos_bearings = np.cos(np.radians(bearings))
                        
                        # 補間用のDataFrameを作成
                        temp_df = pd.DataFrame({
                            'sin': sin_bearings,
                            'cos': cos_bearings
                        }, index=df_subset.index)
                        
                        # 補間
                        temp_resampled = temp_df.reindex(common_timeindex)
                        temp_resampled['sin'] = temp_resampled['sin'].interpolate(method='linear', limit=10)
                        temp_resampled['cos'] = temp_resampled['cos'].interpolate(method='linear', limit=10)
                        
                        # 角度に戻す
                        bearings_new = np.degrees(np.arctan2(temp_resampled['sin'].values, 
                                                           temp_resampled['cos'].values)) % 360
                        df_resampled['bearing'] = bearings_new
                    
                    # boat_id列を追加
                    df_resampled['boat_id'] = boat_id
                    
                    # タイムスタンプを列として復元
                    df_resampled = df_resampled.reset_index()
                    df_resampled = df_resampled.rename(columns={'index': 'timestamp'})
                    
                    # 結果の最適化
                    if self.config['auto_optimize']:
                        df_resampled = self.optimizer.optimize_dataframe(df_resampled)
                    
                    synced_data[boat_id] = df_resampled
        
        self.synced_data = synced_data
        
        # パフォーマンス統計を更新
        elapsed = time.time() - start_proc_time
        self.performance_stats['sync_time'] += elapsed
        
        self._log_performance_step("sync_end")
        
        # メモリ状況の確認と必要に応じてガベージコレクション
        if self.config['auto_gc'] and self.optimizer.check_memory_threshold():
            gc.collect()
        
        return synced_data
    
    def _parallel_resample_boat(self, args):
        """
        並列処理のための艇データ再サンプリング関数
        
        Parameters:
        -----------
        args : tuple
            (boat_id, df, common_timeindex, auto_optimize)のタプル
            
        Returns:
        --------
        tuple or None
            (boat_id, resampled_df)のタプル、失敗時はNone
        """
        boat_id, df, common_timeindex, auto_optimize = args
        
        try:
            # メモリ効率のためにコピーは最小限に
            df_subset = df[['timestamp', 'latitude', 'longitude', 'speed', 'bearing']].copy()
            
            # タイムスタンプの重複を検出して処理
            if df_subset['timestamp'].duplicated().any():
                # 重複インデックスを取得
                dup_mask = df_subset['timestamp'].duplicated(keep=False)
                
                # 重複があるタイムスタンプ値を取得
                dup_times = df_subset.loc[dup_mask, 'timestamp'].unique()
                
                # 各重複タイムスタンプに対応
                for dup_time in dup_times:
                    # 同じタイムスタンプを持つ行のインデックスを取得
                    indices = df_subset[df_subset['timestamp'] == dup_time].index.tolist()
                    
                    # 微小オフセット（マイクロ秒）を追加
                    for i, idx in enumerate(indices):
                        # 最初の重複インデックス以外に微小オフセットを適用
                        if i > 0:
                            offset = i * 1000  # マイクロ秒単位のオフセット
                            df_subset.loc[idx, 'timestamp'] += pd.Timedelta(microseconds=offset)
            
            # タイムスタンプをインデックスに設定
            df_subset = df_subset.set_index('timestamp')
            
            # 共通時間軸の範囲内のデータを抽出
            start_time = common_timeindex[0]
            end_time = common_timeindex[-1]
            mask = (df_subset.index >= start_time) & (df_subset.index <= end_time)
            if mask.any():
                df_subset = df_subset[mask]
                
                # 線形補間で再サンプリング
                df_resampled = df_subset.reindex(
                    common_timeindex,
                    method=None  # 補間なし（後で各列に適した補間を適用）
                )
                
                # 各列に適した補間方法を適用
                # 緯度・経度・速度は線形補間
                for col in ['latitude', 'longitude', 'speed']:
                    df_resampled[col] = df_resampled[col].interpolate(method='linear', limit=10)
                
                # 方位（bearing）は円環データなのでsin/cosに分解して補間
                if 'bearing' in df_resampled.columns:
                    bearings = df_subset['bearing'].values
                    sin_bearings = np.sin(np.radians(bearings))
                    cos_bearings = np.cos(np.radians(bearings))
                    
                    # 補間用のDataFrameを作成
                    temp_df = pd.DataFrame({
                        'sin': sin_bearings,
                        'cos': cos_bearings
                    }, index=df_subset.index)
                    
                    # 補間
                    temp_resampled = temp_df.reindex(common_timeindex)
                    temp_resampled['sin'] = temp_resampled['sin'].interpolate(method='linear', limit=10)
                    temp_resampled['cos'] = temp_resampled['cos'].interpolate(method='linear', limit=10)
                    
                    # 角度に戻す
                    bearings_new = np.degrees(np.arctan2(temp_resampled['sin'].values, 
                                                       temp_resampled['cos'].values)) % 360
                    df_resampled['bearing'] = bearings_new
                
                # boat_id列を追加
                df_resampled['boat_id'] = boat_id
                
                # タイムスタンプを列として復元
                df_resampled = df_resampled.reset_index()
                df_resampled = df_resampled.rename(columns={'index': 'timestamp'})
                
                # 結果の最適化
                if auto_optimize:
                    optimizer = PerformanceOptimizer()  # 並列処理用に新しいインスタンス
                    df_resampled = optimizer.optimize_dataframe(df_resampled)
                
                return (boat_id, df_resampled)
            else:
                return None
                
        except Exception as e:
            warnings.warn(f"並列再サンプリングエラー ({boat_id}): {e}")
            return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        処理パフォーマンスレポートを取得
        
        Returns:
        --------
        Dict[str, Any]
            パフォーマンス統計情報
        """
        # 現在のメモリ使用状況を追加
        memory_info = self.optimizer.get_memory_usage()
        
        # レポートを作成
        report = {
            'load_time_seconds': self.performance_stats['load_time'],
            'process_time_seconds': self.performance_stats['process_time'],
            'sync_time_seconds': self.performance_stats['sync_time'],
            'wind_estimation_time_seconds': self.performance_stats['wind_estimation_time'],
            'total_time_seconds': (self.performance_stats['load_time'] + 
                                 self.performance_stats['process_time'] + 
                                 self.performance_stats['sync_time'] +
                                 self.performance_stats['wind_estimation_time']),
            'total_points_processed': self.performance_stats['total_points_processed'],
            'current_memory_usage': memory_info,
            'memory_history': self.performance_stats['memory_usage'] if self.config['log_performance'] else [],
            'boat_count': len(self.boat_data),
            'processed_boat_count': len(self.processed_data),
            'synced_boat_count': len(self.synced_data),
            'wind_estimates_count': len(self.wind_estimates)
        }
        
        # 処理効率
        if report['total_time_seconds'] > 0:
            report['overall_efficiency'] = report['total_points_processed'] / report['total_time_seconds']
        else:
            report['overall_efficiency'] = 0
            
        # 平均データポイント数
        if len(self.boat_data) > 0:
            report['avg_points_per_boat'] = sum(len(df) for df in self.boat_data.values()) / len(self.boat_data)
        else:
            report['avg_points_per_boat'] = 0
            
        return report
    
    def cleanup_memory(self, keep_types: List[str] = None):
        """
        不要なデータをメモリから解放
        
        Parameters:
        -----------
        keep_types : List[str], optional
            保持するデータ型のリスト ('raw', 'processed', 'synced', 'wind')
        """
        if keep_types is None:
            keep_types = ['synced', 'wind']  # デフォルトでは同期済みデータと風推定データのみ保持
            
        if 'raw' not in keep_types:
            # 元データは保持しない
            self.boat_data = {}
            
        if 'processed' not in keep_types:
            # 処理済みデータは保持しない
            self.processed_data = {}
            
        if 'synced' not in keep_types:
            # 同期済みデータは保持しない
            self.synced_data = {}
            
        if 'wind' not in keep_types:
            # 風推定データは保持しない
            self.wind_estimates = {}
            self.wind_field_data = {}
            
        # 強制的にガベージコレクションを実行
        gc.collect()
        
        # 現在のメモリ使用状況をログに記録
        self._log_performance_step("memory_cleanup")

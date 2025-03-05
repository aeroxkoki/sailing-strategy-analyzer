import pandas as pd
import numpy as np
import gpxpy
import io
import math
from geopy.distance import geodesic
from datetime import datetime, timedelta
from scipy import interpolate
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

class SailingDataProcessor:
    """セーリングデータ処理クラス - GPSデータの読み込み、前処理、分析を担当"""
    
    def __init__(self):
        """初期化"""
        self.boat_data = {}  # 艇ID: DataFrameの辞書
        self.processed_data = {}  # 処理済みデータ
        self.synced_data = {}  # 時間同期済みデータ
        self.max_boats = 80  # 最大処理可能艇数

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
        
        # 最大艇数をチェック
        if len(file_contents) > self.max_boats:
            warnings.warn(f"ファイル数が最大処理可能艇数({self.max_boats})を超えています。最初の{self.max_boats}ファイルのみ処理します。")
            file_contents = file_contents[:self.max_boats]
            if not auto_id and manual_ids is not None:
                manual_ids = manual_ids[:self.max_boats]
        
        for idx, (filename, content, filetype) in enumerate(file_contents):
            # IDの決定
            if auto_id:
                # ファイル名から自動生成（拡張子を除く）
                boat_id = filename.split('.')[0]
                
                # 同じIDが存在する場合は連番を付加
                base_id = boat_id
                counter = 1
                while boat_id in self.boat_data:
                    boat_id = f"{base_id}_{counter}"
                    counter += 1
            else:
                boat_id = manual_ids[idx]
                
            # ファイルタイプに応じた読み込み
            if filetype.lower() == 'gpx':
                df = self._load_gpx(content.decode('utf-8'), boat_id)
            elif filetype.lower() == 'csv':
                df = self._load_csv(content, boat_id)
            else:
                warnings.warn(f"未対応のファイル形式です: {filetype}")
                continue
                
            if df is not None:
                self.boat_data[boat_id] = df
                
        return self.boat_data

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
                    points.extend([{
                        'timestamp': point.time,
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation if point.elevation is not None else 0,
                        'boat_id': boat_id
                    } for point in segment.points])
            
            # 十分なポイントがない場合
            if len(points) < 10:
                warnings.warn(f"{boat_id}: GPXファイルに十分なトラックポイントがありません")
                return None
            
            # DataFrameに変換
            df = pd.DataFrame(points)
            
            # タイムスタンプを日時型に変換
            if df['timestamp'].dtype != 'datetime64[ns]':
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 前処理（速度、方位などの計算）
            df = self._preprocess_gps_data(df)
            
            return df
            
        except Exception as e:
            warnings.warn(f"{boat_id}: GPXファイルの読み込みエラー: {str(e)}")
            return None

    def _load_csv(self, csv_content: bytes, boat_id: str) -> Optional[pd.DataFrame]:
        """
        CSVデータを読み込み、DataFrameに変換
        
        Parameters:
        -----------
        csv_content : bytes
            CSVファイルの内容
        boat_id : str
            艇ID
            
        Returns:
        --------
        Optional[pd.DataFrame]
            解析済みGPSデータ、読み込み失敗時はNone
        """
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
            
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
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 速度や方位が含まれていない場合は計算
            df = self._preprocess_gps_data(df)
            
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
        
        # 距離計算
        df['distance'] = 0.0
        for i in range(1, len(df)):
            df.loc[i, 'distance'] = geodesic(
                (df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']),
                (df.loc[i, 'latitude'], df.loc[i, 'longitude'])
            ).meters
        
        # 速度計算（メートル/秒）- 既に速度がある場合は上書きしない
        if 'speed' not in df.columns:
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
        if 'bearing' not in df.columns:
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

    def synchronize_time(self, target_freq: str = '1s', start_time: Optional[datetime] = None, 
                        end_time: Optional[datetime] = None, min_overlap: float = 0.5) -> Dict[str, pd.DataFrame]:
        """
        複数艇のデータを共通の時間軸に同期
        
        Parameters:
        -----------
        target_freq : str
            リサンプリング頻度（例: '1s'は1秒間隔）
        start_time : datetime, optional
            開始時刻（指定がない場合は全艇の最も遅い開始時刻）
        end_time : datetime, optional
            終了時刻（指定がない場合は全艇の最も早い終了時刻）
        min_overlap : float
            最小オーバーラップ比率（0〜1）- 共通時間枠にこの比率以上含まれる艇のみ処理
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            艇IDをキーとする時間同期済みDataFrameの辞書
        """
        if not self.boat_data:
            return {}
            
        # 各艇の開始・終了時刻を取得
        boat_times = {}
        for boat_id, df in self.boat_data.items():
            boat_times[boat_id] = {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max(),
                'duration': (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
            }
        
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
        
        # 共通の時間インデックスを作成
        common_timeindex = pd.date_range(start=start_time, end=end_time, freq=target_freq)
        
        # 各艇のデータを共通時間軸に再サンプリング
        synced_data = {}
        
        for boat_id in valid_boats:
            df = self.boat_data[boat_id]
            
            # 必要な列のみを抽出
            df_subset = df[['timestamp', 'latitude', 'longitude', 'speed', 'bearing']].copy()
            
            # タイムスタンプの重複を検出して処理
            # タイムスタンプに重複がある場合は微小なオフセットを追加
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
                # 0度と360度の境界を考慮した補間
                bearings = df_subset['bearing'].values
                times = np.array([(t - start_time).total_seconds() for t in df_subset.index])
                times_new = np.array([(t - start_time).total_seconds() for t in common_timeindex])
                
                if len(times) > 0 and len(bearings) > 0:
                    # ベアリングデータをsin/cosに分解して補間
                    sin_bearings = np.sin(np.radians(bearings))
                    cos_bearings = np.cos(np.radians(bearings))
                    
                    if len(times) >= 4 and len(np.unique(times)) >= 4:  # スプライン補間には最低4点必要
                        try:
                            sin_interp = interpolate.splrep(times, sin_bearings, s=0)
                            cos_interp = interpolate.splrep(times, cos_bearings, s=0)
                            
                            sin_new = interpolate.splev(times_new, sin_interp, der=0)
                            cos_new = interpolate.splev(times_new, cos_interp, der=0)
                        except Exception:
                            # スプライン補間に失敗した場合は線形補間にフォールバック
                            sin_interp = interpolate.interp1d(times, sin_bearings, kind='linear', 
                                                           bounds_error=False, fill_value='extrapolate')
                            cos_interp = interpolate.interp1d(times, cos_bearings, kind='linear',
                                                           bounds_error=False, fill_value='extrapolate')
                            
                            sin_new = sin_interp(times_new)
                            cos_new = cos_interp(times_new)
                    else:  # 点が少ない場合は線形補間
                        sin_interp = interpolate.interp1d(times, sin_bearings, kind='linear', 
                                                       bounds_error=False, fill_value='extrapolate')
                        cos_interp = interpolate.interp1d(times, cos_bearings, kind='linear',
                                                       bounds_error=False, fill_value='extrapolate')
                        
                        sin_new = sin_interp(times_new)
                        cos_new = cos_interp(times_new)
                    
                    # 角度に戻す
                    bearings_new = np.degrees(np.arctan2(sin_new, cos_new)) % 360
                    df_resampled['bearing'] = bearings_new
                
                # boat_id列を追加
                df_resampled['boat_id'] = boat_id
                
                # タイムスタンプを列として復元
                df_resampled = df_resampled.reset_index()
                df_resampled = df_resampled.rename(columns={'index': 'timestamp'})
                
                synced_data[boat_id] = df_resampled
        
        self.synced_data = synced_data
        return synced_data

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
            
        df = self.boat_data[boat_id].copy()
        
        # ノットをm/sに変換 (1ノット = 0.514444 m/s)
        max_speed_ms = max_speed_knots * 0.514444
        
        # 異常な速度を検出
        speed_mask = df['speed'] > max_speed_ms
        
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
                            
                    # ベアリングの補間（円環データなので特殊処理）
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
                    for col in ['latitude', 'longitude', 'speed', 'bearing']:
                        df.loc[idx, col] = df.loc[prev_normal_idx, col]
                
                elif next_normal_idx is not None:
                    # 次の値のみがある場合は後方補完
                    for col in ['latitude', 'longitude', 'speed', 'bearing']:
                        df.loc[idx, col] = df.loc[next_normal_idx, col]
        
        # 不要な列を削除
        if 'acceleration' in df.columns:
            df = df.drop(columns=['acceleration'])
        
        # 処理済みデータを保存
        self.processed_data[boat_id] = df
        
        return df

    def process_multiple_boats(self, max_boats: int = 80) -> Dict[str, Any]:
        """
        複数艇データの並列処理（将来的に80艇までスケール可能）
        
        Parameters:
        -----------
        max_boats : int
            処理する最大艇数
            
        Returns:
        --------
        Dict[str, Any]
            処理結果と統計情報
        """
        if not self.boat_data:
            return {}
        
        # 艇数が多い場合のメモリ効率向上のため、必要な列のみを保持
        essential_columns = ['timestamp', 'latitude', 'longitude', 'speed', 'bearing', 'boat_id']
        
        # 各艇を処理
        processed_results = {}
        boat_stats = {}
        
        # 艇数制限（メモリ保護）
        boat_ids = list(self.boat_data.keys())[:min(max_boats, len(self.boat_data))]
        
        for boat_id in boat_ids:
            # 異常値検出・修正
            clean_df = self.detect_and_fix_gps_anomalies(boat_id)
            
            if clean_df is not None:
                # 必要な列のみを保持
                available_columns = [col for col in essential_columns if col in clean_df.columns]
                clean_df = clean_df[available_columns].copy()
                
                # 基本統計情報を計算
                stats = {
                    'duration_seconds': (clean_df['timestamp'].max() - clean_df['timestamp'].min()).total_seconds(),
                    'avg_speed_knots': (clean_df['speed'].mean() * 1.94384),  # m/s → ノット
                    'max_speed_knots': (clean_df['speed'].max() * 1.94384),
                    'distance_meters': clean_df['distance'].sum() if 'distance' in clean_df.columns else 0,
                    'points_count': len(clean_df)
                }
                
                boat_stats[boat_id] = stats
                processed_results[boat_id] = clean_df
        
        self.processed_data = processed_results
        return {
            'data': processed_results,
            'stats': boat_stats
        }

    def export_processed_data(self, boat_id: str, format_type: str = 'csv') -> Optional[bytes]:
        """
        処理済みデータをエクスポート
        
        Parameters:
        -----------
        boat_id : str
            エクスポートする艇のID
        format_type : str
            エクスポート形式（'csv' または 'json'）
            
        Returns:
        --------
        bytes
            エクスポートされたデータ（バイナリ形式）
        """
        data_to_export = self.synced_data.get(boat_id) if boat_id in self.synced_data else \
                        self.processed_data.get(boat_id) if boat_id in self.processed_data else \
                        self.boat_data.get(boat_id)
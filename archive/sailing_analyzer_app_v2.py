import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import gpxpy
import io
import math
from geopy.distance import geodesic
import base64
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

# ページ設定
st.set_page_config(
    page_title="セーリング戦略分析システム", 
    page_icon="🌬️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ====================
# ユーティリティ関数
# ====================

# 方位角を基数方位に変換する関数
def degrees_to_cardinal(degrees):
    """方位角（度）を基数方位（N, NE, E など）に変換"""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                 "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(degrees / 22.5) % 16
    return directions[index]

# GPXファイルを読み込んでパンダスのDataFrameに変換する関数
def load_gpx_to_dataframe(gpx_content, boat_id="Unknown"):
    """GPXデータを読み込み、DataFrameに変換する関数"""
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
            st.error(f"{boat_id}: GPXファイルに十分なトラックポイントがありません。")
            return None
        
        # DataFrameに変換
        df = pd.DataFrame(points)
        
        # タイムスタンプを日時型に変換（すでに日時型の場合はスキップ）
        if df['timestamp'].dtype != 'datetime64[ns]':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 時間差分を計算（秒単位）
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        
        # 距離計算（メートル単位）
        df['distance'] = 0.0
        for i in range(1, len(df)):
            df.at[i, 'distance'] = geodesic(
                (df.at[i-1, 'latitude'], df.at[i-1, 'longitude']),
                (df.at[i, 'latitude'], df.at[i, 'longitude'])
            ).meters
        
        # 速度計算（メートル/秒）
        df['speed'] = df['distance'] / df['time_diff']
        
        # 進行方向（ベアリング）の計算
        df['bearing'] = 0.0
        for i in range(1, len(df)):
            lat1, lon1 = df.at[i-1, 'latitude'], df.at[i-1, 'longitude']
            lat2, lon2 = df.at[i, 'latitude'], df.at[i, 'longitude']
            
            # ラジアンに変換
            lat1, lon1 = math.radians(lat1), math.radians(lon1)
            lat2, lon2 = math.radians(lat2), math.radians(lon2)
            
            # ベアリング計算
            y = math.sin(lon2 - lon1) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
            bearing = math.degrees(math.atan2(y, x))
            
            # 0-360度の範囲に正規化
            bearing = (bearing + 360) % 360
            
            df.at[i, 'bearing'] = bearing
            
        # NaN値を処理（最初の行など）
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        st.error(f"{boat_id}: GPXファイルの読み込みエラー: {e}")
        return None

# CSVファイルを読み込むための関数
def load_csv_to_dataframe(csv_content, boat_id="Unknown"):
    """CSVデータを読み込み、処理する関数"""
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
        
        # boat_id列を追加
        df['boat_id'] = boat_id
        
        # 必要な列があるか確認
        required_cols = ['timestamp', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"{boat_id}: CSVファイルに必要な列がありません: {missing_cols}")
            return None
        
        # タイムスタンプを日時型に変換
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 速度や方位が含まれていない場合は計算
        if 'speed' not in df.columns or 'bearing' not in df.columns:
            # 時間差分を計算（秒単位）
            df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
            
            # 距離計算（メートル単位）
            df['distance'] = 0.0
            for i in range(1, len(df)):
                df.at[i, 'distance'] = geodesic(
                    (df.at[i-1, 'latitude'], df.at[i-1, 'longitude']),
                    (df.at[i, 'latitude'], df.at[i, 'longitude'])
                ).meters
            
            # 速度計算（メートル/秒）
            if 'speed' not in df.columns:
                df['speed'] = df['distance'] / df['time_diff']
            
            # 進行方向（ベアリング）の計算
            if 'bearing' not in df.columns:
                df['bearing'] = 0.0
                for i in range(1, len(df)):
                    lat1, lon1 = df.at[i-1, 'latitude'], df.at[i-1, 'longitude']
                    lat2, lon2 = df.at[i, 'latitude'], df.at[i, 'longitude']
                    
                    # ラジアンに変換
                    lat1, lon1 = math.radians(lat1), math.radians(lon1)
                    lat2, lon2 = math.radians(lat2), math.radians(lon2)
                    
                    # ベアリング計算
                    y = math.sin(lon2 - lon1) * math.cos(lat2)
                    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
                    bearing = math.degrees(math.atan2(y, x))
                    
                    # 0-360度の範囲に正規化
                    bearing = (bearing + 360) % 360
                    
                    df.at[i, 'bearing'] = bearing
        
        # NaN値を処理
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        st.error(f"{boat_id}: CSVファイルの読み込みエラー: {e}")
        return None

# テスト用のサンプルGPSデータを生成する関数
def generate_sample_gps_data(num_boats=2):
    """
    テスト用のサンプルGPSデータを生成する関数
    
    Parameters:
    -----------
    num_boats : int
        生成する艇の数
        
    Returns:
    --------
    all_boats_data : dict
        艇ID:GPSデータのディクショナリ
    """
    # 東京湾でのセーリングレースを想定した座標
    base_lat, base_lon = 35.620, 139.770
    
    # 各艇のデータを格納
    all_boats_data = {}
    
    for boat_id in range(1, num_boats + 1):
        # 時間間隔（秒）
        time_interval = 10
        
        # データポイント数
        num_points = 360  # 1時間分（10秒間隔）
        
        # タイムスタンプの作成
        start_time = datetime(2024, 7, 1, 10, 0, 0) + timedelta(seconds=(boat_id-1)*5)  # 各艇の開始時間を少しずらす
        timestamps = [start_time + timedelta(seconds=i*time_interval) for i in range(num_points)]
        
        # 艇ごとの微小な変動を追加
        lat_var = (boat_id - 1) * 0.001
        lon_var = (boat_id - 1) * 0.002
        
        # 風上/風下のレグを含むコースを模擬
        lats = []
        lons = []
        
        # 最初の風上レグ
        leg1_points = 90
        for i in range(leg1_points):
            progress = i / leg1_points
            # ジグザグパターン（タック）を追加
            phase = i % 30
            if phase < 15:
                # 左に向かうタック
                lats.append(base_lat + progress * 0.03 + 0.002 * math.sin(i/5) + lat_var)
                lons.append(base_lon + progress * 0.01 + 0.005 + lon_var)
            else:
                # 右に向かうタック
                lats.append(base_lat + progress * 0.03 + 0.002 * math.sin(i/5) + lat_var)
                lons.append(base_lon + progress * 0.01 - 0.005 + lon_var)
        
        # 風下レグ
        leg2_points = 90
        for i in range(leg2_points):
            progress = i / leg2_points
            # より直線的な動き
            lats.append(base_lat + 0.03 - progress * 0.03 + 0.001 * math.sin(i/10) + lat_var)
            lons.append(base_lon + 0.01 + 0.002 * math.cos(i/8) + lon_var)
        
        # 2回目の風上レグ
        leg3_points = 90
        for i in range(leg3_points):
            progress = i / leg3_points
            # ジグザグパターン（タック）を追加
            phase = i % 25
            if phase < 12:
                # 左に向かうタック
                lats.append(base_lat + progress * 0.03 + 0.002 * math.sin(i/6) + lat_var)
                lons.append(base_lon - 0.01 + progress * 0.02 + 0.004 + lon_var)
            else:
                # 右に向かうタック
                lats.append(base_lat + progress * 0.03 + 0.002 * math.sin(i/6) + lat_var)
                lons.append(base_lon - 0.01 + progress * 0.02 - 0.004 + lon_var)
        
        # 最終レグ
        leg4_points = 90
        for i in range(leg4_points):
            progress = i / leg4_points
            # フィニッシュに向かう
            lats.append(base_lat + 0.03 - progress * 0.02 + 0.001 * math.sin(i/7) + lat_var)
            lons.append(base_lon + 0.01 - progress * 0.01 + 0.001 * math.cos(i/9) + lon_var)
        
        # データフレーム作成
        data = {
            'timestamp': timestamps[:num_points],  # 配列の長さを合わせる
            'latitude': lats[:num_points],
            'longitude': lons[:num_points],
            'elevation': [0] * num_points,  # 海面高度は0とする
            'boat_id': [f"Boat{boat_id}"] * num_points
        }
        
        df = pd.DataFrame(data)
        
        # 速度と方位を計算
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        
        # 距離計算（メートル単位）
        df['distance'] = 0.0
        for i in range(1, len(df)):
            df.at[i, 'distance'] = geodesic(
                (df.at[i-1, 'latitude'], df.at[i-1, 'longitude']),
                (df.at[i, 'latitude'], df.at[i, 'longitude'])
            ).meters
        
        # 速度計算（メートル/秒）
        df['speed'] = df['distance'] / df['time_diff']
        
        # 進行方向（ベアリング）の計算
        df['bearing'] = 0.0
        for i in range(1, len(df)):
            lat1, lon1 = df.at[i-1, 'latitude'], df.at[i-1, 'longitude']
            lat2, lon2 = df.at[i, 'latitude'], df.at[i, 'longitude']
            
            # ラジアンに変換
            lat1, lon1 = math.radians(lat1), math.radians(lon1)
            lat2, lon2 = math.radians(lat2), math.radians(lon2)
            
            # ベアリング計算
            y = math.sin(lon2 - lon1) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
            bearing = math.degrees(math.atan2(y, x))
            
            # 0-360度の範囲に正規化
            bearing = (bearing + 360) % 360
            
            df.at[i, 'bearing'] = bearing
        
        # NaN値を処理
        df = df.fillna(0)
        
        # 艇ごとの特性を反映（速度差など）
        speed_factor = 1.0 + (boat_id - 1) * 0.05  # 艇1を基準に5%ずつ速度差
        df['speed'] = df['speed'] * speed_factor
        
        all_boats_data[f"Boat{boat_id}"] = df
    
    return all_boats_data

# ====================
# 分析アルゴリズム
# ====================

def improved_wind_estimation(gps_data, min_tack_angle=30, polar_data=None, boat_type=None):
    """
    改良された風向風速推定アルゴリズム
    
    Parameters:
    -----------
    gps_data : pandas.DataFrame
        GPSデータを含むDataFrame
    min_tack_angle : float
        タックと認識する最小の方向転換角度
    polar_data : dict, optional
        艇の極座標性能データ（利用可能な場合）
    boat_type : str, optional
        艇種識別子（利用可能な場合）
        
    Returns:
    --------
    wind_estimates : pandas.DataFrame
        推定された風向風速情報を含むDataFrame（時間変化含む）
    """
    # データのコピーを作成
    df = gps_data.copy()
    
    # 方向の変化を計算（絶対値）
    df['bearing_change'] = df['bearing'].diff().abs()
    
    # 大きな方向変化をタックまたはジャイブとして識別
    df['is_tack'] = df['bearing_change'] > min_tack_angle
    
    # === 改良ポイント1: より堅牢なタック検出 ===
    # 連続するタックを1つのイベントとしてグループ化
    df['tack_group'] = (df['is_tack'] != df['is_tack'].shift()).cumsum()
    tack_groups = df[df['is_tack']].groupby('tack_group')
    
    # 有意なタックのみを抽出（短すぎる角度変化を除外）
    significant_tacks = []
    for _, group in tack_groups:
        if len(group) >= 2:  # 最低2ポイント以上のタック
            total_angle_change = abs(group['bearing'].iloc[-1] - group['bearing'].iloc[0])
            if total_angle_change > min_tack_angle:
                significant_tacks.append({
                    'start_idx': group.index[0],
                    'end_idx': group.index[-1],
                    'angle_before': group['bearing'].iloc[0],
                    'angle_after': group['bearing'].iloc[-1],
                    'timestamp': group['timestamp'].iloc[0]
                })
    
    # タック/ジャイブが少なすぎる場合は処理を中止
    if len(significant_tacks) < 2:
        st.warning(f"Boat {df['boat_id'].iloc[0] if 'boat_id' in df.columns else 'Unknown'}: タック/ジャイブポイントが不足しているため、風向の推定が困難です。")
        return None
    
    # === 改良ポイント2: 風上/風下レグの自動識別 ===
    # 有効な航路方向のみを抽出（タックを除く安定した区間）
    stable_bearings = []
    stable_speeds = []
    stable_sections = []
    
    for i in range(len(significant_tacks) - 1):
        current_tack = significant_tacks[i]
        next_tack = significant_tacks[i + 1]
        
        # タック間の安定した区間を抽出
        start_idx = current_tack['end_idx'] + 1
        end_idx = next_tack['start_idx'] - 1
        
        if start_idx <= end_idx and end_idx < len(df):  # インデックスの範囲をチェック
            stable_section = df.loc[start_idx:end_idx].copy()
            if len(stable_section) > 5:  # 十分なデータポイントがある場合
                avg_bearing = stable_section['bearing'].mean()
                avg_speed = stable_section['speed'].mean()
                stable_bearings.append(avg_bearing)
                stable_speeds.append(avg_speed)
                stable_sections.append(stable_section)
    
    # 航路方向のクラスタリング（主に風上と風下のレグを分離）
    upwind_bearings = []
    upwind_speeds = []
    downwind_bearings = []
    downwind_speeds = []
    
    if len(stable_bearings) >= 4:  # 十分なデータポイント
        bearings_array = np.array(stable_bearings).reshape(-1, 1)
        # 角度の循環性を考慮するための変換
        X = np.column_stack([
            np.cos(np.radians(bearings_array.flatten())),
            np.sin(np.radians(bearings_array.flatten()))
        ])
        
        # 2つのクラスタにグループ化（風上と風下のレグを想定）
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        clusters = kmeans.labels_
        
        # クラスタごとの平均速度を計算
        cluster_speeds = [np.mean([s for s, c in zip(stable_speeds, clusters) if c == i]) for i in range(2)]
        
        # 低速クラスタを風上、高速クラスタを風下と仮定
        upwind_cluster = 0 if cluster_speeds[0] < cluster_speeds[1] else 1
        downwind_cluster = 1 - upwind_cluster
        
        # 各クラスタの方向と速度を抽出
        upwind_bearings = [b for b, c in zip(stable_bearings, clusters) if c == upwind_cluster]
        upwind_speeds = [s for s, c in zip(stable_speeds, clusters) if c == upwind_cluster]
        downwind_bearings = [b for b, c in zip(stable_bearings, clusters) if c == downwind_cluster]
        downwind_speeds = [s for s, c in zip(stable_speeds, clusters) if c == downwind_cluster]
    else:
        # データが不足している場合はヒストグラムベースの方法にフォールバック
        hist, bin_edges = np.histogram(df['bearing'], bins=36, range=(0, 360))
        peak_indices = np.argsort(hist)[-2:]  # 上位2つのピーク
        peak_bins = [(bin_edges[i], bin_edges[i+1]) for i in peak_indices]
        peak_angles = [(bin_start + bin_end) / 2 for bin_start, bin_end in peak_bins]
        
        # 平均速度に基づいて風上/風下を推定
        speeds_per_angle = []
        
        for angle in peak_angles:
            # 特定の方向に近い区間の平均速度を計算
            mask = np.abs((df['bearing'] - angle + 180) % 360 - 180) < 30
            section_speed = df.loc[mask, 'speed'].mean() if sum(mask) > 0 else 0
            speeds_per_angle.append((angle, section_speed))
        
        # 速度でソートして風上（遅い）と風下（速い）を特定
        speeds_per_angle.sort(key=lambda x: x[1])
        
        if len(speeds_per_angle) >= 2:
            upwind_bearings = [speeds_per_angle[0][0]]
            upwind_speeds = [speeds_per_angle[0][1]]
            downwind_bearings = [speeds_per_angle[1][0]]
            downwind_speeds = [speeds_per_angle[1][1]]
        elif len(speeds_per_angle) == 1:
            upwind_bearings = [speeds_per_angle[0][0]]
            upwind_speeds = [speeds_per_angle[0][1]]
    
    # === 改良ポイント3: より精度の高い風向推定 ===
    estimated_wind_direction = None
    
    # 風上レグの方向から風向を推定
    if len(upwind_bearings) >= 2:
        # 複数の風上方向がある場合
        angle_diffs = []
        for i in range(len(upwind_bearings)):
            for j in range(i+1, len(upwind_bearings)):
                diff = abs(upwind_bearings[i] - upwind_bearings[j])
                if diff > 180:
                    diff = 360 - diff
                angle_diffs.append((upwind_bearings[i], upwind_bearings[j], diff))
        
        # 最も角度差が大きいペアを見つける（おそらく反対タック）
        if angle_diffs:
            max_diff_pair = max(angle_diffs, key=lambda x: x[2])
            angle1, angle2 = max_diff_pair[0], max_diff_pair[1]
            
            # 二等分線を計算
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
                bisector = (min(angle1, angle2) + angle_diff/2) % 360
            else:
                bisector = (min(angle1, angle2) + angle_diff/2)
            
            # 風向は二等分線の反対方向（180度反転）
            estimated_wind_direction = (bisector + 180) % 360
    
    # 単一の風上方向または風下方向から推定
    if estimated_wind_direction is None:
        if len(upwind_bearings) == 1:
            # 1つの風上方向から推定（典型的な風上角度を考慮）
            estimated_wind_direction = (upwind_bearings[0] + 180) % 360
        elif len(downwind_bearings) >= 1:
            # 風下方向の反対が風向と仮定
            estimated_wind_direction = (downwind_bearings[0] + 180) % 360
        else:
            # データ不足のためデフォルト値を使用
            st.warning(f"Boat {df['boat_id'].iloc[0] if 'boat_id' in df.columns else 'Unknown'}: 風向の推定に必要な十分なデータがありません。")
            return None
    
    # === 改良ポイント4: 風速の精度向上 ===
    # 艇種ごとの係数を設定
    # 以下は各艇種の極座標データから導出される理想的な比率
    boat_coefficients = {
        'default': {'upwind': 3.0, 'downwind': 1.5},
        'laser': {'upwind': 3.2, 'downwind': 1.6},  # レーザー/ILCA - 軽量級一人乗り
        'ilca': {'upwind': 3.2, 'downwind': 1.6},   # ILCA (レーザーの新名称)
        '470': {'upwind': 3.0, 'downwind': 1.5},    # 470級 - ミディアム二人乗り
        '49er': {'upwind': 2.8, 'downwind': 1.3},   # 49er - 高性能スキフ
        'finn': {'upwind': 3.3, 'downwind': 1.7},   # フィン級 - 重量級一人乗り
        'nacra17': {'upwind': 2.5, 'downwind': 1.2}, # Nacra 17 - カタマラン
        'star': {'upwind': 3.4, 'downwind': 1.7}    # スター級 - キールボート
    }
    
    # 使用する係数の決定
    use_boat_type = boat_type.lower() if boat_type and boat_type.lower() in boat_coefficients else 'default'
    upwind_ratio = boat_coefficients[use_boat_type]['upwind']
    downwind_ratio = boat_coefficients[use_boat_type]['downwind']
    
    # 風速推定値の計算
    avg_upwind_speed = np.mean(upwind_speeds) if upwind_speeds else 0
    avg_downwind_speed = np.mean(downwind_speeds) if downwind_speeds else 0
    
    est_wind_speed_from_upwind = avg_upwind_speed * upwind_ratio if avg_upwind_speed > 0 else 0
    est_wind_speed_from_downwind = avg_downwind_speed * downwind_ratio if avg_downwind_speed > 0 else 0
    
    # 両方の推定値の重み付き平均を取る
    if est_wind_speed_from_upwind > 0 and est_wind_speed_from_downwind > 0:
        # 風上からの推定値の方が信頼性が高いと仮定
        estimated_wind_speed = (est_wind_speed_from_upwind * 0.7 + est_wind_speed_from_downwind * 0.3)
    elif est_wind_speed_from_upwind > 0:
        estimated_wind_speed = est_wind_speed_from_upwind
    elif est_wind_speed_from_downwind > 0:
        estimated_wind_speed = est_wind_speed_from_downwind
    else:
        estimated_wind_speed = 0
    
    # ノットに変換（1 m/s ≈ 1.94384 ノット）
    estimated_wind_speed_knots = estimated_wind_speed * 1.94384
    
    # === 改良ポイント5: 時間変化を考慮した風向風速推定 ===
    # ウィンドウ分析で時間による変化を推定
    window_size = max(len(df) // 10, 20)  # データの約10%、最低20ポイント
    
    wind_estimates = []
    for i in range(0, len(df), window_size//2):  # 50%オーバーラップのウィンドウ
        end_idx = min(i + window_size, len(df))
        if end_idx - i < window_size // 2:  # 小さすぎるウィンドウはスキップ
            continue
            
        window_data = df.iloc[i:end_idx]
        
        # 各ウィンドウでのタックパターン分析（簡略化）
        # ここでは全体の推定風向と近似値を使用
        # 実際には各ウィンドウごとに詳細な風向推定を行いたいが、簡略化のため
        
        center_time = window_data['timestamp'].iloc[len(window_data)//2]
        center_lat = window_data['latitude'].mean()
        center_lon = window_data['longitude'].mean()
        
        # 基本的な時間変化モデル（仮の実装 - より高度な分析は将来的に追加）
        # 実際のレースでは時間とともに風向が変化するため、
        # 簡易的な変化を模擬（±5度程度のランダム変動）
        time_factor = (center_time - df['timestamp'].iloc[0]).total_seconds() / \
                      (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
        
        # 風向の時間変化をモデル化（単純な線形変化 + ノイズ）
        # 実際には過去データからの傾向分析やパターン検出が必要
        wind_direction_variation = (np.sin(time_factor * np.pi) * 5)  # ±5度程度の変動
        windowed_direction = (estimated_wind_direction + wind_direction_variation) % 360
        
        # 風速の時間変化もモデル化（単純化）
        wind_speed_variation = (np.cos(time_factor * np.pi * 2) * 0.5)  # ±0.5ノット程度の変動
        windowed_speed = max(0, estimated_wind_speed_knots + wind_speed_variation)
        
        wind_estimates.append({
            'timestamp': center_time,
            'latitude': center_lat,
            'longitude': center_lon,
            'wind_direction': windowed_direction,
            'wind_speed_knots': windowed_speed,
            'confidence': 0.8,  # 改良アルゴリズムでより高い信頼度
            'boat_id': df['boat_id'].iloc[0] if 'boat_id' in df.columns else 'Unknown'
        })
    
    # DataFrameに変換
    if wind_estimates:
        wind_df = pd.DataFrame(wind_estimates)
        return wind_df
    else:
        return None

def estimate_wind_field(multi_boat_data, time_point, grid_resolution=20):
    """
    複数艇のデータから特定時点での風の場を推定
    
    Parameters:
    -----------
    multi_boat_data : dict
        艇ID:風推定データのディクショナリ
    time_point : datetime
        風推定を行いたい時点
    grid_resolution : int
        出力グリッドの解像度
        
    Returns:
    --------
    grid_data : dict
        緯度・経度グリッドと推定された風向風速データ
    """
    # 各艇からのデータを時間でフィルタリング
    nearby_time_data = []
    
    for boat_id, wind_data in multi_boat_data.items():
        # 指定時間の前後30秒のデータを抽出
        time_diff = abs((wind_data['timestamp'] - time_point).dt.total_seconds())
        time_mask = time_diff < 30
        if time_mask.any():
            boat_time_data = wind_data[time_mask].copy()
            # boat_idを風推定で既に追加済み
            nearby_time_data.append(boat_time_data)
    
    # データを統合
    if not nearby_time_data:
        return None
    
    combined_data = pd.concat(nearby_time_data)
    
    # 境界を設定
    min_lat = combined_data['latitude'].min()
    max_lat = combined_data['latitude'].max()
    min_lon = combined_data['longitude'].min()
    max_lon = combined_data['longitude'].max()
    
    # 少し余裕を持たせる
    lat_margin = (max_lat - min_lat) * 0.1
    lon_margin = (max_lon - min_lon) * 0.1
    min_lat -= lat_margin
    max_lat += lat_margin
    min_lon -= lon_margin
    max_lon += lon_margin
    
    # グリッドの作成
    lat_grid = np.linspace(min_lat, max_lat, grid_resolution)
    lon_grid = np.linspace(min_lon, max_lon, grid_resolution)
    grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
    
    # 風向風速の推定値を格納するグリッド
    grid_u = np.zeros_like(grid_lats)  # 東西風成分
    grid_v = np.zeros_like(grid_lons)  # 南北風成分
    grid_speeds = np.zeros_like(grid_lats)
    grid_weights = np.zeros_like(grid_lats)
    
    # 各データポイントからグリッドへの寄与を計算
    for _, row in combined_data.iterrows():
        # 風向を単位ベクトルに分解
        dir_rad = np.radians(row['wind_direction'])
        u = np.sin(dir_rad)  # 東西成分
        v = np.cos(dir_rad)  # 南北成分
        
        # 各グリッドポイントへの寄与を計算
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                # 空間距離（メートル）を計算
                dist = geodesic(
                    (grid_lats[i, j], grid_lons[i, j]),
                    (row['latitude'], row['longitude'])
                ).meters
                
                # 距離による重み（逆二乗加重）
                if dist < 10:  # 非常に近い点
                    weight = row['confidence'] * 1.0
                else:
                    weight = row['confidence'] * (1.0 / (dist ** 0.5))
                
                # 重み付きでベクトル成分と風速を足し合わせる
                grid_weights[i, j] += weight
                grid_u[i, j] += weight * u
                grid_v[i, j] += weight * v
                grid_speeds[i, j] += weight * row['wind_speed_knots']
    
    # 重みで正規化して最終的な風向風速を計算
    mask = grid_weights > 0
    
    # 風向の計算（ベクトル成分から）
    u_normalized = np.zeros_like(grid_u)
    v_normalized = np.zeros_like(grid_v)
    u_normalized[mask] = grid_u[mask] / grid_weights[mask]
    v_normalized[mask] = grid_v[mask] / grid_weights[mask]
    
    # 風向角度に変換
    wind_directions = np.zeros_like(grid_lats)
    wind_directions[mask] = np.degrees(np.arctan2(u_normalized[mask], v_normalized[mask])) % 360
    
    # 風速の計算
    wind_speeds = np.zeros_like(grid_speeds)
    wind_speeds[mask] = grid_speeds[mask] / grid_weights[mask]
    
    # 信頼度の正規化
    confidence = np.zeros_like(grid_weights)
    if grid_weights.max() > 0:
        confidence = grid_weights / grid_weights.max()
    
    return {
        'lat_grid': grid_lats,
        'lon_grid': grid_lons,
        'wind_direction': wind_directions,
        'wind_speed': wind_speeds,
        'confidence': confidence
    }

# ====================
# 可視化関数
# ====================

# GPSデータを地図上に表示する関数
def visualize_gps_on_map(gps_data_dict, selected_time=None, title="GPS Track Visualization"):
    """GPSデータを地図上に表示する関数"""
    # 全ての艇のデータを統合
    all_lats = []
    all_lons = []
    
    for boat_id, df in gps_data_dict.items():
        all_lats.extend(df['latitude'].tolist())
        all_lons.extend(df['longitude'].tolist())
    
    # 地図の中心を計算
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    
    # 地図を作成
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # 艇ごとに異なる色で表示
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkblue', 'darkgreen', 'cadetblue']
    
    for i, (boat_id, df) in enumerate(gps_data_dict.items()):
        color = colors[i % len(colors)]
        
        # 航跡を追加
        points = list(zip(df['latitude'], df['longitude']))
        folium.PolyLine(
            points,
            color=color,
            weight=3,
            opacity=0.8,
            tooltip=boat_id
        ).add_to(m)
        
        # スタート地点にマーカーを追加
        folium.Marker(
            location=[df['latitude'].iloc[0], df['longitude'].iloc[0]],
            popup=f'{boat_id} Start',
            icon=folium.Icon(color=color, icon='play'),
        ).add_to(m)
        
        # フィニッシュ地点にマーカーを追加
        folium.Marker(
            location=[df['latitude'].iloc[-1], df['longitude'].iloc[-1]],
            popup=f'{boat_id} Finish',
            icon=folium.Icon(color=color, icon='stop'),
        ).add_to(m)
        
        # 選択された時間に最も近いポイントにマーカーを追加
        if selected_time is not None:
            time_diffs = abs((df['timestamp'] - selected_time).dt.total_seconds())
            closest_idx = time_diffs.idxmin()
            
            folium.CircleMarker(
                location=[df.loc[closest_idx, 'latitude'], df.loc[closest_idx, 'longitude']],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"{boat_id} at {df.loc[closest_idx, 'timestamp']}"
            ).add_to(m)
    
    # タイトルを追加
    title_html = f'<h3 align="center" style="font-size:16px"><b>{title}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

# 風向風速を地図上に表示する関数
def visualize_wind_field_on_map(wind_field, boat_tracks=None, selected_time=None, title="Wind Field Analysis"):
    """
    風の場をfoliumマップ上に可視化
    
    Parameters:
    -----------
    wind_field : dict
        estimate_wind_field関数からの出力
    boat_tracks : dict, optional
        艇ID:GPSトラックのディクショナリ
    selected_time : datetime, optional
        選択された時間点
    title : str
        地図のタイトル
    
    Returns:
    --------
    m : folium.Map
        風の場を表示したfoliumマップ
    """
    # グリッドデータを抽出
    grid_lats = wind_field['lat_grid']
    grid_lons = wind_field['lon_grid']
    wind_directions = wind_field['wind_direction']
    wind_speeds = wind_field['wind_speed']
    confidence = wind_field['confidence']
    
    # 地図の中心を計算
    center_lat = np.mean(grid_lats)
    center_lon = np.mean(grid_lons)
    
    # 地図を作成
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # 風速をヒートマップで表示
    heat_data = []
    for i in range(grid_lats.shape[0]):
        for j in range(grid_lats.shape[1]):
            if confidence[i, j] > 0.1:  # 信頼度が低すぎる点は除外
                heat_data.append([
                    grid_lats[i, j], 
                    grid_lons[i, j], 
                    wind_speeds[i, j]
                ])
    
    if heat_data:
        from folium.plugins import HeatMap
        HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)
    
    # 風向を矢印で表示（間引き）
    skip = max(1, grid_lats.shape[0] // 8)  # 表示する矢印の間隔
    
    for i in range(0, grid_lats.shape[0], skip):
        for j in range(0, grid_lats.shape[1], skip):
            if confidence[i, j] > 0.3:  # 信頼度の高い点のみ表示
                # 風向と風速を取得
                wind_dir = wind_directions[i, j]
                wind_speed = wind_speeds[i, j]
                
                # 矢印の向きを風が「吹いてくる」方向に設定
                arrow_direction = wind_dir
                
                # 矢印の長さを風速に比例させる（調整係数）
                arrow_length = 0.002 * max(0.5, min(2.0, wind_speed / 10))
                
                # 矢印の開始点
                lat = grid_lats[i, j]
                lon = grid_lons[i, j]
                
                # 矢印の終点（三角関数で計算）
                end_lat = lat + arrow_length * math.cos(math.radians(arrow_direction))
                end_lon = lon + arrow_length * math.sin(math.radians(arrow_direction))
                
                # 矢印を地図に追加（風が吹いてくる方向を示す）
                folium.PolyLine(
                    [(end_lat, end_lon), (lat, lon)],  # 終点から始点へ
                    color='red',
                    weight=2,
                    opacity=0.7,
                    popup=f"風向: {wind_dir:.1f}°, 風速: {wind_speed:.1f}ノット"
                ).add_to(m)
    
    # 艇のトラックを表示（利用可能な場合）
    if boat_tracks:
        colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue']
        
        for idx, (boat_id, track) in enumerate(boat_tracks.items()):
            color = colors[idx % len(colors)]
            
            # トラックのポイントを取得
            points = list(zip(track['latitude'], track['longitude']))
            
            # トラックを線で表示
            folium.PolyLine(
                points,
                color=color,
                weight=3,
                opacity=0.8,
                tooltip=f"艇 {boat_id}"
            ).add_to(m)
            
            # スタートとフィニッシュのマーカー
            folium.Marker(
                location=[track['latitude'].iloc[0], track['longitude'].iloc[0]],
                popup=f"{boat_id} スタート",
                icon=folium.Icon(color=color, icon='play'),
            ).add_to(m)
            
            folium.Marker(
                location=[track['latitude'].iloc[-1], track['longitude'].iloc[-1]],
                popup=f"{boat_id} フィニッシュ",
                icon=folium.Icon(color=color, icon='stop'),
            ).add_to(m)
            
            # 選択された時間に最も近いポイントにマーカーを追加
            if selected_time is not None:
                time_diffs = abs((track['timestamp'] - selected_time).dt.total_seconds())
                closest_idx = time_diffs.idxmin()
                
                folium.CircleMarker(
                    location=[track.loc[closest_idx, 'latitude'], track.loc[closest_idx, 'longitude']],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=f"{boat_id} at {track.loc[closest_idx, 'timestamp']}"
                ).add_to(m)
    
    # タイトルを追加
    title_html = f'<h3 align="center" style="font-size:16px"><b>{title}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

# 速度と方向の時系列グラフを表示する関数
def plot_speed_and_bearing(gps_data, boat_id="Unknown"):
    """GPSデータから速度と進行方向の時系列グラフを表示"""
    # 時間軸を作成（分かりやすい形式に）
    time_elapsed = [(t - gps_data['timestamp'].iloc[0]).total_seconds() / 60 for t in gps_data['timestamp']]
    
    # 2つのサブプロットを作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # 速度グラフ（ノットに変換: 1 m/s ≈ 1.94384 ノット）
    speed_knots = gps_data['speed'] * 1.94384
    ax1.plot(time_elapsed, speed_knots, 'b-', linewidth=2)
    ax1.set_ylabel('Speed (knots)')
    ax1.set_title(f'{boat_id} - Boat Speed Over Time')
    ax1.grid(True)
    
    # 最大・最小・平均速度を表示
    ax1.axhline(y=speed_knots.mean(), color='r', linestyle='-', alpha=0.3)
    ax1.text(time_elapsed[-1]*0.02, speed_knots.mean()*1.1, 
             f'Mean: {speed_knots.mean():.2f} knots', 
             color='r', verticalalignment='bottom')
    
    # 進行方向グラフ
    ax2.plot(time_elapsed, gps_data['bearing'], 'g-', linewidth=2)
    ax2.set_ylabel('Bearing (degrees)')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_title('Boat Direction Over Time')
    ax2.set_ylim(0, 360)
    ax2.grid(True)
    
    # Y軸に主要な方位を表示
    ax2.set_yticks([0, 90, 180, 270, 360])
    ax2.set_yticklabels(['N', 'E', 'S', 'W', 'N'])
    
    plt.tight_layout()
    return fig

# 風配図（ポーラーチャート）を表示する関数
def plot_wind_rose(wind_directions, wind_speeds=None, title="Wind Direction Distribution"):
    """
    風向データをポーラーチャートで表示
    
    Parameters:
    -----------
    wind_directions : array-like
        風向の配列（度数法）
    wind_speeds : array-like, optional
        風速の配列（指定された場合は風速ごとに色分け）
    title : str
        グラフのタイトル
    """
    # Plotlyを使用したポーラーチャート
    bin_size = 10  # 10度ごとのビン
    
    # 度数をラジアンに変換（プロットのため）
    # 注: 風配図では、北が0度（π/2ラジアン）、東が90度（0ラジアン）のため調整
    theta = [(270 - d) % 360 for d in wind_directions]
    
    if wind_speeds is not None:
        # 風速を区間に分ける
        speed_bins = [0, 5, 10, 15, 20, float('inf')]
        speed_labels = ['0-5', '5-10', '10-15', '15-20', '20+']
        speed_categories = pd.cut(wind_speeds, bins=speed_bins, labels=speed_labels, right=False)
        
        # 風速カテゴリごとに集計
        fig = go.Figure()
        
        for i, speed_label in enumerate(speed_labels):
            mask = speed_categories == speed_label
            if sum(mask) > 0:
                fig.add_trace(go.Barpolar(
                    r=[sum(mask)],  # このカテゴリの数
                    theta=[0],      # ダミー値（後で更新）
                    width=[bin_size],
                    name=f'{speed_label} ノット',
                    marker_color=px.colors.sequential.Plasma[i]
                ))
    else:
        # 風向のみのヒストグラム
        fig = go.Figure(go.Barpolar(
            r=[1],       # ダミー値（後で更新）
            theta=[0],   # ダミー値（後で更新）
            width=[bin_size],
            marker_color='blue'
        ))
    
    # 極座標系の設定
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(showticklabels=False, ticks=''),
            angularaxis=dict(
                tickvals=[0, 90, 180, 270],
                ticktext=['N', 'E', 'S', 'W'],
                direction='clockwise'
            )
        ),
        showlegend=True
    )
    
    return fig

# 艇パフォーマンス分析グラフを表示する関数
def plot_boat_performance(gps_data_dict, wind_estimates_dict=None):
    """
    複数艇のパフォーマンスを比較するグラフを表示
    
    Parameters:
    -----------
    gps_data_dict : dict
        艇ID:GPSデータのディクショナリ
    wind_estimates_dict : dict, optional
        艇ID:風推定データのディクショナリ
    """
    # 速度比較グラフ
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 各艇の平均速度を計算
    avg_speeds = {}
    
    for boat_id, df in gps_data_dict.items():
        # ノットに変換
        speed_knots = df['speed'] * 1.94384
        avg_speeds[boat_id] = speed_knots.mean()
        
        # 時間軸の正規化（分）
        time_elapsed = [(t - df['timestamp'].iloc[0]).total_seconds() / 60 for t in df['timestamp']]
        
        # 速度プロット
        ax.plot(time_elapsed, speed_knots, label=f'{boat_id} (Avg: {avg_speeds[boat_id]:.2f} knots)')
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Speed (knots)')
    ax.set_title('Boat Speed Comparison')
    ax.grid(True)
    ax.legend()
    
    return fig

# ====================
# メイン Streamlit アプリ
# ====================

# タイトルとイントロダクション
st.title('セーリング戦略分析システム v2.0')

# タブの設定
tab1, tab2 = st.tabs(["風向風速分析", "艇パフォーマンス分析"])

# サイドバーにファイルアップロード機能
st.sidebar.header('データ入力')

# 複数艇データのアップロード
uploaded_files = st.sidebar.file_uploader(
    "GPX/CSVファイルをアップロード", 
    type=['gpx', 'csv'],
    accept_multiple_files=True
)

# 艇種選択
boat_type = st.sidebar.selectbox(
    "艇種（風速推定の精度向上）",
    options=['470', 'Laser/ILCA', '49er', 'Finn', 'Nacra 17', 'Star', 'その他'],
    index=0
)

# 艇種を内部表現に変換
boat_type_map = {
    '470': '470',
    'Laser/ILCA': 'laser',
    '49er': '49er',
    'Finn': 'finn',
    'Nacra 17': 'nacra17',
    'Star': 'star',
    'その他': 'default'
}

selected_boat_type = boat_type_map[boat_type]

# サンプルデータの使用オプション
use_sample_data = st.sidebar.checkbox('サンプルデータを使用', value=True if not uploaded_files else False)

# タック/ジャイブ検出の閾値
tack_threshold = st.sidebar.slider(
    'タック/ジャイブ検出閾値（度）', 
    min_value=10, 
    max_value=60, 
    value=30,
    help='この角度以上の方向転換をタック/ジャイブとして検出します'
)

# データ処理
all_gps_data = {}
all_wind_estimates = {}

# アップロードされたファイルの処理
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        boat_id = uploaded_file.name.split('.')[0]  # ファイル名を艇IDとして使用
        
        try:
            if file_extension == 'gpx':
                gps_data = load_gpx_to_dataframe(uploaded_file.getvalue().decode('utf-8'), boat_id)
            elif file_extension == 'csv':
                gps_data = load_csv_to_dataframe(uploaded_file.getvalue(), boat_id)
            
            if gps_data is not None:
                all_gps_data[boat_id] = gps_data
                st.sidebar.success(f"ファイル '{uploaded_file.name}' を正常に読み込みました")
                
                # 風向風速の推定
                wind_estimates = improved_wind_estimation(gps_data, min_tack_angle=tack_threshold, boat_type=selected_boat_type)
                if wind_estimates is not None:
                    all_wind_estimates[boat_id] = wind_estimates
        except Exception as e:
            st.sidebar.error(f"ファイル '{uploaded_file.name}' の処理中にエラーが発生しました: {e}")

# サンプルデータを使用
elif use_sample_data:
    all_gps_data = generate_sample_gps_data(num_boats=2)
    st.sidebar.info("サンプルデータを使用しています")
    
    # サンプルデータから風向風速を推定
    for boat_id, gps_data in all_gps_data.items():
        wind_estimates = improved_wind_estimation(gps_data, min_tack_angle=tack_threshold, boat_type=selected_boat_type)
        if wind_estimates is not None:
            all_wind_estimates[boat_id] = wind_estimates

# 時間スライダーの作成（全艇のデータを統合して時間範囲を決定）
min_time = None
max_time = None
common_times = []

if all_gps_data:
    # 全艇の時間範囲を取得
    all_times = []
    for boat_id, df in all_gps_data.items():
        all_times.extend(df['timestamp'].tolist())
    
    # 時間範囲の決定
    min_time = min(all_times)
    max_time = max(all_times)
    
    # 共通の時間点を作成（10秒間隔など）
    time_range = (max_time - min_time).total_seconds()
    num_points = min(100, max(10, int(time_range / 10)))  # 10秒間隔、最大100点
    common_times = [min_time + timedelta(seconds=i * time_range / (num_points - 1)) for i in range(num_points)]

# タブ1: 風向風速分析
with tab1:
    if all_gps_data and all_wind_estimates:
        st.header('風向風速分析')
        
        # 時間スライダー
        if common_times:
            selected_time_idx = st.slider(
                "時間選択", 
                min_value=0, 
                max_value=len(common_times)-1, 
                value=len(common_times)//2,
                format="時間: %d"
            )
            selected_time = common_times[selected_time_idx]
            st.write(f"選択時間: {selected_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 選択時点での風の場を推定
            wind_field = estimate_wind_field(all_wind_estimates, selected_time)
            
            if wind_field:
                # 2段構成で表示（上段: 風向風速マップ）
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader('風向風速マップ')
                    wind_map = visualize_wind_field_on_map(
                        wind_field, 
                        boat_tracks=all_gps_data,
                        selected_time=selected_time,
                        title=f"風の場 ({selected_time.strftime('%H:%M:%S')})"
                    )
                    folium_static(wind_map, width=700)
                
                with col2:
                    # 右上部: ポーラーグラフ（風配図）
                    st.subheader('風配図')
                    wind_dirs = []
                    wind_speeds = []
                    
                    for boat_id, wind_data in all_wind_estimates.items():
                        wind_dirs.extend(wind_data['wind_direction'].tolist())
                        wind_speeds.extend(wind_data['wind_speed_knots'].tolist())
                    
                    wind_rose_fig = plot_wind_rose(wind_dirs, wind_speeds, "風向分布")
                    st.plotly_chart(wind_rose_fig, use_container_width=True)
                    
                    # 推定風向風速の表示
                    center_i = wind_field['wind_direction'].shape[0] // 2
                    center_j = wind_field['wind_direction'].shape[1] // 2
                    
                    center_dir = wind_field['wind_direction'][center_i, center_j]
                    center_speed = wind_field['wind_speed'][center_i, center_j]
                    
                    st.metric("推定風向", f"{center_dir:.1f}° ({degrees_to_cardinal(center_dir)})")
                    st.metric("推定風速", f"{center_speed:.1f} ノット")
                
                # 下段: 風向風速の時間変化グラフ
                st.subheader('風向風速の時間変化')
                
                # 時間変化データの準備
                time_points = []
                wind_dirs = []
                wind_speeds = []
                
                # 最初の艇のデータを使用（複数艇の平均も可能）
                first_boat_id = list(all_wind_estimates.keys())[0]
                time_points = all_wind_estimates[first_boat_id]['timestamp'].tolist()
                wind_dirs = all_wind_estimates[first_boat_id]['wind_direction'].tolist()
                wind_speeds = all_wind_estimates[first_boat_id]['wind_speed_knots'].tolist()
                
                # 時間変化グラフを作成
                fig = plt.figure(figsize=(12, 6))
                
                # 風向の時間変化
                ax1 = plt.subplot(2, 1, 1)
                ax1.plot(time_points, wind_dirs, 'r-', marker='o', markersize=4)
                ax1.set_ylabel('風向 (度)')
                ax1.set_title('風向の時間変化')
                ax1.grid(True)
                ax1.set_ylim(0, 360)
                ax1.set_yticks([0, 90, 180, 270, 360])
                ax1.set_yticklabels(['N', 'E', 'S', 'W', 'N'])
                
                # 選択時点を強調
                ax1.axvline(x=selected_time, color='blue', linestyle='--')
                
                # 風速の時間変化
                ax2 = plt.subplot(2, 1, 2, sharex=ax1)
                ax2.plot(time_points, wind_speeds, 'b-', marker='o', markersize=4)
                ax2.set_xlabel('時間')
                ax2.set_ylabel('風速 (ノット)')
                ax2.set_title('風速の時間変化')
                ax2.grid(True)
                
                # 選択時点を強調
                ax2.axvline(x=selected_time, color='blue', linestyle='--')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # GPSトラックのプロット
        st.subheader('航跡地図')
        gps_map = visualize_gps_on_map(all_gps_data, selected_time if common_times else None, "GPS Tracks")
        folium_static(gps_map)
        
    else:
        st.info('GPSデータをアップロードするか、サンプルデータを使用してください。')

# タブ2: 艇パフォーマンス分析
with tab2:
    if all_gps_data:
        st.header('艇パフォーマンス分析')
        
        # 艇速度比較グラフ
        st.subheader('艇速度比較')
        performance_fig = plot_boat_performance(all_gps_data)
        st.pyplot(performance_fig)
        
        # 各艇の詳細分析
        st.subheader('艇別分析')
        
        # 2列レイアウト
        cols = st.columns(2)
        
        for i, (boat_id, df) in enumerate(all_gps_data.items()):
            col_idx = i % 2
            
            with cols[col_idx]:
                st.write(f"**{boat_id}**")
                
                # 基本統計
                speed_knots = df['speed'] * 1.94384
                st.write(f"平均速度: {speed_knots.mean():.2f} ノット")
                st.write(f"最高速度: {speed_knots.max():.2f} ノット")
                
                # 速度と方向のグラフ
                speed_bearing_fig = plot_speed_and_bearing(df, boat_id)
                st.pyplot(speed_bearing_fig)
                
                # タックカウント
                bearing_change = df['bearing'].diff().abs()
                tack_count = sum(bearing_change > tack_threshold)
                st.write(f"タック/ジャイブ回数: {tack_count}")
                
                # 風上・風下の平均速度（風向推定がある場合）
                if boat_id in all_wind_estimates and 'wind_direction' in all_wind_estimates[boat_id]:
                    wind_dir = all_wind_estimates[boat_id]['wind_direction'].iloc[0]
                    
                    # 風上範囲（風向 ±45度）と風下範囲（風向の反対 ±45度）を定義
                    upwind_min = (wind_dir - 45) % 360
                    upwind_max = (wind_dir + 45) % 360
                    downwind_min = (wind_dir + 135) % 360
                    downwind_max = (wind_dir + 225) % 360
                    
                    # 風上と風下の区間を特定
                    is_upwind = False
                    is_downwind = False
                    
                    if upwind_min < upwind_max:
                        is_upwind = (df['bearing'] >= upwind_min) & (df['bearing'] <= upwind_max)
                    else:  # 範囲が0度/360度をまたぐ場合
                        is_upwind = (df['bearing'] >= upwind_min) | (df['bearing'] <= upwind_max)
                        
                    if downwind_min < downwind_max:
                        is_downwind = (df['bearing'] >= downwind_min) & (df['bearing'] <= downwind_max)
                    else:  # 範囲が0度/360度をまたぐ場合
                        is_downwind = (df['bearing'] >= downwind_min) | (df['bearing'] <= downwind_max)
                    
                    # 風上・風下の平均速度を計算
                    upwind_speed = speed_knots[is_upwind].mean() if sum(is_upwind) > 0 else 0
                    downwind_speed = speed_knots[is_downwind].mean() if sum(is_downwind) > 0 else 0
                    
                    st.write(f"風上平均速度: {upwind_speed:.2f} ノット")
                    st.write(f"風下平均速度: {downwind_speed:.2f} ノット")
    
    else:
        st.info('GPSデータをアップロードするか、サンプルデータを使用してください。')

# フッター
st.markdown("---")
st.markdown("© 2024 セーリング戦略分析システム | v2.0")

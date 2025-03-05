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
from datetime import datetime

# ページ設定
st.set_page_config(page_title="セーリング戦略分析システム", layout="wide")

# タイトルとイントロダクション
st.title('セーリング戦略分析システム')
st.markdown("""
このアプリケーションは、セーリング競技のGPSデータから風向風速を推定し、コース戦略の分析を行います。
GPXまたはCSVファイルをアップロードするか、サンプルデータを使用して機能をお試しください。
""")

# 方位角を基数方位に変換する関数
def degrees_to_cardinal(degrees):
    """方位角（度）を基数方位（N, NE, E など）に変換"""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                 "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(degrees / 22.5) % 16
    return directions[index]

# GPXファイルを読み込んでパンダスのDataFrameに変換する関数
def load_gpx_to_dataframe(gpx_content):
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
                    })
        
        # 十分なポイントがない場合
        if len(points) < 10:
            st.error("GPXファイルに十分なトラックポイントがありません。")
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
        st.error(f"GPXファイルの読み込みエラー: {e}")
        return None

# CSVファイルを読み込むための関数
def load_csv_to_dataframe(csv_content):
    """CSVデータを読み込み、処理する関数"""
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
        
        # 必要な列があるか確認
        required_cols = ['timestamp', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"CSVファイルに必要な列がありません: {missing_cols}")
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
        st.error(f"CSVファイルの読み込みエラー: {e}")
        return None

# テスト用のサンプルGPSデータを生成する関数
def generate_sample_gps_data():
    """テスト用のサンプルGPSデータを生成する関数"""
    # 東京湾でのセーリングレースを想定した座標
    base_lat, base_lon = 35.620, 139.770
    
    # 時間間隔（秒）
    time_interval = 10
    
    # データポイント数
    num_points = 360  # 1時間分（10秒間隔）
    
    # タイムスタンプの作成
    start_time = datetime(2024, 7, 1, 10, 0, 0)  # 2024年7月1日 10:00:00
    timestamps = [start_time + pd.Timedelta(seconds=i*time_interval) for i in range(num_points)]
    
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
            lats.append(base_lat + progress * 0.03 + 0.002 * math.sin(i/5))
            lons.append(base_lon + progress * 0.01 + 0.005)
        else:
            # 右に向かうタック
            lats.append(base_lat + progress * 0.03 + 0.002 * math.sin(i/5))
            lons.append(base_lon + progress * 0.01 - 0.005)
    
    # 風下レグ
    leg2_points = 90
    for i in range(leg2_points):
        progress = i / leg2_points
        # より直線的な動き
        lats.append(base_lat + 0.03 - progress * 0.03 + 0.001 * math.sin(i/10))
        lons.append(base_lon + 0.01 + 0.002 * math.cos(i/8))
    
    # 2回目の風上レグ
    leg3_points = 90
    for i in range(leg3_points):
        progress = i / leg3_points
        # ジグザグパターン（タック）を追加
        phase = i % 25
        if phase < 12:
            # 左に向かうタック
            lats.append(base_lat + progress * 0.03 + 0.002 * math.sin(i/6))
            lons.append(base_lon - 0.01 + progress * 0.02 + 0.004)
        else:
            # 右に向かうタック
            lats.append(base_lat + progress * 0.03 + 0.002 * math.sin(i/6))
            lons.append(base_lon - 0.01 + progress * 0.02 - 0.004)
    
    # 最終レグ
    leg4_points = 90
    for i in range(leg4_points):
        progress = i / leg4_points
        # フィニッシュに向かう
        lats.append(base_lat + 0.03 - progress * 0.02 + 0.001 * math.sin(i/7))
        lons.append(base_lon + 0.01 - progress * 0.01 + 0.001 * math.cos(i/9))
    
    # データフレーム作成
    data = {
        'timestamp': timestamps,
        'latitude': lats,
        'longitude': lons,
        'elevation': [0] * num_points  # 海面高度は0とする
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
    
    return df

# GPSデータを地図上に表示する関数
def visualize_gps_on_map(gps_data, title="GPS Track Visualization"):
    """GPSデータを地図上に表示する関数"""
    # 地図の中心を計算
    center_lat = gps_data['latitude'].mean()
    center_lon = gps_data['longitude'].mean()
    
    # 地図を作成
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # スタート地点にマーカーを追加
    folium.Marker(
        location=[gps_data['latitude'].iloc[0], gps_data['longitude'].iloc[0]],
        popup='Start',
        icon=folium.Icon(color='green', icon='play'),
    ).add_to(m)
    
    # フィニッシュ地点にマーカーを追加
    folium.Marker(
        location=[gps_data['latitude'].iloc[-1], gps_data['longitude'].iloc[-1]],
        popup='Finish',
        icon=folium.Icon(color='red', icon='stop'),
    ).add_to(m)
    
    # 航跡を追加
    points = list(zip(gps_data['latitude'], gps_data['longitude']))
    folium.PolyLine(
        points,
        color='blue',
        weight=3,
        opacity=0.8
    ).add_to(m)
    
    # タック/ジャイブポイント（大きな方向転換）を検出して表示
    bearing_change = gps_data['bearing'].diff().abs()
    significant_turns = gps_data[bearing_change > 30].copy()  # 30度以上の方向転換を重要な転換点とする
    
    if not significant_turns.empty:
        for i, row in significant_turns.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color='orange',
                fill=True,
                fill_color='orange',
                fill_opacity=0.7,
                popup=f"Time: {row['timestamp']}<br>Bearing: {row['bearing']:.1f}°"
            ).add_to(m)
    
    return m

# 速度と方向の時系列グラフを表示する関数
def plot_speed_and_bearing(gps_data):
    """GPSデータから速度と進行方向の時系列グラフを表示"""
    # 時間軸を作成（分かりやすい形式に）
    time_elapsed = [(t - gps_data['timestamp'].iloc[0]).total_seconds() / 60 for t in gps_data['timestamp']]
    
    # 2つのサブプロットを作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # 速度グラフ（ノットに変換: 1 m/s ≈ 1.94384 ノット）
    speed_knots = gps_data['speed'] * 1.94384
    ax1.plot(time_elapsed, speed_knots, 'b-', linewidth=2)
    ax1.set_ylabel('Speed (knots)')
    ax1.set_title('Boat Speed Over Time')
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

# 風向風速を推定する関数
def estimate_wind_from_tacks(gps_data, min_tack_angle=30, smoothing_window=5):
    """タックパターンから風向を推定する関数"""
    # データのコピーを作成
    df = gps_data.copy()
    
    # 方向の変化を計算（絶対値）
    df['bearing_change'] = df['bearing'].diff().abs()
    
    # 大きな方向変化をタックまたはジャイブとして識別
    df['is_tack'] = df['bearing_change'] > min_tack_angle
    
    # タック/ジャイブポイントを抽出
    tack_points = df[df['is_tack']].copy()
    
    # タック/ジャイブが少なすぎる場合は処理を中止
    if len(tack_points) < 3:
        st.warning("タック/ジャイブポイントが不足しているため、風向の推定が困難です。")
        return None
    
    # 主要な方向を特定するためのヒストグラム分析
    hist, bin_edges = np.histogram(df['bearing'], bins=36, range=(0, 360))
    
    # 上位2つのピークを見つける
    peak_indices = np.argsort(hist)[-2:]
    peak_bins = [(bin_edges[i], bin_edges[i+1]) for i in peak_indices]
    
    # ピークの中心角度を計算
    peak_angles = [(bin_start + bin_end) / 2 for bin_start, bin_end in peak_bins]
    
    # 2つの主要方向から風向を推定
    # 風向は2つの主要タック方向の二等分線の反対方向
    angle_diff = abs(peak_angles[0] - peak_angles[1])
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    # 二等分線を計算
    bisector = min(peak_angles) + angle_diff / 2
    
    # 風向は二等分線の反対（180度反転）
    estimated_wind_direction = (bisector + 180) % 360
    
    # 風速の簡易推定（仮定: 風上での平均艇速の約3倍が風速）
    # より正確な推定には艇の極座標表(polar diagram)が必要
    upwind_segments = []
    downwind_segments = []
    
    # 各タック方向に近い区間を風上/風下セグメントに分類
    for peak_angle in peak_angles:
        # 風上セグメント（タック方向に近い区間）
        upwind_mask = np.abs((df['bearing'] - peak_angle + 180) % 360 - 180) < 30
        upwind_segment = df[upwind_mask]
        if len(upwind_segment) > 0:
            upwind_segments.append(upwind_segment)
        
        # 風下セグメント（タック方向の反対方向に近い区間）
        opposite_angle = (peak_angle + 180) % 360
        downwind_mask = np.abs((df['bearing'] - opposite_angle + 180) % 360 - 180) < 30
        downwind_segment = df[downwind_mask]
        if len(downwind_segment) > 0:
            downwind_segments.append(downwind_segment)
    
    # 風上・風下セグメントの平均速度を計算
    upwind_speeds = [segment['speed'].mean() for segment in upwind_segments if len(segment) > 0]
    downwind_speeds = [segment['speed'].mean() for segment in downwind_segments if len(segment) > 0]
    
    # 平均速度を計算
    avg_upwind_speed = np.mean(upwind_speeds) if upwind_speeds else 0
    avg_downwind_speed = np.mean(downwind_speeds) if downwind_speeds else 0
    
    # 風速推定（メートル/秒）- 単純な経験則ベースの推定
    # 風上では艇速:風速 ≈ 1:3、風下では艇速:風速 ≈ 1:1.5 と仮定
    est_wind_speed_from_upwind = avg_upwind_speed * 3 if avg_upwind_speed > 0 else 0
    est_wind_speed_from_downwind = avg_downwind_speed * 1.5 if avg_downwind_speed > 0 else 0
    
    # 両方の推定値の平均を取る（両方有効な場合）
    if est_wind_speed_from_upwind > 0 and est_wind_speed_from_downwind > 0:
        estimated_wind_speed = (est_wind_speed_from_upwind + est_wind_speed_from_downwind) / 2
    elif est_wind_speed_from_upwind > 0:
        estimated_wind_speed = est_wind_speed_from_upwind
    elif est_wind_speed_from_downwind > 0:
        estimated_wind_speed = est_wind_speed_from_downwind
    else:
        estimated_wind_speed = 0
    
    # ノットに変換（1 m/s ≈ 1.94384 ノット）
    estimated_wind_speed_knots = estimated_wind_speed * 1.94384
    
    # 推定された風向風速を時間窓で平滑化して返す
    # まず各時間帯での推定値を生成
    wind_estimates = pd.DataFrame({
        'timestamp': df['timestamp'],
        'latitude': df['latitude'],
        'longitude': df['longitude'],
        'wind_direction': estimated_wind_direction,
        'wind_speed_knots': estimated_wind_speed_knots,
        'confidence': 0.7  # 単純な固定信頼度
    })
    
    return wind_estimates

# 風向風速を地図上に表示する関数
def visualize_wind_on_map(gps_data, wind_estimates, title="Wind Analysis"):
    """GPSデータと風推定を地図上に表示する関数"""
    # 地図の中心を計算
    center_lat = gps_data['latitude'].mean()
    center_lon = gps_data['longitude'].mean()
    
    # 地図を作成
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # スタート地点にマーカーを追加
    folium.Marker(
        location=[gps_data['latitude'].iloc[0], gps_data['longitude'].iloc[0]],
        popup='Start',
        icon=folium.Icon(color='green', icon='play'),
    ).add_to(m)
    
    # フィニッシュ地点にマーカーを追加
    folium.Marker(
        location=[gps_data['latitude'].iloc[-1], gps_data['longitude'].iloc[-1]],
        popup='Finish',
        icon=folium.Icon(color='red', icon='stop'),
    ).add_to(m)
    
    # 航跡を追加
    points = list(zip(gps_data['latitude'], gps_data['longitude']))
    folium.PolyLine(
        points,
        color='blue',
        weight=3,
        opacity=0.8
    ).add_to(m)
    
    # タック/ジャイブポイント（大きな方向転換）を検出して表示
    bearing_change = gps_data['bearing'].diff().abs()
    significant_turns = gps_data[bearing_change > 30].copy()  # 30度以上の方向転換を重要な転換点とする
    
    if not significant_turns.empty:
        for i, row in significant_turns.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color='orange',
                fill=True,
                fill_color='orange',
                fill_opacity=0.7,
                popup=f"Time: {row['timestamp']}<br>Bearing: {row['bearing']:.1f}°"
            ).add_to(m)
    
    # サンプル間隔（すべてのポイントではなく一部を表示）
    sample_interval = max(1, len(gps_data) // 20)  # 最大20個の風表示
    
    # 風向を矢印で表示
    if wind_estimates is not None:
        wind_dir = wind_estimates['wind_direction'].iloc[0]
        wind_speed = wind_estimates['wind_speed_knots'].iloc[0]
        
        # 風向を矢印として一定間隔で表示
        for i in range(0, len(gps_data), sample_interval):
            # 矢印の向きを風が「吹いてくる」方向に設定
            arrow_direction = wind_dir
            
            # 矢印の長さを風速に比例させる
            arrow_length = 0.002  # スケーリング係数
            
            # 矢印の開始点
            lat = gps_data['latitude'].iloc[i]
            lon = gps_data['longitude'].iloc[i]
            
            # 矢印の終点（三角関数で計算）
            end_lat = lat + arrow_length * math.sin(math.radians(arrow_direction))
            end_lon = lon + arrow_length * math.cos(math.radians(arrow_direction))
            
            # 矢印を地図に追加（風が吹いてくる方向を示す）
            folium.PolyLine(
                [(end_lat, end_lon), (lat, lon)],  # 終点から始点へ
                color='red',
                weight=2,
                opacity=0.7,
                arrow_style='->',
                popup=f"Wind: {wind_dir:.1f}° at {wind_speed:.1f} knots"
            ).add_to(m)
    
    # 風情報を表示するポップアップをコーナーに追加
    if wind_estimates is not None:
        wind_dir = wind_estimates['wind_direction'].iloc[0]
        wind_speed = wind_estimates['wind_speed_knots'].iloc[0]
        
        wind_info_html = f"""
        <div style="background-color: white; padding: 10px; border-radius: 5px; opacity: 0.8;">
            <h4>推定風情報</h4>
            <p>風向: {wind_dir:.1f}° ({degrees_to_cardinal(wind_dir)})</p>
            <p>風速: {wind_speed:.1f} ノット</p>
        </div>
        """
        
        folium.map.Marker(
            [gps_data['latitude'].min(), gps_data['longitude'].min()],
            icon=folium.DivIcon(
                icon_size=(150,100),
                icon_anchor=(0,0),
                html=wind_info_html
            )
        ).add_to(m)
    
    return m

# サイドバーにファイルアップロード機能
st.sidebar.header('データ入力')
uploaded_file = st.sidebar.file_uploader("GPX/CSVファイルをアップロード", type=['gpx', 'csv'])

# サンプルデータの使用オプション
use_sample_data = st.sidebar.checkbox('サンプルデータを使用', value=True)

# データ処理
gps_data = None

if uploaded_file is not None:
    # アップロードされたファイルの処理
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'gpx':
            gps_data = load_gpx_to_dataframe(uploaded_file.getvalue().decode('utf-8'))
        elif file_extension == 'csv':
            gps_data = load_csv_to_dataframe(uploaded_file.getvalue())
        
        if gps_data is not None:
            st.sidebar.success(f"ファイル '{uploaded_file.name}' を正常に読み込みました")
        else:
            st.sidebar.error("ファイルの読み込みに失敗しました。")
    except Exception as e:
        st.sidebar.error(f"ファイルの読み込みエラー: {e}")
elif use_sample_data:
    # サンプルデータの生成
    gps_data = generate_sample_gps_data()
    st.sidebar.info("サンプルデータを使用しています")

# 分析オプション
if gps_data is not None:
    st.sidebar.header('分析オプション')
    
    # タック/ジャイブ検出の閾値
    tack_threshold = st.sidebar.slider(
        'タック/ジャイブ検出閾値（度）', 
        min_value=10, 
        max_value=60, 
        value=30,
        help='この角度以上の方向転換をタック/ジャイブとして検出します'
    )
    
    # 分析の実行
    st.header('データ分析')
    
    # 基本情報を表示
    st.subheader('基本情報')
    duration = (gps_data['timestamp'].iloc[-1] - gps_data['timestamp'].iloc[0]).total_seconds() / 60
    distance = gps_data['distance'].sum() / 1000  # キロメートルに変換
    avg_speed = (distance * 1000) / (duration * 60) * 1.94384  # ノットに変換
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("航走時間", f"{duration:.1f} 分")
    with col2:
        st.metric("総距離", f"{distance:.2f} km")
    with col3:
        st.metric("平均速度", f"{avg_speed:.2f} ノット")
    
    # GPSトラックの表示
    st.subheader('航跡地図')
    track_map = visualize_gps_on_map(gps_data, "GPS Track")
    folium_static(track_map)
    
    # 速度と方向のグラフ
    st.subheader('速度と進行方向')
    speed_bearing_fig = plot_speed_and_bearing(gps_data)
    st.pyplot(speed_bearing_fig)
    
    # 風向風速の推定
    st.header('風向風速推定')
    wind_estimates = estimate_wind_from_tacks(gps_data, min_tack_angle=tack_threshold)
    
    if wind_estimates is not None:
        wind_dir = wind_estimates['wind_direction'].iloc[0]
        wind_speed = wind_estimates['wind_speed_knots'].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("推定風向", f"{wind_dir:.1f}° ({degrees_to_cardinal(wind_dir)})")
        with col2:
            st.metric("推定風速", f"{wind_speed:.1f} ノット")
        
        # 風向風速の地図表示
        st.subheader('風向風速マップ')
        wind_map = visualize_wind_on_map(gps_data, wind_estimates, "Wind Analysis")
        folium_static(wind_map)
        
        # 主要な帆走方向のヒストグラム
        st.subheader('主要な帆走方向')
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(gps_data['bearing'], bins=36, range=(0, 360), alpha=0.7)
        ax.axvline(x=wind_dir, color='r', linestyle='--', linewidth=2)
        ax.text(wind_dir+5, plt.ylim()[1]*0.9, f"推定風向: {wind_dir:.1f}°", color='r')
        ax.set_xlabel('方向 (度)')
        ax.set_ylabel('頻度')
        ax.set_title('帆走方向のヒストグラム')
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xticklabels(['N', 'E', 'S', 'W', 'N'])
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # 戦略分析
        st.header('戦略分析')
        st.write("タック/ジャイブポイントの分析")
        
        # 方向転換ポイントの検出
        bearing_change = gps_data['bearing'].diff().abs()
        significant_turns = gps_data[bearing_change > tack_threshold].copy()
        
        if not significant_turns.empty:
            # タック/ジャイブの表形式での表示
            st.subheader('重要な方向転換ポイント')
            turn_data = significant_turns[['timestamp', 'latitude', 'longitude', 'bearing', 'speed']].copy()
            turn_data['speed_knots'] = turn_data['speed'] * 1.94384
            turn_data = turn_data.rename(columns={
                'timestamp': '時間', 
                'latitude': '緯度', 
                'longitude': '経度',
                'bearing': '方向（度）',
                'speed_knots': '速度（ノット）'
            })
            st.dataframe(turn_data)
        
        # ダウンロードリンク
        st.header('データダウンロード')
        
        # 分析結果のCSV
        csv = gps_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="sailing_analysis.csv">分析データをCSV形式でダウンロード</a>'
        st.markdown(href, unsafe_allow_html=True)
    
else:
    st.info('GPSデータをアップロードするか、サンプルデータを使用してください。')

# フッター
st.markdown("---")
st.markdown("© 2024 セーリング戦略分析システム | Powered by Streamlit")

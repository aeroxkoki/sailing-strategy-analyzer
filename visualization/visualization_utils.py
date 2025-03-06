"""
セーリング戦略分析システム - 可視化ユーティリティモジュール

このモジュールは、可視化モジュール全体で使用される共通のユーティリティ関数を提供します。
データ変換、カラーマッピング、フォーマット変換などの機能を含みます。

作成日: 2025-03-05
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import folium
import colorsys
import math
import os
import json


def convert_degrees_to_cardinal(degrees):
    """
    度数を16方位（北、北北東、北東...）に変換します
    
    Parameters:
    -----------
    degrees : float
        方位角（度数）
        
    Returns:
    --------
    str
        対応する方位名（日本語）
    """
    dirs = [
        "北", "北北東", "北東", "東北東",
        "東", "東南東", "南東", "南南東",
        "南", "南南西", "南西", "西南西",
        "西", "西北西", "北西", "北北西"
    ]
    
    # -22.5〜+22.5を「北」にするため、+22.5度シフト
    adjusted = (degrees + 22.5) % 360
    index = int(adjusted // 22.5)
    
    return dirs[index]


def generate_distinct_colors(n):
    """
    視覚的に区別しやすいn個の色を生成します
    
    Parameters:
    -----------
    n : int
        必要な色の数
        
    Returns:
    --------
    list
        RGB16進数形式の色コードのリスト（例: '#FF0000'）
    """
    colors = []
    for i in range(n):
        # HSVカラーモデルで均等に分布させる
        h = i / n
        s = 0.7  # 彩度
        v = 0.9  # 明度
        
        # HSV → RGB変換
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # RGB値を0-255の範囲に変換し、16進数形式にする
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        colors.append(hex_color)
    
    return colors


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    2点間の距離をハバーサイン公式で計算します（単位: メートル）
    
    Parameters:
    -----------
    lat1, lon1 : float
        1点目の緯度・経度（度数）
    lat2, lon2 : float
        2点目の緯度・経度（度数）
        
    Returns:
    --------
    float
        2点間の距離（メートル）
    """
    # 地球の半径（メートル）
    R = 6371000
    
    # 度数からラジアンに変換
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # 差分
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # ハバーサイン公式
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    2点間の方位角を計算します（北を0度とした時計回りの角度）
    
    Parameters:
    -----------
    lat1, lon1 : float
        1点目の緯度・経度（度数）
    lat2, lon2 : float
        2点目の緯度・経度（度数）
        
    Returns:
    --------
    float
        方位角（度数、0-360）
    """
    # 度数からラジアンに変換
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # 経度の差分
    dlon = lon2_rad - lon1_rad
    
    # 方位角の計算
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    bearing_rad = math.atan2(y, x)
    
    # ラジアンから度数に変換（0-360度）
    bearing = (math.degrees(bearing_rad) + 360) % 360
    
    return bearing


def calculate_vmg(speed, course, wind_direction):
    """
    VMG（Velocity Made Good）を計算します
    風に向かってどれだけ進んでいるかを示す値
    
    Parameters:
    -----------
    speed : float
        ボートの速度
    course : float
        ボートの進行方向（度数、0-360）
    wind_direction : float
        風向（度数、0-360）
        
    Returns:
    --------
    float
        VMG値（正の値は風上に向かっている、負の値は風下に向かっている）
    """
    # 相対的な角度（ボートの進行方向と風向の差）
    angle_diff_rad = math.radians((wind_direction - course) % 360)
    
    # VMGの計算
    vmg = speed * math.cos(angle_diff_rad)
    
    return vmg


def create_arrow_marker(lat, lon, direction, color='blue', size=10):
    """
    矢印のマーカーをFoliumで作成します
    
    Parameters:
    -----------
    lat, lon : float
        矢印の位置（緯度・経度）
    direction : float
        矢印の方向（度数、0-360）
    color : str, optional
        矢印の色
    size : int, optional
        矢印のサイズ
        
    Returns:
    --------
    folium.RegularPolygonMarker
        矢印マーカー
    """
    return folium.RegularPolygonMarker(
        location=(lat, lon),
        number_of_sides=3,
        rotation=direction,
        fill_color=color,
        fill_opacity=0.8,
        stroke=False,
        radius=size
    )


def smooth_data(data, column, window_size=5):
    """
    移動平均によるデータのスムージング
    
    Parameters:
    -----------
    data : pandas.DataFrame
        スムージングするデータフレーム
    column : str
        スムージングする列の名前
    window_size : int, optional
        移動平均の窓サイズ
        
    Returns:
    --------
    pandas.Series
        スムージングされたデータ
    """
    return data[column].rolling(window=window_size, center=True).mean()


def detect_tacks(data, course_column='course', threshold=80):
    """
    タック（方向転換）を検出します
    
    Parameters:
    -----------
    data : pandas.DataFrame
        ボートのデータフレーム
    course_column : str, optional
        コース（進行方向）の列名
    threshold : float, optional
        タックと判定する角度変化の閾値
        
    Returns:
    --------
    pandas.DataFrame
        タックが発生した位置と時間のデータフレーム
    """
    # コース変化の計算
    data = data.copy()
    data['course_diff'] = data[course_column].diff().abs()
    
    # 大きな変化（タック）の検出
    tacks = data[data['course_diff'] > threshold].copy()
    
    return tacks


def calculate_optimal_vmg_angles(polar_data, wind_speed_column='wind_speed', 
                                boat_speed_column='speed', wind_angle_column='wind_angle'):
    """
    最適なVMG角度を計算します（風上および風下）
    
    Parameters:
    -----------
    polar_data : pandas.DataFrame
        ポーラーデータを含むデータフレーム
    wind_speed_column : str, optional
        風速の列名
    boat_speed_column : str, optional
        ボート速度の列名
    wind_angle_column : str, optional
        風向角の列名
        
    Returns:
    --------
    dict
        風速ごとの最適VMG角度（風上および風下）
    """
    result = {}
    
    # 風速の一意な値をループ
    for wind_speed in polar_data[wind_speed_column].unique():
        # この風速のデータをフィルタリング
        wind_data = polar_data[polar_data[wind_speed_column] == wind_speed]
        
        # VMGの計算（風上）
        wind_data['vmg_upwind'] = wind_data[boat_speed_column] * np.cos(np.radians(wind_data[wind_angle_column]))
        
        # VMGの計算（風下）
        wind_data['vmg_downwind'] = wind_data[boat_speed_column] * np.cos(np.radians(wind_data[wind_angle_column] - 180))
        
        # 最適角度の特定
        best_upwind_idx = wind_data['vmg_upwind'].idxmax()
        best_downwind_idx = wind_data['vmg_downwind'].idxmax()
        
        # 結果の格納
        result[wind_speed] = {
            'upwind': {
                'angle': wind_data.loc[best_upwind_idx, wind_angle_column],
                'speed': wind_data.loc[best_upwind_idx, boat_speed_column],
                'vmg': wind_data.loc[best_upwind_idx, 'vmg_upwind']
            },
            'downwind': {
                'angle': wind_data.loc[best_downwind_idx, wind_angle_column],
                'speed': wind_data.loc[best_downwind_idx, boat_speed_column],
                'vmg': wind_data.loc[best_downwind_idx, 'vmg_downwind']
            }
        }
    
    return result


def resample_time_series(data, timestamp_column='timestamp', freq='1S'):
    """
    時系列データを一定間隔で再サンプリングします
    
    Parameters:
    -----------
    data : pandas.DataFrame
        時系列データを含むデータフレーム
    timestamp_column : str, optional
        タイムスタンプの列名
    freq : str, optional
        再サンプリングの頻度（例: '1S'は1秒ごと）
        
    Returns:
    --------
    pandas.DataFrame
        再サンプリングされたデータフレーム
    """
    # インデックスをタイムスタンプに設定
    data_copy = data.copy()
    data_copy = data_copy.set_index(timestamp_column)
    
    # 再サンプリング（線形補間）
    resampled = data_copy.resample(freq).interpolate(method='linear')
    
    # インデックスを列に戻す
    resampled = resampled.reset_index()
    
    return resampled


def interpolate_position(lat1, lon1, lat2, lon2, ratio):
    """
    2点間の位置を線形補間します
    
    Parameters:
    -----------
    lat1, lon1 : float
        1点目の緯度・経度
    lat2, lon2 : float
        2点目の緯度・経度
    ratio : float
        補間位置（0.0から1.0の間、0.0は1点目、1.0は2点目）
        
    Returns:
    --------
    tuple
        補間された位置（緯度, 経度）
    """
    lat = lat1 + (lat2 - lat1) * ratio
    lon = lon1 + (lon2 - lon1) * ratio
    return (lat, lon)


def synchronize_boat_data(boats_data, reference_times):
    """
    複数のボートデータを共通の時間軸で同期します
    
    Parameters:
    -----------
    boats_data : dict
        ボート名をキー、DataFrameを値とする辞書
    reference_times : list
        同期する基準時間のリスト
        
    Returns:
    --------
    dict
        同期されたデータを含む辞書
    """
    synchronized_data = {}
    
    for boat_name, df in boats_data.items():
        # タイムスタンプ列の存在確認
        if 'timestamp' not in df.columns:
            print(f"警告: {boat_name} のデータにはタイムスタンプ列がありません。スキップします。")
            continue
        
        # 同期されたデータを格納するための新しいDataFrame
        synced_df = pd.DataFrame()
        synced_df['timestamp'] = reference_times
        
        # すべての列について補間
        for column in df.columns:
            if column == 'timestamp':
                continue
            
            # 各参照時間について最も近い値または補間値を取得
            values = []
            for ref_time in reference_times:
                # 最も近い前後の時間を見つける
                before = df[df['timestamp'] <= ref_time]
                after = df[df['timestamp'] >= ref_time]
                
                if before.empty and after.empty:
                    # データがない場合
                    values.append(None)
                elif before.empty:
                    # 開始前の場合は最初の値を使用
                    values.append(after.iloc[0][column])
                elif after.empty:
                    # 終了後の場合は最後の値を使用
                    values.append(before.iloc[-1][column])
                elif ref_time in df['timestamp'].values:
                    # 完全に一致する時間がある場合
                    values.append(df[df['timestamp'] == ref_time][column].values[0])
                else:
                    # 線形補間が必要な場合
                    before_row = before.iloc[-1]
                    after_row = after.iloc[0]
                    before_time = before_row['timestamp']
                    after_time = after_row['timestamp']
                    
                    # 時間の差分を計算
                    time_diff = (after_time - before_time).total_seconds()
                    if time_diff == 0:
                        # 同じ時間の場合（不正なデータ）
                        values.append(before_row[column])
                    else:
                        # 時間に基づいて線形補間
                        ratio = (ref_time - before_time).total_seconds() / time_diff
                        interpolated = before_row[column] + ratio * (after_row[column] - before_row[column])
                        values.append(interpolated)
            
            # 列に補間値を追加
            synced_df[column] = values
        
        synchronized_data[boat_name] = synced_df
    
    return synchronized_data


def export_to_geojson(data, lat_column='latitude', lon_column='longitude', 
                    properties=None, output_file='track.geojson'):
    """
    DataFrameからGeoJSONファイルを作成します
    
    Parameters:
    -----------
    data : pandas.DataFrame
        GeoJSONに変換するデータフレーム
    lat_column : str, optional
        緯度の列名
    lon_column : str, optional
        経度の列名
    properties : list, optional
        GeoJSONのプロパティとして含める列名のリスト
    output_file : str, optional
        出力ファイル名
        
    Returns:
    --------
    bool
        エクスポートが成功したかどうか
    """
    if lat_column not in data.columns or lon_column not in data.columns:
        print(f"エラー: データには '{lat_column}' と '{lon_column}' 列が必要です")
        return False
    
    if properties is None:
        # デフォルトでは緯度・経度以外のすべての列をプロパティに含める
        properties = [col for col in data.columns if col not in [lat_column, lon_column]]
    
    # GeoJSONの特徴を作成
    features = []
    for _, row in data.iterrows():
        # プロパティの抽出
        props = {}
        for prop in properties:
            if prop in row:
                # NumPy型をPythonネイティブ型に変換（JSON互換性のため）
                value = row[prop]
                if isinstance(value, np.integer):
                    value = int(value)
                elif isinstance(value, np.floating):
                    value = float(value)
                elif isinstance(value, np.ndarray):
                    value = value.tolist()
                elif isinstance(value, datetime):
                    value = value.isoformat()
                
                props[prop] = value
        
        # 特徴の作成
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row[lon_column]), float(row[lat_column])]
            },
            "properties": props
        }
        features.append(feature)
    
    # GeoJSONオブジェクトの作成
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # ファイルへの書き込み
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"GeoJSONファイルの書き込みエラー: {e}")
        return False


def export_to_kml(data, lat_column='latitude', lon_column='longitude', 
                name_column=None, description_column=None, output_file='track.kml'):
    """
    DataFrameからKMLファイルを作成します
    
    Parameters:
    -----------
    data : pandas.DataFrame
        KMLに変換するデータフレーム
    lat_column : str, optional
        緯度の列名
    lon_column : str, optional
        経度の列名
    name_column : str, optional
        名前として使用する列名
    description_column : str, optional
        説明として使用する列名
    output_file : str, optional
        出力ファイル名
        
    Returns:
    --------
    bool
        エクスポートが成功したかどうか
    """
    try:
        # シンプルなKMLファイルを作成
        kml_header = '<?xml version="1.0" encoding="UTF-8"?>\n'
        kml_header += '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        kml_header += '<Document>\n'
        kml_header += '  <name>セーリング航跡</name>\n'
        kml_header += '  <Style id="sailingTrack">\n'
        kml_header += '    <LineStyle>\n'
        kml_header += '      <color>ff0000ff</color>\n'  # 赤色
        kml_header += '      <width>3</width>\n'
        kml_header += '    </LineStyle>\n'
        kml_header += '  </Style>\n'
        
        # プレースマークとラインストリングの開始
        kml_content = '  <Placemark>\n'
        if name_column and name_column in data.columns:
            kml_content += f'    <name>{data[name_column].iloc[0]}</name>\n'
        else:
            kml_content += '    <name>セーリング航跡</name>\n'
        
        if description_column and description_column in data.columns:
            kml_content += f'    <description>{data[description_column].iloc[0]}</description>\n'
        
        kml_content += '    <styleUrl>#sailingTrack</styleUrl>\n'
        kml_content += '    <LineString>\n'
        kml_content += '      <coordinates>\n'
        
        # 座標の追加
        for _, row in data.iterrows():
            kml_content += f'        {row[lon_column]},{row[lat_column]},0\n'
        
        # プレースマークとラインストリングの終了
        kml_content += '      </coordinates>\n'
        kml_content += '    </LineString>\n'
        kml_content += '  </Placemark>\n'
        
        # KMLの終了
        kml_footer = '</Document>\n'
        kml_footer += '</kml>'
        
        # ファイルへの書き込み
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(kml_header + kml_content + kml_footer)
        
        return True
        
    except Exception as e:
        print(f"KMLファイルの書き込みエラー: {e}")
        return False


def calculate_statistics(data, speed_column='speed', course_column='course', 
                        wind_direction_column='wind_direction'):
    """
    セーリングデータの基本統計を計算します
    
    Parameters:
    -----------
    data : pandas.DataFrame
        統計を計算するデータフレーム
    speed_column : str, optional
        速度の列名
    course_column : str, optional
        コース（進行方向）の列名
    wind_direction_column : str, optional
        風向の列名
        
    Returns:
    --------
    dict
        計算された統計値
    """
    stats = {}
    
    # 速度の統計
    if speed_column in data.columns:
        speed_data = data[speed_column].dropna()
        if not speed_data.empty:
            stats['speed'] = {
                'min': speed_data.min(),
                'max': speed_data.max(),
                'avg': speed_data.mean(),
                'median': speed_data.median(),
                'std': speed_data.std()
            }
    
    # 方向の統計（円環データなので平均の計算に注意）
    if course_column in data.columns:
        course_data = data[course_column].dropna()
        if not course_data.empty:
            # 度数からラジアンに変換
            course_rad = np.radians(course_data)
            
            # 方向の平均を計算
            sin_sum = np.sum(np.sin(course_rad))
            cos_sum = np.sum(np.cos(course_rad))
            avg_angle_rad = np.arctan2(sin_sum, cos_sum)
            avg_angle_deg = (np.degrees(avg_angle_rad) + 360) % 360
            
            stats['course'] = {
                'avg_direction': avg_angle_deg,
                'avg_direction_cardinal': convert_degrees_to_cardinal(avg_angle_deg)
            }
    
    # 風向の統計
    if wind_direction_column in data.columns:
        wind_data = data[wind_direction_column].dropna()
        if not wind_data.empty:
            # 度数からラジアンに変換
            wind_rad = np.radians(wind_data)
            
            # 方向の平均を計算
            sin_sum = np.sum(np.sin(wind_rad))
            cos_sum = np.sum(np.cos(wind_rad))
            avg_angle_rad = np.arctan2(sin_sum, cos_sum)
            avg_angle_deg = (np.degrees(avg_angle_rad) + 360) % 360
            
            stats['wind'] = {
                'avg_direction': avg_angle_deg,
                'avg_direction_cardinal': convert_degrees_to_cardinal(avg_angle_deg)
            }
    
    # 総距離の計算（隣接点間の距離の合計）
    if 'latitude' in data.columns and 'longitude' in data.columns:
        total_distance = 0
        for i in range(1, len(data)):
            lat1 = data['latitude'].iloc[i-1]
            lon1 = data['longitude'].iloc[i-1]
            lat2 = data['latitude'].iloc[i]
            lon2 = data['longitude'].iloc[i]
            
            distance = calculate_distance(lat1, lon1, lat2, lon2)
            total_distance += distance
        
        stats['total_distance_m'] = total_distance
        stats['total_distance_nm'] = total_distance / 1852  # メートルから海里に変換
    
    # 所要時間の計算
    if 'timestamp' in data.columns:
        start_time = data['timestamp'].min()
        end_time = data['timestamp'].max()
        duration = end_time - start_time
        
        stats['duration_seconds'] = duration.total_seconds()
        stats['duration_formatted'] = str(duration)
    
    return stats

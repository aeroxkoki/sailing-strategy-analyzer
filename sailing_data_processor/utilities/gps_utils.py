"""
GPS関連ユーティリティ - セーリングデータ処理用
"""
import math
import numpy as np
from typing import Tuple, List, Optional, Union


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    ハバーサイン公式を使用して2点間の距離を計算します
    
    Parameters:
    -----------
    lat1, lon1 : float
        1点目の緯度・経度（度数法）
    lat2, lon2 : float
        2点目の緯度・経度（度数法）
        
    Returns:
    --------
    float
        2点間の距離（メートル）
    """
    # 地球の半径（メートル）
    R = 6371000.0
    
    # 度数法からラジアンに変換
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


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    2点間の方位角（真北を0°として時計回り）を計算します
    
    Parameters:
    -----------
    lat1, lon1 : float
        始点の緯度・経度（度数法）
    lat2, lon2 : float
        終点の緯度・経度（度数法）
        
    Returns:
    --------
    float
        方位角（0-360度）
    """
    # 度数法からラジアンに変換
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # 方位角の計算
    y = math.sin(lon2_rad - lon1_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)
    bearing_rad = math.atan2(y, x)
    
    # ラジアンから度数法に変換し、0-360の範囲に調整
    bearing = (math.degrees(bearing_rad) + 360) % 360
    
    return bearing


def project_position(lat: float, lon: float, bearing: float, distance: float) -> Tuple[float, float]:
    """
    始点、方位角、距離から終点の座標を計算します
    
    Parameters:
    -----------
    lat, lon : float
        始点の緯度・経度（度数法）
    bearing : float
        方位角（度数法、0-360）
    distance : float
        距離（メートル）
        
    Returns:
    --------
    Tuple[float, float]
        終点の緯度・経度（度数法）
    """
    # 地球の半径（メートル）
    R = 6371000.0
    
    # 度数法からラジアンに変換
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing)
    
    # 角度距離（ラジアン）
    angular_distance = distance / R
    
    # 終点の緯度計算
    lat2_rad = math.asin(
        math.sin(lat_rad) * math.cos(angular_distance) + 
        math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
    )
    
    # 終点の経度計算
    lon2_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
        math.cos(angular_distance) - math.sin(lat_rad) * math.sin(lat2_rad)
    )
    
    # ラジアンから度数法に変換
    lat2 = math.degrees(lat2_rad)
    lon2 = math.degrees(lon2_rad)
    
    return lat2, lon2


def detect_tack_points(latitudes: List[float], longitudes: List[float], 
                     bearings: List[float], min_angle: float = 30.0) -> List[int]:
    """
    GPSトラックからタックポイントを検出します
    
    Parameters:
    -----------
    latitudes : List[float]
        緯度のリスト
    longitudes : List[float]
        経度のリスト
    bearings : List[float]
        方位角のリスト
    min_angle : float
        タックと見なす最小角度変化
        
    Returns:
    --------
    List[int]
        タックポイントのインデックスリスト
    """
    if len(bearings) < 3:
        return []
    
    tack_points = []
    
    # 方位角の変化を計算
    for i in range(1, len(bearings) - 1):
        # 前後の方位角の差分（循環を考慮）
        diff_prev = abs((bearings[i] - bearings[i-1] + 180) % 360 - 180)
        diff_next = abs((bearings[i+1] - bearings[i] + 180) % 360 - 180)
        
        # 大きな方位角変化がある場合はタックポイントと判定
        if diff_prev > min_angle or diff_next > min_angle:
            tack_points.append(i)
    
    return tack_points


def identify_upwind_downwind(latitudes: List[float], longitudes: List[float], 
                           speeds: List[float], bearings: List[float]) -> Tuple[List[int], List[int]]:
    """
    GPSトラックから風上レグと風下レグを識別します
    
    Parameters:
    -----------
    latitudes : List[float]
        緯度のリスト
    longitudes : List[float]
        経度のリスト
    speeds : List[float]
        速度のリスト（メートル/秒）
    bearings : List[float]
        方位角のリスト
        
    Returns:
    --------
    Tuple[List[int], List[int]]
        (風上レグのインデックスリスト, 風下レグのインデックスリスト)
    """
    from sklearn.cluster import KMeans
    
    if len(speeds) < 10 or len(bearings) < 10:
        return [], []
    
    # 速度と方位角のデータポイントを準備
    X = np.column_stack([
        np.array(speeds),
        np.sin(np.radians(bearings)),
        np.cos(np.radians(bearings))
    ])
    
    # KMeansクラスタリングで2つのレグに分類
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    labels = kmeans.labels_
    
    # クラスタ中心を取得
    cluster_centers = kmeans.cluster_centers_
    
    # 平均速度が遅いクラスタを風上レグとする
    if cluster_centers[0][0] < cluster_centers[1][0]:
        upwind_cluster = 0
        downwind_cluster = 1
    else:
        upwind_cluster = 1
        downwind_cluster = 0
    
    # 各レグのインデックスを抽出
    upwind_indices = [i for i, label in enumerate(labels) if label == upwind_cluster]
    downwind_indices = [i for i, label in enumerate(labels) if label == downwind_cluster]
    
    return upwind_indices, downwind_indices


def interpolate_gps_track(latitudes: List[float], longitudes: List[float], num_points: int) -> Tuple[List[float], List[float]]:
    """
    GPSトラックを補間して指定された数のポイントを生成します
    
    Parameters:
    -----------
    latitudes : List[float]
        元の緯度のリスト
    longitudes : List[float]
        元の経度のリスト
    num_points : int
        補間後のポイント数
        
    Returns:
    --------
    Tuple[List[float], List[float]]
        補間された(緯度リスト, 経度リスト)
    """
    if len(latitudes) < 2 or len(longitudes) < 2:
        return latitudes, longitudes
    
    # 元のインデックス（0から1の範囲に正規化）
    indices = np.linspace(0, 1, len(latitudes))
    
    # 新しいインデックス
    new_indices = np.linspace(0, 1, num_points)
    
    # 補間
    interp_lats = np.interp(new_indices, indices, latitudes)
    interp_lons = np.interp(new_indices, indices, longitudes)
    
    return interp_lats.tolist(), interp_lons.tolist()


def filter_gps_noise(latitudes: List[float], longitudes: List[float], 
                   timestamps: List[float], max_speed: float = 15.0) -> Tuple[List[float], List[float], List[float]]:
    """
    GPSノイズを除去します（異常な速度のポイントを検出・除去）
    
    Parameters:
    -----------
    latitudes : List[float]
        緯度のリスト
    longitudes : List[float]
        経度のリスト
    timestamps : List[float]
        タイムスタンプのリスト（秒単位）
    max_speed : float
        最大想定速度（メートル/秒）
        
    Returns:
    --------
    Tuple[List[float], List[float], List[float]]
        フィルタリングされた(緯度リスト, 経度リスト, タイムスタンプリスト)
    """
    if len(latitudes) < 3:
        return latitudes, longitudes, timestamps
    
    filtered_lats = [latitudes[0]]
    filtered_lons = [longitudes[0]]
    filtered_times = [timestamps[0]]
    
    for i in range(1, len(latitudes)):
        # 前のポイントからの距離を計算
        distance = haversine_distance(
            filtered_lats[-1], filtered_lons[-1],
            latitudes[i], longitudes[i]
        )
        
        # 時間差を計算
        time_diff = timestamps[i] - filtered_times[-1]
        
        # ゼロ除算を防ぐ
        if time_diff <= 0:
            continue
        
        # 速度を計算
        speed = distance / time_diff
        
        # 最大速度以下なら有効なポイントとして追加
        if speed <= max_speed:
            filtered_lats.append(latitudes[i])
            filtered_lons.append(longitudes[i])
            filtered_times.append(timestamps[i])
    
    return filtered_lats, filtered_lons, filtered_times

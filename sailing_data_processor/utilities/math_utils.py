"""
数学関連ユーティリティ - セーリングデータ処理用

角度計算、統計処理、補間などの数学的ユーティリティ関数を提供します。
"""
import math
import numpy as np
from typing import List, Tuple, Optional, Union, Any


def normalize_angle(angle: float) -> float:
    """
    角度を0-360度の範囲に正規化します
    
    Parameters:
    -----------
    angle : float
        正規化する角度（度数法）
        
    Returns:
    --------
    float
        0-360度の範囲に正規化された角度
    """
    return angle % 360


def angle_difference(angle1: float, angle2: float) -> float:
    """
    2つの角度間の最小差分を計算します（-180〜180度の範囲）
    
    Parameters:
    -----------
    angle1, angle2 : float
        比較する角度（度数法）
        
    Returns:
    --------
    float
        角度の差分（-180〜180度）
    """
    return ((angle1 - angle2 + 180) % 360) - 180


def average_angle(angles: List[float], weights: Optional[List[float]] = None) -> float:
    """
    角度の平均を計算します（循環を考慮）
    
    Parameters:
    -----------
    angles : List[float]
        平均する角度のリスト（度数法）
    weights : List[float], optional
        各角度の重み
        
    Returns:
    --------
    float
        角度の平均（0-360度）
    """
    if not angles:
        return 0.0
    
    # 単位ベクトルに変換
    sin_sum = 0.0
    cos_sum = 0.0
    
    if weights is None:
        weights = [1.0] * len(angles)
    
    # 重み付きの合計
    for angle, weight in zip(angles, weights):
        angle_rad = math.radians(angle)
        sin_sum += math.sin(angle_rad) * weight
        cos_sum += math.cos(angle_rad) * weight
    
    # アークタンジェントで平均角度を計算
    avg_angle_rad = math.atan2(sin_sum, cos_sum)
    
    # ラジアンから度数法に変換し、0-360度の範囲に調整
    avg_angle = (math.degrees(avg_angle_rad) + 360) % 360
    
    return avg_angle


def angle_dispersion(angles: List[float]) -> float:
    """
    角度の分散を計算します（循環を考慮）
    
    Parameters:
    -----------
    angles : List[float]
        分散を計算する角度のリスト（度数法）
        
    Returns:
    --------
    float
        角度の分散（0-1の範囲、0は完全に整列、1は完全にランダム）
    """
    if not angles or len(angles) < 2:
        return 0.0
    
    # 単位ベクトルに変換
    sin_vals = [math.sin(math.radians(angle)) for angle in angles]
    cos_vals = [math.cos(math.radians(angle)) for angle in angles]
    
    # 平均ベクトルの長さを計算
    sin_mean = sum(sin_vals) / len(sin_vals)
    cos_mean = sum(cos_vals) / len(cos_vals)
    r = math.sqrt(sin_mean**2 + cos_mean**2)
    
    # r = 1 は完全に整列、r = 0 は完全にランダム
    # 1 - r で分散を返す（0が整列、1がランダム）
    return 1.0 - r


def windward_efficiency(boat_speed: float, wind_speed: float, angle: float, 
                     boat_type: str = 'default') -> float:
    """
    風上効率（風速に対するVMGの比率）を計算します
    
    Parameters:
    -----------
    boat_speed : float
        艇速（メートル/秒）
    wind_speed : float
        風速（メートル/秒）
    angle : float
        艇の進行方向と風向の相対角度（度数法）
    boat_type : str
        艇種識別子
        
    Returns:
    --------
    float
        風上効率（0-1の範囲）
    """
    # 相対角度をラジアンに変換
    angle_rad = math.radians(angle)
    
    # Velocity Made Good (VMG) の計算
    vmg = boat_speed * math.cos(angle_rad)
    
    # 風速に対する比率（理論上の最大値を1とする）
    efficiency = vmg / wind_speed
    
    # 艇種ごとの補正係数（簡易実装）
    boat_coefficients = {
        'default': 0.4,
        'laser': 0.42,
        'ilca': 0.42,
        '470': 0.45,
        '49er': 0.38,
        'finn': 0.44,
        'nacra17': 0.35,
        'star': 0.48
    }
    
    # 補正係数で正規化
    coefficient = boat_coefficients.get(boat_type.lower(), boat_coefficients['default'])
    normalized_efficiency = efficiency / coefficient
    
    # 0-1の範囲に制限
    return max(0.0, min(1.0, normalized_efficiency))


def interpolate_wind_field(lat_points: List[float], lon_points: List[float], 
                        wind_dirs: List[float], wind_speeds: List[float],
                        grid_lat: np.ndarray, grid_lon: np.ndarray, 
                        method: str = 'rbf') -> Tuple[np.ndarray, np.ndarray]:
    """
    風向風速の空間補間を行います
    
    Parameters:
    -----------
    lat_points, lon_points : List[float]
        観測点の緯度・経度
    wind_dirs, wind_speeds : List[float]
        各観測点の風向・風速
    grid_lat, grid_lon : np.ndarray
        補間先のグリッド座標
    method : str
        補間方法 ('rbf', 'idw', 'linear')
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        補間された風向・風速のグリッド
    """
    from scipy.interpolate import Rbf, LinearNDInterpolator
    
    # 入力データの検証
    if len(lat_points) < 3 or len(lon_points) < 3:
        raise ValueError("補間には少なくとも3つの観測点が必要です")
    
    # 観測点の座標
    points = np.column_stack([lat_points, lon_points])
    
    # 風向を sin/cos 成分に分解
    wind_dir_sin = np.sin(np.radians(wind_dirs))
    wind_dir_cos = np.cos(np.radians(wind_dirs))
    
    # グリッドポイント
    grid_points = np.column_stack([grid_lat.flatten(), grid_lon.flatten()])
    
    # 補間方法の選択
    if method == 'rbf':
        # Radial Basis Function 補間
        rbf_sin = Rbf(points[:, 0], points[:, 1], wind_dir_sin, function='multiquadric')
        rbf_cos = Rbf(points[:, 0], points[:, 1], wind_dir_cos, function='multiquadric')
        rbf_speed = Rbf(points[:, 0], points[:, 1], wind_speeds, function='multiquadric')
        
        # 補間の実行
        interp_sin = rbf_sin(grid_lat, grid_lon)
        interp_cos = rbf_cos(grid_lat, grid_lon)
        interp_speed = rbf_speed(grid_lat, grid_lon)
        
    elif method == 'linear':
        # 線形補間
        interp_sin = LinearNDInterpolator(points, wind_dir_sin)(grid_points).reshape(grid_lat.shape)
        interp_cos = LinearNDInterpolator(points, wind_dir_cos)(grid_points).reshape(grid_lat.shape)
        interp_speed = LinearNDInterpolator(points, wind_speeds)(grid_points).reshape(grid_lat.shape)
        
        # NaN値を処理（外挿が必要な領域）
        mask = np.isnan(interp_sin)
        if np.any(mask):
            # 最近傍法で外挿
            from scipy.interpolate import NearestNDInterpolator
            nearest_sin = NearestNDInterpolator(points, wind_dir_sin)
            nearest_cos = NearestNDInterpolator(points, wind_dir_cos)
            nearest_speed = NearestNDInterpolator(points, wind_speeds)
            
            interp_sin[mask] = nearest_sin(grid_points[mask.flatten()][:, 0], grid_points[mask.flatten()][:, 1])
            interp_cos[mask] = nearest_cos(grid_points[mask.flatten()][:, 0], grid_points[mask.flatten()][:, 1])
            interp_speed[mask] = nearest_speed(grid_points[mask.flatten()][:, 0], grid_points[mask.flatten()][:, 1])
        
    else:  # 'idw' or fallback
        # 逆距離加重法
        interp_sin = np.zeros(grid_lat.shape)
        interp_cos = np.zeros(grid_lat.shape)
        interp_speed = np.zeros(grid_lat.shape)
        
        for i in range(grid_lat.shape[0]):
            for j in range(grid_lat.shape[1]):
                grid_point = np.array([grid_lat[i, j], grid_lon[i, j]])
                
                # 各観測点までの距離を計算
                distances = np.sqrt(np.sum((points - grid_point)**2, axis=1))
                
                # ゼロ距離の処理
                if np.any(distances == 0):
                    idx = np.where(distances == 0)[0][0]
                    interp_sin[i, j] = wind_dir_sin[idx]
                    interp_cos[i, j] = wind_dir_cos[idx]
                    interp_speed[i, j] = wind_speeds[idx]
                    continue
                
                # 逆距離重み
                weights = 1.0 / distances**2
                weights /= weights.sum()
                
                # 重み付き平均
                interp_sin[i, j] = np.sum(wind_dir_sin * weights)
                interp_cos[i, j] = np.sum(wind_dir_cos * weights)
                interp_speed[i, j] = np.sum(wind_speeds * weights)
    
    # sin/cos から風向を復元
    interp_dir = np.degrees(np.arctan2(interp_sin, interp_cos)) % 360
    
    # 風速の非負制約
    interp_speed = np.maximum(0, interp_speed)
    
    return interp_dir, interp_speed


def moving_average(data: List[float], window_size: int = 5) -> List[float]:
    """
    移動平均を計算します
    
    Parameters:
    -----------
    data : List[float]
        平均する数値のリスト
    window_size : int
        移動窓のサイズ
        
    Returns:
    --------
    List[float]
        移動平均の結果
    """
    if window_size < 1:
        return data
    
    result = []
    
    # 最初のポイントは元の値を使用
    for i in range(min(window_size // 2, len(data))):
        result.append(data[i])
    
    # 移動平均の計算
    for i in range(window_size // 2, len(data) - window_size // 2):
        window = data[i - window_size // 2:i + window_size // 2 + 1]
        result.append(sum(window) / len(window))
    
    # 最後のポイントは元の値を使用
    for i in range(len(data) - window_size // 2, len(data)):
        if i >= len(result):  # インデックス範囲チェック
            result.append(data[i])
    
    return result


def exponential_smoothing(data: List[float], alpha: float = 0.3) -> List[float]:
    """
    指数平滑化を計算します
    
    Parameters:
    -----------
    data : List[float]
        平滑化する数値のリスト
    alpha : float
        平滑化係数（0-1、大きいほど元データに近い）
        
    Returns:
    --------
    List[float]
        平滑化された結果
    """
    if not data:
        return []
    
    result = [data[0]]  # 最初の値は元のデータを使用
    
    for i in range(1, len(data)):
        smoothed = alpha * data[i] + (1 - alpha) * result[-1]
        result.append(smoothed)
    
    return result


def confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    信頼区間を計算します
    
    Parameters:
    -----------
    data : List[float]
        信頼区間を計算するデータ
    confidence : float
        信頼水準（0-1）
        
    Returns:
    --------
    Tuple[float, float]
        (下限, 上限)
    """
    import scipy.stats as stats
    
    if not data or len(data) < 2:
        return 0.0, 0.0
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # 不偏標準偏差
    
    # 自由度
    n = len(data)
    df = n - 1
    
    # t分布の臨界値
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    
    # 標準誤差
    se = std / np.sqrt(n)
    
    # 信頼区間
    lower = mean - t_crit * se
    upper = mean + t_crit * se
    
    return lower, upper


def weighted_avg_and_std(values: List[float], weights: List[float]) -> Tuple[float, float]:
    """
    重み付き平均と標準偏差を計算します
    
    Parameters:
    -----------
    values : List[float]
        値のリスト
    weights : List[float]
        重みのリスト
        
    Returns:
    --------
    Tuple[float, float]
        (重み付き平均, 重み付き標準偏差)
    """
    if not values or not weights or len(values) != len(weights):
        return 0.0, 0.0
    
    # 重みの合計がゼロの場合のチェック
    sum_weights = sum(weights)
    if sum_weights == 0:
        return 0.0, 0.0
    
    # 重み付き平均
    weighted_mean = sum(v * w for v, w in zip(values, weights)) / sum_weights
    
    # 重み付き分散
    variance = sum(w * ((v - weighted_mean) ** 2) for v, w in zip(values, weights)) / sum_weights
    
    # 標準偏差
    std_dev = math.sqrt(variance)
    
    return weighted_mean, std_dev


def linear_trend(data: List[float], times: List[float] = None) -> Tuple[float, float]:
    """
    線形トレンドを計算します
    
    Parameters:
    -----------
    data : List[float]
        値のリスト
    times : List[float], optional
        時間点のリスト（指定がなければ等間隔と仮定）
        
    Returns:
    --------
    Tuple[float, float]
        (傾き, 切片)
    """
    if not data or len(data) < 2:
        return 0.0, 0.0
    
    # 時間点が指定されていなければ等間隔と仮定
    if times is None:
        times = list(range(len(data)))
    
    # データ点数のチェック
    if len(data) != len(times):
        raise ValueError("データと時間点のリストの長さが一致しません")
    
    # NumPyの最小二乗法関数で傾きと切片を計算
    slope, intercept = np.polyfit(times, data, 1)
    
    return slope, intercept


def bayesian_update(prior_mean: float, prior_std: float, 
                  likelihood_mean: float, likelihood_std: float) -> Tuple[float, float]:
    """
    ベイズ更新を計算します（ガウス分布の場合）
    
    Parameters:
    -----------
    prior_mean, prior_std : float
        事前分布の平均と標準偏差
    likelihood_mean, likelihood_std : float
        尤度の平均と標準偏差
        
    Returns:
    --------
    Tuple[float, float]
        事後分布の(平均, 標準偏差)
    """
    # 精度（分散の逆数）を計算
    prior_precision = 1.0 / (prior_std ** 2) if prior_std > 0 else 0.0
    likelihood_precision = 1.0 / (likelihood_std ** 2) if likelihood_std > 0 else 0.0
    
    # 事後精度
    posterior_precision = prior_precision + likelihood_precision
    
    # ゼロ除算防止
    if posterior_precision == 0:
        return likelihood_mean, float('inf')
    
    # 事後分散
    posterior_var = 1.0 / posterior_precision
    
    # 事後平均
    posterior_mean = (prior_mean * prior_precision + likelihood_mean * likelihood_precision) / posterior_precision
    
    # 事後標準偏差
    posterior_std = math.sqrt(posterior_var)
    
    return posterior_mean, posterior_std

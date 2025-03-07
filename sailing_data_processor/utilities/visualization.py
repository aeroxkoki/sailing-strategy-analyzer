"""
可視化ユーティリティ - セーリングデータ処理用

風向風速データ、GPSトラック、およびその他の分析結果を視覚化するための
ユーティリティ関数を提供します。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import folium
from folium.plugins import HeatMap
import math
from typing import Dict, List, Tuple, Optional, Union, Any


def plot_gps_track(latitudes: List[float], longitudes: List[float], 
                 timestamps: List[float] = None, title: str = "GPS Track", 
                 color: str = 'blue', alpha: float = 0.8, 
                 marker_size: int = 5, show_start_end: bool = True,
                 ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    GPSトラックを描画します
    
    Parameters:
    -----------
    latitudes : List[float]
        緯度のリスト
    longitudes : List[float]
        経度のリスト
    timestamps : List[float], optional
        タイムスタンプのリスト（カラーマップのグラデーション用）
    title : str
        図のタイトル
    color : str
        トラックの色（timestampsが指定されていない場合）
    alpha : float
        透明度（0-1）
    marker_size : int
        マーカーサイズ
    show_start_end : bool
        スタートとエンドのマーカーを表示するかどうか
    ax : plt.Axes, optional
        描画先の軸オブジェクト（指定がなければ新規作成）
        
    Returns:
    --------
    plt.Figure
        描画された図のオブジェクト
    """
    # 軸オブジェクトの準備
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # トラックの描画
    if timestamps is not None:
        # タイムスタンプに基づくカラーマップ
        norm = Normalize(min(timestamps), max(timestamps))
        cmap = plt.cm.viridis
        colors = cmap(norm(timestamps))
        
        # 散布図として描画
        scatter = ax.scatter(longitudes, latitudes, c=timestamps, cmap=cmap, 
                         s=marker_size, alpha=alpha, edgecolors='none')
        
        # カラーバーを追加
        plt.colorbar(scatter, ax=ax, label='Time')
    else:
        # 単色で描画
        ax.plot(longitudes, latitudes, color=color, alpha=alpha, marker='.', 
               markersize=marker_size, linestyle='-')
    
    # スタートとエンドを強調表示
    if show_start_end and latitudes and longitudes:
        ax.plot(longitudes[0], latitudes[0], 'go', markersize=marker_size*2, label='Start')
        ax.plot(longitudes[-1], latitudes[-1], 'ro', markersize=marker_size*2, label='End')
        ax.legend()
    
    # 軸ラベルとタイトル
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 縦横比を等しくする
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


def plot_wind_arrows(grid_lats: np.ndarray, grid_lons: np.ndarray, 
                   wind_directions: np.ndarray, wind_speeds: np.ndarray,
                   confidence: Optional[np.ndarray] = None,
                   title: str = "Wind Field", skip: int = 1,
                   ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    風向風速を矢印で描画します
    
    Parameters:
    -----------
    grid_lats, grid_lons : np.ndarray
        緯度・経度のグリッド
    wind_directions : np.ndarray
        風向のグリッド（度数法、0-360）
    wind_speeds : np.ndarray
        風速のグリッド
    confidence : np.ndarray, optional
        信頼度のグリッド（0-1）
    title : str
        図のタイトル
    skip : int
        表示する矢印を間引く間隔
    ax : plt.Axes, optional
        描画先の軸オブジェクト（指定がなければ新規作成）
        
    Returns:
    --------
    plt.Figure
        描画された図のオブジェクト
    """
    # 軸オブジェクトの準備
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure
    
    # 風速のカラーマップ
    norm = Normalize(vmin=np.min(wind_speeds), vmax=np.max(wind_speeds))
    cmap = plt.cm.viridis
    
    # 風速をカラーマップで表示
    contour = ax.contourf(grid_lons, grid_lats, wind_speeds, cmap=cmap, levels=20)
    plt.colorbar(contour, ax=ax, label='Wind Speed (knots)')
    
    # 風向を矢印で表示（風が吹いてくる方向）
    u = -np.sin(np.radians(wind_directions[::skip, ::skip]))
    v = -np.cos(np.radians(wind_directions[::skip, ::skip]))
    
    # 信頼度に基づく透明度
    if confidence is not None:
        alpha = confidence[::skip, ::skip]
    else:
        alpha = np.ones_like(u) * 0.8
    
    # 矢印の描画
    quiver = ax.quiver(grid_lons[::skip, ::skip], grid_lats[::skip, ::skip], 
                     u, v, alpha=alpha, color='black', scale=25)
    
    # 矢印の凡例
    ax.quiverkey(quiver, 0.9, 0.1, 1, '5 knots', labelpos='E', coordinates='figure')
    
    # 信頼度を等高線で表示（オプション）
    if confidence is not None:
        ax.contour(grid_lons, grid_lats, confidence, 
                  levels=[0.3, 0.5, 0.7, 0.9], colors='red', alpha=0.5, linestyles='dashed')
    
    # 軸ラベルとタイトル
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 縦横比を等しくする
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


def create_folium_map(center_lat: float, center_lon: float, zoom_start: int = 13,
                    tiles: str = 'CartoDB positron') -> folium.Map:
    """
    Folium地図を作成します
    
    Parameters:
    -----------
    center_lat, center_lon : float
        地図の中心座標
    zoom_start : int
        初期ズームレベル
    tiles : str
        地図タイルのスタイル
        
    Returns:
    --------
    folium.Map
        Folium地図オブジェクト
    """
    # 地図を作成
    m = folium.Map(location=[center_lat, center_lon], 
                 zoom_start=zoom_start, 
                 tiles=tiles)
    
    return m


def add_gps_track_to_map(m: folium.Map, latitudes: List[float], longitudes: List[float],
                        timestamps: List[float] = None, boat_id: str = "Boat",
                        color: str = 'blue', weight: int = 3) -> folium.Map:
    """
    Folium地図にGPSトラックを追加します
    
    Parameters:
    -----------
    m : folium.Map
        Folium地図オブジェクト
    latitudes, longitudes : List[float]
        緯度・経度のリスト
    timestamps : List[float], optional
        タイムスタンプのリスト
    boat_id : str
        艇の識別子
    color : str
        トラックの色
    weight : int
        トラックの線の太さ
        
    Returns:
    --------
    folium.Map
        更新された地図オブジェクト
    """
    # GPSトラックを追加
    points = list(zip(latitudes, longitudes))
    folium.PolyLine(
        points,
        color=color,
        weight=weight,
        opacity=0.8,
        tooltip=boat_id
    ).add_to(m)
    
    # スタートポイントにマーカーを追加
    if points:
        folium.Marker(
            location=points[0],
            popup=f"{boat_id} Start",
            icon=folium.Icon(color='green', icon='play'),
        ).add_to(m)
        
        # フィニッシュポイントにマーカーを追加
        folium.Marker(
            location=points[-1],
            popup=f"{boat_id} Finish",
            icon=folium.Icon(color='red', icon='stop'),
        ).add_to(m)
    
    return m


def add_wind_field_to_map(m: folium.Map, grid_lats: np.ndarray, grid_lons: np.ndarray,
                         wind_directions: np.ndarray, wind_speeds: np.ndarray,
                         confidence: Optional[np.ndarray] = None,
                         skip: int = 2) -> folium.Map:
    """
    Folium地図に風の場を追加します
    
    Parameters:
    -----------
    m : folium.Map
        Folium地図オブジェクト
    grid_lats, grid_lons : np.ndarray
        緯度・経度のグリッド
    wind_directions : np.ndarray
        風向のグリッド
    wind_speeds : np.ndarray
        風速のグリッド
    confidence : np.ndarray, optional
        信頼度のグリッド
    skip : int
        表示する矢印を間引く間隔
        
    Returns:
    --------
    folium.Map
        更新された地図オブジェクト
    """
    # 風速をヒートマップとして表示
    heat_data = []
    for i in range(0, grid_lats.shape[0], skip):
        for j in range(0, grid_lats.shape[1], skip):
            lat = grid_lats[i, j]
            lon = grid_lons[i, j]
            speed = wind_speeds[i, j]
            conf = confidence[i, j] if confidence is not None else 1.0
            
            if conf > 0.3:  # 信頼度が低すぎる点は除外
                heat_data.append([lat, lon, speed])
    
    # ヒートマップを追加
    HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)
    
    # 風向を矢印アイコンで表示
    for i in range(0, grid_lats.shape[0], skip*2):  # さらに間引く
        for j in range(0, grid_lats.shape[1], skip*2):
            lat = grid_lats[i, j]
            lon = grid_lons[i, j]
            direction = wind_directions[i, j]
            speed = wind_speeds[i, j]
            conf = confidence[i, j] if confidence is not None else 1.0
            
            if conf > 0.4:  # 信頼度が低すぎる点は除外
                # 矢印の向きを風が吹いてくる方向に
                icon = folium.features.DivIcon(
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                    html=f'<div style="transform: rotate({direction}deg);">'
                         f'<span style="font-size: 20px; color: black;">→</span></div>'
                )
                
                folium.Marker(
                    location=[lat, lon],
                    icon=icon,
                    tooltip=f"風向: {direction:.1f}°, 風速: {speed:.1f}ノット"
                ).add_to(m)
    
    return m


def plot_wind_time_series(times: List[float], wind_directions: List[float], 
                        wind_speeds: List[float], confidence: Optional[List[float]] = None,
                        title: str = "Wind Time Series") -> plt.Figure:
    """
    風向風速の時系列を描画します
    
    Parameters:
    -----------
    times : List[float]
        時間軸データ
    wind_directions : List[float]
        風向のリスト
    wind_speeds : List[float]
        風速のリスト
    confidence : List[float], optional
        信頼度のリスト
    title : str
        図のタイトル
        
    Returns:
    --------
    plt.Figure
        描画された図のオブジェクト
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 風向の描画
    ax1.plot(times, wind_directions, 'r-', marker='o', markersize=4)
    ax1.set_ylabel('Wind Direction (degrees)')
    ax1.set_title(title)
    ax1.grid(True)
    ax1.set_ylim(0, 360)
    ax1.set_yticks([0, 90, 180, 270, 360])
    ax1.set_yticklabels(['N', 'E', 'S', 'W', 'N'])
    
    # 信頼区間の追加（オプション）
    if confidence:
        # 信頼度から信頼区間の幅を計算
        interval_width = [10 * (1 - conf) * 10 for conf in confidence]
        
        # 信頼区間の上下限
        lower_bounds = [(dir - width) % 360 for dir, width in zip(wind_directions, interval_width)]
        upper_bounds = [(dir + width) % 360 for dir, width in zip(wind_directions, interval_width)]
        
        # 信頼区間の描画
        for i in range(len(times)):
            ax1.plot([times[i], times[i]], [lower_bounds[i], upper_bounds[i]], 
                    'r-', alpha=0.2)
    
    # 風速の描画
    ax2.plot(times, wind_speeds, 'b-', marker='o', markersize=4)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Wind Speed (knots)')
    ax2.grid(True)
    
    # 信頼区間の追加（オプション）
    if confidence:
        # 信頼度から信頼区間の幅を計算
        interval_width = [spd * (1 - conf) * 0.5 for spd, conf in zip(wind_speeds, confidence)]
        
        # 信頼区間の上下限
        lower_bounds = [max(0, spd - width) for spd, width in zip(wind_speeds, interval_width)]
        upper_bounds = [spd + width for spd, width in zip(wind_speeds, interval_width)]
        
        # 信頼区間の描画
        for i in range(len(times)):
            ax2.plot([times[i], times[i]], [lower_bounds[i], upper_bounds[i]], 
                    'b-', alpha=0.2)
    
    plt.tight_layout()
    return fig


def plot_wind_rose(wind_directions: List[float], wind_speeds: Optional[List[float]] = None,
                 bins: int = 16, title: str = "Wind Rose") -> plt.Figure:
    """
    風配図（ウィンドローズ）を描画します
    
    Parameters:
    -----------
    wind_directions : List[float]
        風向のリスト（度数法、0-360）
    wind_speeds : List[float], optional
        風速のリスト（指定された場合は風速ごとに色分け）
    bins : int
        方位の分割数
    title : str
        図のタイトル
        
    Returns:
    --------
    plt.Figure
        描画された図のオブジェクト
    """
    # 極座標プロットの準備
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # 方位の分割（ラジアンに変換）
    bin_width = 2 * np.pi / bins
    bin_edges = np.linspace(0, 2*np.pi, bins+1)
    
    # 度数法からラジアンに変換
    # 注: 極座標プロットでは北が0、時計回りが正方向なので変換が必要
    directions_rad = np.radians((90 - np.array(wind_directions)) % 360)
    
    if wind_speeds is not None:
        # 風速の区間を設定
        speed_bins = [0, 5, 10, 15, 20, float('inf')]
        speed_labels = ['0-5', '5-10', '10-15', '15-20', '20+']
        
        # 風速カテゴリごとにヒストグラムを作成
        for i, (lower, upper) in enumerate(zip(speed_bins[:-1], speed_bins[1:])):
            # 指定風速範囲のデータのみを抽出
            mask = (np.array(wind_speeds) >= lower) & (np.array(wind_speeds) < upper)
            if np.any(mask):
                # 極座標ヒストグラムの計算
                hist, _ = np.histogram(directions_rad[mask], bins=bin_edges)
                
                # バーの描画（風向が来る方向なので、-を付けて反転）
                ax.bar(bin_edges[:-1], hist, width=bin_width, bottom=0,
                      alpha=0.7, label=f'{speed_labels[i]} knots',
                      color=plt.cm.viridis(i / len(speed_bins)))
    else:
        # 風向のみのヒストグラム
        hist, _ = np.histogram(directions_rad, bins=bin_edges)
        
        # バーの描画
        ax.bar(bin_edges[:-1], hist, width=bin_width, bottom=0, color='skyblue')
    
    # 方位ラベルの設定
    ax.set_theta_zero_location('N')  # 北を0度に設定
    ax.set_theta_direction(-1)  # 時計回りに設定
    
    # 方位の主軸のラベル
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(['N', 'E', 'S', 'W'])
    
    # タイトルと凡例
    ax.set_title(title)
    if wind_speeds is not None:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    return fig


def plot_boat_performance(boat_speeds: List[float], wind_speeds: List[float], 
                        wind_angles: List[float], boat_type: str = "Unknown",
                        title: str = None) -> plt.Figure:
    """
    艇のパフォーマンス（極座標）グラフを描画します
    
    Parameters:
    -----------
    boat_speeds : List[float]
        艇速のリスト
    wind_speeds : List[float]
        風速のリスト
    wind_angles : List[float]
        相対風向角のリスト（艇首を0度として）
    boat_type : str
        艇種
    title : str, optional
        図のタイトル
        
    Returns:
    --------
    plt.Figure
        描画された図のオブジェクト
    """
    # タイトルの設定（指定がなければ艇種から自動生成）
    if title is None:
        title = f"{boat_type} Performance Polar"
    
    # 極座標プロットの準備
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # 相対風向角をラジアンに変換
    angles_rad = np.radians(wind_angles)
    
    # 風速比（艇速/風速）を計算
    speed_ratios = [b/w if w > 0 else 0 for b, w in zip(boat_speeds, wind_speeds)]
    
    # 散布図として描画
    scatter = ax.scatter(angles_rad, speed_ratios, c=wind_speeds, 
                       cmap=plt.cm.viridis, alpha=0.7, s=30, edgecolors='none')
    
    # カラーバーを追加
    plt.colorbar(scatter, ax=ax, label='Wind Speed (knots)')
    
    # 平均曲線の計算（風速区間ごと）
    wind_speed_bins = [0, 5, 10, 15, 20, 25]
    
    for i in range(len(wind_speed_bins) - 1):
        low, high = wind_speed_bins[i], wind_speed_bins[i+1]
        mask = (np.array(wind_speeds) >= low) & (np.array(wind_speeds) < high)
        
        if np.sum(mask) > 10:  # 十分なデータポイントがある場合
            # 角度ごとのグループ化
            angle_bins = np.linspace(0, 360, 37)  # 10度刻み
            avg_ratios = []
            bin_centers = []
            
            for j in range(len(angle_bins) - 1):
                angle_low, angle_high = angle_bins[j], angle_bins[j+1]
                angle_mask = mask & (np.array(wind_angles) >= angle_low) & (np.array(wind_angles) < angle_high)
                
                if np.sum(angle_mask) > 0:
                    avg_ratio = np.mean(np.array(speed_ratios)[angle_mask])
                    avg_ratios.append(avg_ratio)
                    bin_centers.append((angle_low + angle_high) / 2)
            
            if bin_centers:
                # ラジアンに変換
                bin_centers_rad = np.radians(bin_centers)
                
                # 曲線を描画
                ax.plot(bin_centers_rad, avg_ratios, '-', 
                      label=f'{low}-{high} knots', 
                      linewidth=2, alpha=0.8)
    
    # 方位ラベルの設定
    ax.set_theta_zero_location('N')  # 船首方向を0度に設定
    ax.set_theta_direction(-1)  # 時計回りに設定
    
    # 方位のラベル
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
    
    # タイトルと凡例
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    return fig


def plot_comparative_performance(boat_data: Dict[str, Dict[str, List[float]]],
                               title: str = "Boat Performance Comparison") -> plt.Figure:
    """
    複数艇のパフォーマンス比較グラフを描画します
    
    Parameters:
    -----------
    boat_data : Dict[str, Dict[str, List[float]]]
        艇ID: {'speeds': [...], 'times': [...], 'vmg': [...]} の辞書
    title : str
        図のタイトル
        
    Returns:
    --------
    plt.Figure
        描画された図のオブジェクト
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 各艇の速度をプロット
    for boat_id, data in boat_data.items():
        times = data.get('times', [])
        speeds = data.get('speeds', [])
        vmg = data.get('vmg', [])
        
        if times and speeds:
            ax1.plot(times, speeds, '-', label=f'{boat_id}', linewidth=2, alpha=0.8)
        
        if times and vmg:
            ax2.plot(times, vmg, '--', label=f'{boat_id}', linewidth=2, alpha=0.8)
    
    # 軸ラベルとタイトル
    ax1.set_ylabel('Boat Speed (knots)')
    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('VMG (knots)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig


def create_wind_field_animation(wind_fields: List[Dict[str, Any]], 
                              output_file: str = 'wind_animation.gif', 
                              fps: int = 2) -> None:
    """
    風の場のアニメーションGIFを作成します
    
    Parameters:
    -----------
    wind_fields : List[Dict[str, Any]]
        時系列の風の場データ
    output_file : str
        出力ファイルパス
    fps : int
        フレームレート（1秒あたりのフレーム数）
    """
    import matplotlib.animation as animation
    from datetime import datetime
    
    if not wind_fields:
        raise ValueError("風の場データが空です")
    
    # 最初のフレームからグリッドサイズを取得
    grid_shape = wind_fields[0]['wind_direction'].shape
    
    # 図とアニメーションの設定
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def update(frame):
        ax.clear()
        
        # 現在のフレームの風の場データ
        field = wind_fields[frame]
        
        # 風速をカラーマップで表示
        contour = ax.contourf(field['lon_grid'], field['lat_grid'], 
                            field['wind_speed'], cmap=plt.cm.viridis, levels=20)
        
        # 風向を矢印で表示
        skip = max(1, grid_shape[0] // 20)
        
        # 風が吹いてくる方向に矢印を向ける
        u = -np.sin(np.radians(field['wind_direction'][::skip, ::skip]))
        v = -np.cos(np.radians(field['wind_direction'][::skip, ::skip]))
        
        quiver = ax.quiver(field['lon_grid'][::skip, ::skip], field['lat_grid'][::skip, ::skip], 
                         u, v, color='black', scale=25)
        
        # 信頼度を等高線で表示（存在する場合）
        if 'confidence' in field:
            ax.contour(field['lon_grid'], field['lat_grid'], field['confidence'], 
                      levels=[0.3, 0.5, 0.7, 0.9], colors='red', alpha=0.5, linestyles='dashed')
        
        # 時間を表示（存在する場合）
        if 'time' in field:
            time_str = field['time'].strftime('%Y-%m-%d %H:%M:%S')
            ax.set_title(f"Wind Field at {time_str}")
        else:
            ax.set_title(f"Wind Field - Frame {frame+1}/{len(wind_fields)}")
        
        # 軸ラベル
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 縦横比を等しくする
        ax.set_aspect('equal', adjustable='box')
        
        return contour, quiver
    
    # アニメーションの作成
    ani = animation.FuncAnimation(
        fig, update, frames=len(wind_fields), 
        blit=False, interval=1000//fps
    )
    
    # GIFとして保存
    ani.save(output_file, writer='pillow', fps=fps)
    
    plt.close(fig)

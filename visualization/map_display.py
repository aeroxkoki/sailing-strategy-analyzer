"""
セーリング戦略分析システム - マップ表示モジュール

このモジュールは、Foliumを使用してセーリングデータのマップ表示機能を提供します。
航跡表示、マーク表示、ヒートマップなどの機能を含みます。

作成日: 2025-03-05
"""

import folium
from folium import plugins
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import branca.colormap as cm


class SailingMapDisplay:
    """
    Foliumを使用してセーリングデータのマップ表示を行うクラス
    """
    
    def __init__(self):
        """
        SailingMapDisplayクラスの初期化
        """
        self.map_object = None
        self.colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']
        self.default_tile = "CartoDB positron"
        self.available_tiles = {
            "ポジトロン": "CartoDB positron",
            "オープンストリートマップ": "OpenStreetMap",
            "衛星写真": "Stamen Terrain",
            "地形図": "Stamen Toner",
            "暗い背景": "CartoDB dark_matter"
        }
    
    def create_map(self, center=None, zoom_start=13, tile=None):
        """
        ベースとなる地図を作成します
        
        Parameters:
        -----------
        center : tuple, optional
            地図の中心座標 (緯度, 経度)
        zoom_start : int, optional
            初期ズームレベル
        tile : str, optional
            使用するタイルのタイプ
            
        Returns:
        --------
        folium.Map
            作成された地図オブジェクト
        """
        # デフォルト中心位置（東京湾）
        if center is None:
            center = (35.5, 139.8)
        
        # タイルが指定されていない場合はデフォルトを使用
        if tile is None or tile not in self.available_tiles:
            tile = self.default_tile
        else:
            tile = self.available_tiles[tile]
        
        # 地図オブジェクトの作成
        self.map_object = folium.Map(
            location=center,
            zoom_start=zoom_start,
            tiles=tile
        )
        
        # スケールバーの追加
        plugins.MousePosition().add_to(self.map_object)
        
        return self.map_object
    
    def add_track(self, data, boat_name, color=None):
        """
        ボートの航跡を地図に追加します
        
        Parameters:
        -----------
        data : pandas.DataFrame
            緯度・経度を含むボートのデータフレーム
        boat_name : str
            ボートの名前
        color : str, optional
            航跡の色
            
        Returns:
        --------
        folium.Map
            航跡が追加された地図オブジェクト
        """
        if self.map_object is None:
            raise ValueError("先にcreate_map()を呼び出して地図を作成してください")
        
        if color is None:
            # 色が指定されていない場合、デフォルトのカラーリストから選択
            color_index = hash(boat_name) % len(self.colors)
            color = self.colors[color_index]
        
        # 航跡のポイントリストを作成
        track_points = list(zip(data['latitude'], data['longitude']))
        
        # 航跡の描画
        track_line = folium.PolyLine(
            track_points,
            color=color,
            weight=3,
            opacity=0.8,
            tooltip=f"{boat_name}の航跡"
        )
        track_line.add_to(self.map_object)
        
        # スタート位置とゴール位置にマーカーを追加
        start_point = track_points[0]
        folium.Marker(
            location=start_point,
            popup=f"{boat_name} スタート",
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(self.map_object)
        
        end_point = track_points[-1]
        folium.Marker(
            location=end_point,
            popup=f"{boat_name} フィニッシュ",
            icon=folium.Icon(color='red', icon='stop', prefix='fa')
        ).add_to(self.map_object)
        
        # 地図の表示範囲を航跡に合わせて調整
        self.map_object.fit_bounds([start_point, end_point])
        
        return self.map_object
    
    def add_course_marks(self, marks_data):
        """
        コースのマーク（ブイなど）を地図に追加します
        
        Parameters:
        -----------
        marks_data : dict or pandas.DataFrame
            マークのデータ（名前、位置など）
            
        Returns:
        --------
        folium.Map
            マークが追加された地図オブジェクト
        """
        if self.map_object is None:
            raise ValueError("先にcreate_map()を呼び出して地図を作成してください")
        
        if isinstance(marks_data, pd.DataFrame):
            # DataFrameの場合、各行をマークとして処理
            for idx, row in marks_data.iterrows():
                mark_name = row.get('name', f'マーク{idx}')
                lat = row['latitude']
                lon = row['longitude']
                
                folium.Marker(
                    location=(lat, lon),
                    popup=mark_name,
                    icon=folium.Icon(color='blue', icon='flag', prefix='fa')
                ).add_to(self.map_object)
        else:
            # 辞書の場合、各キーをマーク名として処理
            for mark_name, mark_data in marks_data.items():
                lat = mark_data['latitude']
                lon = mark_data['longitude']
                
                folium.Marker(
                    location=(lat, lon),
                    popup=mark_name,
                    icon=folium.Icon(color='blue', icon='flag', prefix='fa')
                ).add_to(self.map_object)
        
        return self.map_object
    
    def add_start_finish_line(self, start_line=None, finish_line=None):
        """
        スタートラインとフィニッシュラインを地図に追加します
        
        Parameters:
        -----------
        start_line : list of tuples, optional
            スタートラインの座標 [(lat1, lon1), (lat2, lon2)]
        finish_line : list of tuples, optional
            フィニッシュラインの座標 [(lat1, lon1), (lat2, lon2)]
            
        Returns:
        --------
        folium.Map
            ラインが追加された地図オブジェクト
        """
        if self.map_object is None:
            raise ValueError("先にcreate_map()を呼び出して地図を作成してください")
        
        # スタートラインの追加
        if start_line and len(start_line) == 2:
            folium.PolyLine(
                start_line,
                color='green',
                weight=3,
                opacity=1.0,
                tooltip="スタートライン"
            ).add_to(self.map_object)
            
            # スタートライン両端にマーカー追加
            folium.Marker(
                location=start_line[0],
                popup="スタートライン（左）",
                icon=folium.Icon(color='green', icon='flag', prefix='fa')
            ).add_to(self.map_object)
            
            folium.Marker(
                location=start_line[1],
                popup="スタートライン（右）",
                icon=folium.Icon(color='green', icon='flag', prefix='fa')
            ).add_to(self.map_object)
        
        # フィニッシュラインの追加
        if finish_line and len(finish_line) == 2:
            folium.PolyLine(
                finish_line,
                color='red',
                weight=3,
                opacity=1.0,
                tooltip="フィニッシュライン"
            ).add_to(self.map_object)
            
            # フィニッシュライン両端にマーカー追加
            folium.Marker(
                location=finish_line[0],
                popup="フィニッシュライン（左）",
                icon=folium.Icon(color='red', icon='flag', prefix='fa')
            ).add_to(self.map_object)
            
            folium.Marker(
                location=finish_line[1],
                popup="フィニッシュライン（右）",
                icon=folium.Icon(color='red', icon='flag', prefix='fa')
            ).add_to(self.map_object)
        
        return self.map_object
    
    def add_wind_direction(self, lat, lon, direction, strength=None):
        """
        風向を示す矢印を地図に追加します
        
        Parameters:
        -----------
        lat : float
            風向矢印の位置（緯度）
        lon : float
            風向矢印の位置（経度）
        direction : float
            風向（度、0-360）
        strength : float, optional
            風速
            
        Returns:
        --------
        folium.Map
            風向が追加された地図オブジェクト
        """
        if self.map_object is None:
            raise ValueError("先にcreate_map()を呼び出して地図を作成してください")
        
        # 風向の矢印を追加
        arrow = folium.RegularPolygonMarker(
            location=(lat, lon),
            number_of_sides=3,
            rotation=direction,
            fill_color='darkblue',
            fill_opacity=0.6,
            radius=10 if strength is None else min(8 + strength, 20)
        )
        arrow.add_to(self.map_object)
        
        # 風速が指定されている場合、情報を追加
        if strength is not None:
            folium.Marker(
                location=(lat, lon),
                popup=f"風向: {direction}°, 風速: {strength}ノット",
                icon=folium.DivIcon(
                    icon_size=(0, 0),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size: 10pt; color: darkblue;">{strength}kt</div>'
                )
            ).add_to(self.map_object)
        
        return self.map_object
    
    def add_speed_heatmap(self, data):
        """
        速度のヒートマップを地図に追加します
        
        Parameters:
        -----------
        data : pandas.DataFrame
            緯度・経度・速度を含むデータフレーム
            
        Returns:
        --------
        folium.Map
            ヒートマップが追加された地図オブジェクト
        """
        if self.map_object is None:
            raise ValueError("先にcreate_map()を呼び出して地図を作成してください")
        
        if 'speed' not in data.columns or 'latitude' not in data.columns or 'longitude' not in data.columns:
            raise ValueError("データには 'latitude', 'longitude', 'speed' 列が必要です")
        
        # ヒートマップ用のデータを準備
        heat_data = [[row['latitude'], row['longitude'], row['speed']] for _, row in data.iterrows()]
        
        # カラーマップの作成
        colormap = cm.LinearColormap(
            ['blue', 'green', 'yellow', 'red'],
            vmin=data['speed'].min(),
            vmax=data['speed'].max()
        )
        
        # ヒートマップをマップに追加
        plugins.HeatMap(
            heat_data,
            radius=15,
            blur=10,
            gradient={0.4: 'blue', 0.65: 'green', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(self.map_object)
        
        # カラーバーを追加
        colormap.caption = '速度 (ノット)'
        self.map_object.add_child(colormap)
        
        return self.map_object
    
    def add_time_slider(self, data_dict):
        """
        時間スライダーを追加して、時間に基づく航跡表示を可能にします
        
        Parameters:
        -----------
        data_dict : dict
            ボート名をキー、時系列データフレームを値とする辞書
            
        Returns:
        --------
        folium.Map
            時間スライダーが追加された地図オブジェクト
        """
        if self.map_object is None:
            raise ValueError("先にcreate_map()を呼び出して地図を作成してください")
        
        # この機能はWeek 2で実装予定
        # プレースホルダーとして基本的な実装だけ行います
        
        # 時間の範囲を特定
        all_times = []
        for boat_name, df in data_dict.items():
            if 'timestamp' in df.columns:
                all_times.extend(df['timestamp'].tolist())
        
        if not all_times:
            raise ValueError("データに 'timestamp' 列が必要です")
        
        min_time = min(all_times)
        max_time = max(all_times)
        
        # ユーザーへのメッセージを表示
        folium.Marker(
            location=[0, 0],  # 画面外の位置
            icon=folium.DivIcon(
                icon_size=(0, 0),
                icon_anchor=(0, 0),
                html=f'<div style="background-color: white; padding: 5px; border-radius: 5px;">時間スライダー機能は開発中です（Week 2）</div>'
            )
        ).add_to(self.map_object)
        
        return self.map_object
    
    def save_map(self, file_path="sailing_map.html"):
        """
        現在の地図をHTMLファイルとして保存します
        
        Parameters:
        -----------
        file_path : str, optional
            保存先のファイルパス
            
        Returns:
        --------
        bool
            保存が成功したかどうか
        """
        if self.map_object is None:
            print("地図が作成されていません")
            return False
        
        try:
            self.map_object.save(file_path)
            print(f"地図を保存しました: {file_path}")
            return True
        except Exception as e:
            print(f"地図の保存エラー: {e}")
            return False

"""
セーリング戦略分析システム - 可視化モジュール

このモジュールは、セーリングのGPSデータを視覚的に表現するための
可視化機能を提供します。Foliumを使用したマップ表示や、
Plotlyを用いたパフォーマンスグラフ表示などの機能を含みます。

作成日: 2025-03-05
"""

import os
import pandas as pd
import numpy as np
import folium
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class SailingVisualizer:
    """
    セーリングデータの可視化を行うメインクラス
    
    このクラスは、データ処理モジュールから受け取ったデータを
    マップ表示やグラフ表示などの形で可視化する機能を提供します。
    """
    
    def __init__(self, data_processor=None):
        """
        SailingVisualizerクラスの初期化
        
        Parameters:
        -----------
        data_processor : object, optional
            データ処理モジュールのインスタンス。None の場合は後で set_data_processor メソッドで設定します。
        """
        self.data_processor = data_processor
        self.boats_data = {}  # ボート名をキー、データフレームを値とする辞書
        self.course_data = None  # コース情報（マーク位置など）
        self.map_object = None
        self.colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']
        
    def set_data_processor(self, data_processor):
        """
        データ処理モジュールを設定します
        
        Parameters:
        -----------
        data_processor : object
            データ処理モジュールのインスタンス
        """
        self.data_processor = data_processor
        
    def load_boat_data(self, boat_name, data=None, file_path=None):
        """
        ボートのGPSデータを読み込みます
        
        Parameters:
        -----------
        boat_name : str
            ボートの名前（識別子）
        data : pandas.DataFrame, optional
            すでに処理済みのデータフレーム
        file_path : str, optional
            GPXファイルパス（dataが指定されていない場合）
        
        Returns:
        --------
        bool
            データの読み込みが成功したかどうか
        """
        if data is not None:
            # 既に処理済みのデータを使用
            self.boats_data[boat_name] = data
            return True
            
        elif file_path is not None and self.data_processor is not None:
            # データ処理モジュールを使用してファイルからデータを読み込む
            try:
                processed_data = self.data_processor.process_gpx_file(file_path)
                self.boats_data[boat_name] = processed_data
                return True
            except Exception as e:
                print(f"データ読み込みエラー: {e}")
                return False
        else:
            print("データまたはファイルパスを指定してください")
            return False
    
    def load_course_data(self, course_data=None, file_path=None):
        """
        コース情報（マーク位置など）を読み込みます
        
        Parameters:
        -----------
        course_data : pandas.DataFrame or dict, optional
            既に処理済みのコースデータ
        file_path : str, optional
            コースデータのファイルパス
        
        Returns:
        --------
        bool
            コースデータの読み込みが成功したかどうか
        """
        if course_data is not None:
            self.course_data = course_data
            return True
            
        elif file_path is not None and self.data_processor is not None:
            try:
                self.course_data = self.data_processor.load_course_data(file_path)
                return True
            except Exception as e:
                print(f"コースデータ読み込みエラー: {e}")
                return False
        else:
            print("コースデータまたはファイルパスを指定してください")
            return False
    
    def create_base_map(self, center=None, zoom_start=13):
        """
        ベースとなる地図を作成します
        
        Parameters:
        -----------
        center : tuple, optional
            地図の中心座標 (緯度, 経度)。Noneの場合は最初のボートデータから自動取得。
        zoom_start : int, optional
            初期ズームレベル
        
        Returns:
        --------
        folium.Map
            作成された地図オブジェクト
        """
        # 中心位置が指定されていない場合、データから取得
        if center is None and self.boats_data:
            # 最初のボートのデータを使用
            first_boat_data = list(self.boats_data.values())[0]
            lat_center = first_boat_data['latitude'].mean()
            lon_center = first_boat_data['longitude'].mean()
            center = (lat_center, lon_center)
        elif center is None:
            # デフォルト値（東京湾）
            center = (35.5, 139.8)
        
        # 地図オブジェクトの作成
        self.map_object = folium.Map(
            location=center,
            zoom_start=zoom_start,
            tiles="CartoDB positron"
        )
        
        return self.map_object
    
    def visualize_single_boat(self, boat_name, map_object=None, color=None):
        """
        単一のボートの航跡を可視化します
        
        Parameters:
        -----------
        boat_name : str
            表示するボートの名前
        map_object : folium.Map, optional
            使用するマップオブジェクト。Noneの場合は新規作成。
        color : str, optional
            航跡の色。Noneの場合はデフォルトカラーリストから選択。
        
        Returns:
        --------
        folium.Map
            航跡が追加された地図オブジェクト
        """
        if boat_name not in self.boats_data:
            print(f"ボート '{boat_name}' のデータがありません")
            return None
        
        boat_data = self.boats_data[boat_name]
        
        # 地図オブジェクトがない場合は作成
        if map_object is None and self.map_object is None:
            map_object = self.create_base_map()
        elif map_object is None:
            map_object = self.map_object
        
        # 色が指定されていない場合、デフォルトカラーリストから選択
        if color is None:
            color_index = len(self.boats_data) % len(self.colors)
            color = self.colors[color_index]
        
        # 航跡の描画
        points = list(zip(boat_data['latitude'], boat_data['longitude']))
        folium.PolyLine(
            points,
            color=color,
            weight=3,
            opacity=0.8,
            tooltip=boat_name
        ).add_to(map_object)
        
        # スタート位置とゴール位置にマーカーを追加
        start_point = points[0]
        folium.Marker(
            location=start_point,
            popup=f"{boat_name} スタート",
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(map_object)
        
        end_point = points[-1]
        folium.Marker(
            location=end_point,
            popup=f"{boat_name} フィニッシュ",
            icon=folium.Icon(color='red', icon='stop', prefix='fa')
        ).add_to(map_object)
        
        return map_object
    
    def visualize_all_boats(self, map_object=None):
        """
        すべてのボートの航跡を可視化します
        
        Parameters:
        -----------
        map_object : folium.Map, optional
            使用するマップオブジェクト。Noneの場合は新規作成。
        
        Returns:
        --------
        folium.Map
            航跡が追加された地図オブジェクト
        """
        # 地図オブジェクトがない場合は作成
        if map_object is None and self.map_object is None:
            map_object = self.create_base_map()
        elif map_object is None:
            map_object = self.map_object
        
        # 各ボートの航跡を追加
        for i, boat_name in enumerate(self.boats_data.keys()):
            color = self.colors[i % len(self.colors)]
            self.visualize_single_boat(boat_name, map_object, color)
        
        # コース情報が存在する場合は表示
        if self.course_data is not None:
            self._add_course_markers(map_object)
        
        return map_object
    
    def _add_course_markers(self, map_object):
        """
        コース情報（マークなど）を地図に追加します
        
        Parameters:
        -----------
        map_object : folium.Map
            マーカーを追加する地図オブジェクト
        """
        if self.course_data is None:
            return
        
        # マークの追加（実際の形式によって実装を調整する必要あり）
        for mark_name, mark_data in self.course_data.items():
            if 'latitude' in mark_data and 'longitude' in mark_data:
                folium.Marker(
                    location=(mark_data['latitude'], mark_data['longitude']),
                    popup=mark_name,
                    icon=folium.Icon(color='blue', icon='flag', prefix='fa')
                ).add_to(map_object)
    
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
    
    # 以下の関数は次のスプリントで実装予定（Week 2）
    def create_speed_polar_plot(self, boat_name):
        """
        ボートの速度極座標グラフを作成します（Week 2）
        
        Parameters:
        -----------
        boat_name : str
            表示するボートの名前
        
        Returns:
        --------
        plotly.graph_objects.Figure
            ポーラーグラフの図オブジェクト
        """
        # プレースホルダー - Week 2で実装予定
        pass
    
    def create_timeline_visualization(self, boat_names=None):
        """
        ボートの時間ベースのビジュアライゼーションを作成します（Week 2）
        
        Parameters:
        -----------
        boat_names : list of str, optional
            表示するボートの名前のリスト。None の場合はすべてのボート。
        
        Returns:
        --------
        plotly.graph_objects.Figure
            タイムラインの図オブジェクト
        """
        # プレースホルダー - Week 2で実装予定
        pass
    
    def visualize_multiple_boats(self, boat_names=None, map_object=None, show_labels=True, sync_time=False):
        """
        指定した複数のボートの航跡を可視化します
        
        Parameters:
        -----------
        boat_names : list of str, optional
            表示するボートの名前のリスト。None の場合はすべてのボート。
        map_object : folium.Map, optional
            使用するマップオブジェクト。Noneの場合は新規作成。
        show_labels : bool, optional
            艇名ラベルを表示するかどうか
        sync_time : bool, optional
            時間を同期して表示するかどうか
        
        Returns:
        --------
        folium.Map
            航跡が追加された地図オブジェクト
        """
        # 表示対象のボートを特定
        if boat_names is None:
            boat_names = list(self.boats_data.keys())
        
        # 存在するボートのみをフィルタリング
        valid_boat_names = [name for name in boat_names if name in self.boats_data]
        
        if not valid_boat_names:
            print("表示するボートのデータがありません")
            return None
        
        # 地図オブジェクトがない場合は作成
        if map_object is None and self.map_object is None:
            map_object = self.create_base_map()
        elif map_object is None:
            map_object = self.map_object
        
        # 時間同期が必要な場合
        if sync_time and len(valid_boat_names) > 1:
            # 共通の時間軸を作成（すべてのボートデータの時間範囲をカバー）
            all_times = []
            for name in valid_boat_names:
                if 'timestamp' in self.boats_data[name].columns:
                    all_times.extend(self.boats_data[name]['timestamp'].tolist())
            
            if all_times:
                min_time = min(all_times)
                max_time = max(all_times)
                # 1秒間隔の時間軸を作成
                reference_times = pd.date_range(min_time, max_time, freq='1S')
                
                # visualization_utilsの関数を使用してデータを同期
                from .visualization_utils import synchronize_boat_data
                synced_data = synchronize_boat_data(
                    {name: self.boats_data[name] for name in valid_boat_names},
                    reference_times
                )
                
                # 同期データを使って各ボートを表示
                for i, boat_name in enumerate(valid_boat_names):
                    color = self.colors[i % len(self.colors)]
                    boat_data = synced_data[boat_name]
                    
                    # 航跡の描画
                    points = list(zip(boat_data['latitude'], boat_data['longitude']))
                    folium.PolyLine(
                        points,
                        color=color,
                        weight=3,
                        opacity=0.8,
                        tooltip=boat_name
                    ).add_to(map_object)
                    
                    # スタート位置とゴール位置にマーカーを追加
                    start_point = points[0]
                    folium.Marker(
                        location=start_point,
                        popup=f"{boat_name} スタート",
                        icon=folium.Icon(color='green', icon='play', prefix='fa')
                    ).add_to(map_object)
                    
                    end_point = points[-1]
                    folium.Marker(
                        location=end_point,
                        popup=f"{boat_name} フィニッシュ",
                        icon=folium.Icon(color='red', icon='stop', prefix='fa')
                    ).add_to(map_object)
                    
                    # 艇名ラベルを表示（オプション）
                    if show_labels:
                        # 中間点にラベルを表示
                        mid_idx = len(points) // 2
                        mid_point = points[mid_idx]
                        folium.Marker(
                            location=mid_point,
                            icon=folium.DivIcon(
                                icon_size=(150, 36),
                                icon_anchor=(75, 0),
                                html=f'<div style="background-color: {color}; color: white; padding: 3px; border-radius: 3px;">{boat_name}</div>'
                            )
                        ).add_to(map_object)
            else:
                # タイムスタンプがない場合は通常の表示
                for i, boat_name in enumerate(valid_boat_names):
                    color = self.colors[i % len(self.colors)]
                    self.visualize_single_boat(boat_name, map_object, color)
                    
                    # ラベル表示の対応は後で追加
        else:
            # 時間同期なしの通常表示
            for i, boat_name in enumerate(valid_boat_names):
                color = self.colors[i % len(self.colors)]
                self.visualize_single_boat(boat_name, map_object, color)
                
                # 艇名ラベルを表示（オプション）
                if show_labels:
                    boat_data = self.boats_data[boat_name]
                    points = list(zip(boat_data['latitude'], boat_data['longitude']))
                    
                    # 中間点にラベルを表示
                    mid_idx = len(points) // 2
                    mid_point = points[mid_idx]
                    folium.Marker(
                        location=mid_point,
                        icon=folium.DivIcon(
                            icon_size=(150, 36),
                            icon_anchor=(75, 0),
                            html=f'<div style="background-color: {color}; color: white; padding: 3px; border-radius: 3px;">{boat_name}</div>'
                        )
                    ).add_to(map_object)
        
        # コース情報が存在する場合は表示
        if self.course_data is not None:
            self._add_course_markers(map_object)
        
        return map_object
    
    def create_performance_summary(self, boat_name):
        """
        ボートのパフォーマンス指標のサマリーを作成します
        
        Parameters:
        -----------
        boat_name : str
            表示するボートの名前
        
        Returns:
        --------
        dict
            パフォーマンス指標のサマリー
        """
        if boat_name not in self.boats_data:
            print(f"ボート '{boat_name}' のデータがありません")
            return None
        
        boat_data = self.boats_data[boat_name]
        
        # visualization_utilsの関数を使用して統計を計算
        from .visualization_utils import calculate_statistics
        stats = calculate_statistics(boat_data)
        
        # VMGの計算（風向と速度データがある場合）
        if 'wind_direction' in boat_data.columns and 'course' in boat_data.columns and 'speed' in boat_data.columns:
            from .visualization_utils import calculate_vmg
            boat_data['vmg'] = boat_data.apply(
                lambda row: calculate_vmg(row['speed'], row['course'], row['wind_direction']),
                axis=1
            )
            
            # VMGの統計
            vmg_stats = {
                'avg_vmg': boat_data['vmg'].mean(),
                'max_vmg': boat_data['vmg'].max(),
                'min_vmg': boat_data['vmg'].min()
            }
            stats['vmg'] = vmg_stats
        
        # タックの検出と分析
        if 'course' in boat_data.columns:
            from .visualization_utils import detect_tacks
            tacks = detect_tacks(boat_data)
            
            # タック回数
            stats['tack_count'] = len(tacks)
            
            # タック前後の速度損失分析（タックが検出された場合）
            if not tacks.empty and 'speed' in boat_data.columns and 'timestamp' in boat_data.columns:
                tack_speed_loss = []
                window_size = pd.Timedelta(seconds=30)
                
                for idx, tack in tacks.iterrows():
                    tack_time = tack['timestamp']
                    
                    # タック前後30秒のデータを取得
                    before_tack = boat_data[
                        (boat_data['timestamp'] >= tack_time - window_size) &
                        (boat_data['timestamp'] < tack_time)
                    ]
                    
                    after_tack = boat_data[
                        (boat_data['timestamp'] > tack_time) &
                        (boat_data['timestamp'] <= tack_time + window_size)
                    ]
                    
                    if not before_tack.empty and not after_tack.empty:
                        # タック前後の平均速度
                        before_speed = before_tack['speed'].mean()
                        after_speed = after_tack['speed'].mean()
                        
                        # 速度損失の計算
                        speed_loss = before_speed - after_speed
                        speed_loss_percent = (speed_loss / before_speed) * 100 if before_speed > 0 else 0
                        
                        tack_speed_loss.append({
                            'time': tack_time,
                            'before_speed': before_speed,
                            'after_speed': after_speed,
                            'loss_knots': speed_loss,
                            'loss_percent': speed_loss_percent
                        })
                
                if tack_speed_loss:
                    # タック損失の平均を計算
                    avg_loss_knots = sum(item['loss_knots'] for item in tack_speed_loss) / len(tack_speed_loss)
                    avg_loss_percent = sum(item['loss_percent'] for item in tack_speed_loss) / len(tack_speed_loss)
                    
                    stats['tack_analysis'] = {
                        'avg_loss_knots': avg_loss_knots,
                        'avg_loss_percent': avg_loss_percent,
                        'details': tack_speed_loss
                    }
        
        return stats

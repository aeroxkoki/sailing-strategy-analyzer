"""
セーリング戦略分析システム - パフォーマンスグラフ表示モジュール

このモジュールは、Plotlyを使用してセーリングデータのパフォーマンスグラフを
表示する機能を提供します。速度ポーラー図、タイムライン表示などを含みます。

作成日: 2025-03-05
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math


class SailingPerformancePlots:
    """
    Plotlyを使用してセーリングデータのパフォーマンスグラフを表示するクラス
    """
    
    def __init__(self):
        """
        SailingPerformancePlotsクラスの初期化
        """
        self.color_sequence = px.colors.qualitative.Plotly  # デフォルトの色シーケンス
        self.template = "plotly_white"  # デフォルトのテンプレート
    
    def create_speed_polar_plot(self, data, boat_name=None, bin_size=10):
        """
        ボートの風向に対する速度をポーラーチャートで表示します
        
        Parameters:
        -----------
        data : pandas.DataFrame
            風向と速度を含むデータフレーム
        boat_name : str, optional
            ボートの名前（タイトルに表示）
        bin_size : int, optional
            風向のビンサイズ（度）
            
        Returns:
        --------
        plotly.graph_objects.Figure
            ポーラーチャートの図オブジェクト
        """
        if 'wind_direction' not in data.columns or 'speed' not in data.columns:
            raise ValueError("データには 'wind_direction' と 'speed' 列が必要です")
        
        # ボート相対風向の計算（コースと風向の差）
        if 'course' in data.columns:
            data['relative_wind'] = (data['wind_direction'] - data['course']) % 360
        else:
            data['relative_wind'] = data['wind_direction']
        
        # 風向をビンに分類
        bins = list(range(0, 361, bin_size))
        labels = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        data['wind_bin'] = pd.cut(data['relative_wind'], bins=bins, labels=labels, include_lowest=True)
        
        # 各ビンごとの平均速度を計算
        polar_data = data.groupby('wind_bin')['speed'].mean().reset_index()
        
        # ポーラーチャートの作成
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=polar_data['speed'],
            theta=polar_data['wind_bin'],
            mode='lines+markers',
            name=boat_name if boat_name else 'ボート速度',
            line=dict(color=self.color_sequence[0], width=2),
            marker=dict(size=8)
        ))
        
        # レイアウト設定
        title = f"{boat_name} 速度ポーラーチャート" if boat_name else "ボート速度ポーラーチャート"
        fig.update_layout(
            title=title,
            template=self.template,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(data['speed']) * 1.1]
                ),
                angularaxis=dict(
                    tickmode='array',
                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                    ticktext=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
                    direction='clockwise'
                )
            ),
            showlegend=True
        )
        
        return fig
    
    def create_speed_vs_time_plot(self, data, boat_name=None):
        """
        時間に対する速度のプロットを作成します
        
        Parameters:
        -----------
        data : pandas.DataFrame
            タイムスタンプと速度を含むデータフレーム
        boat_name : str, optional
            ボートの名前
            
        Returns:
        --------
        plotly.graph_objects.Figure
            時系列グラフの図オブジェクト
        """
        if 'timestamp' not in data.columns or 'speed' not in data.columns:
            raise ValueError("データには 'timestamp' と 'speed' 列が必要です")
        
        # 時系列グラフの作成
        fig = px.line(
            data, 
            x='timestamp', 
            y='speed',
            labels={'timestamp': '時間', 'speed': '速度 (ノット)'},
            title=f"{boat_name} 速度変化" if boat_name else "ボート速度変化"
        )
        
        # レイアウト設定
        fig.update_layout(
            template=self.template,
            xaxis_title="時間",
            yaxis_title="速度 (ノット)",
            hovermode="x unified"
        )
        
        # ホバー情報の設定
        fig.update_traces(
            hovertemplate="<b>時間</b>: %{x}<br><b>速度</b>: %{y:.2f} ノット"
        )
        
        return fig
    
    def create_multi_boat_speed_comparison(self, data_dict):
        """
        複数のボートの速度を比較するプロットを作成します
        
        Parameters:
        -----------
        data_dict : dict
            ボート名をキー、データフレームを値とする辞書
            
        Returns:
        --------
        plotly.graph_objects.Figure
            複数ボート比較グラフの図オブジェクト
        """
        fig = go.Figure()
        
        for i, (boat_name, data) in enumerate(data_dict.items()):
            if 'timestamp' not in data.columns or 'speed' not in data.columns:
                print(f"警告: {boat_name} のデータには必要な列がありません。スキップします。")
                continue
            
            color = self.color_sequence[i % len(self.color_sequence)]
            
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['speed'],
                mode='lines',
                name=boat_name,
                line=dict(color=color, width=2)
            ))
        
        # レイアウト設定
        fig.update_layout(
            title="ボート速度比較",
            template=self.template,
            xaxis_title="時間",
            yaxis_title="速度 (ノット)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_performance_dashboard(self, data, boat_name=None):
        """
        パフォーマンスダッシュボードを作成します
        
        Parameters:
        -----------
        data : pandas.DataFrame
            セーリングデータを含むデータフレーム
        boat_name : str, optional
            ボートの名前
            
        Returns:
        --------
        plotly.graph_objects.Figure
            ダッシュボードの図オブジェクト
        """
        # 必要な列の確認
        required_columns = ['timestamp', 'speed', 'wind_direction']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"データには次の列が必要です: {', '.join(missing_columns)}")
        
        # サブプロットの作成
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "xy"}, {"type": "polar"}],
                [{"type": "xy", "colspan": 2}, None]
            ],
            subplot_titles=(
                "速度変化", "速度ポーラー",
                "風向と速度の関係"
            )
        )
        
        # 1. 速度の時系列プロット（左上）
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['speed'],
                mode='lines',
                name='速度',
                line=dict(color=self.color_sequence[0], width=2)
            ),
            row=1, col=1
        )
        
        # 2. 速度ポーラーチャート（右上）
        # ボート相対風向の計算（コースと風向の差）
        if 'course' in data.columns:
            data['relative_wind'] = (data['wind_direction'] - data['course']) % 360
        else:
            data['relative_wind'] = data['wind_direction']
        
        # 風向をビンに分類
        bin_size = 10
        bins = list(range(0, 361, bin_size))
        labels = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        data['wind_bin'] = pd.cut(data['relative_wind'], bins=bins, labels=labels, include_lowest=True)
        
        # 各ビンごとの平均速度を計算
        polar_data = data.groupby('wind_bin')['speed'].mean().reset_index()
        
        fig.add_trace(
            go.Scatterpolar(
                r=polar_data['speed'],
                theta=polar_data['wind_bin'],
                mode='lines+markers',
                name='ポーラー速度',
                line=dict(color=self.color_sequence[1], width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # 3. 風向と速度の関係（下段全体）
        fig.add_trace(
            go.Scatter(
                x=data['wind_direction'],
                y=data['speed'],
                mode='markers',
                name='風向vs速度',
                marker=dict(
                    color=self.color_sequence[2],
                    size=8,
                    opacity=0.6
                )
            ),
            row=2, col=1
        )
        
        # レイアウト設定
        title = f"{boat_name} パフォーマンスダッシュボード" if boat_name else "セーリングパフォーマンスダッシュボード"
        fig.update_layout(
            title_text=title,
            template=self.template,
            height=800,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(data['speed']) * 1.1]
                ),
                angularaxis=dict(
                    tickmode='array',
                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                    ticktext=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
                    direction='clockwise'
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # X軸とY軸のラベル設定
        fig.update_xaxes(title_text="時間", row=1, col=1)
        fig.update_yaxes(title_text="速度 (ノット)", row=1, col=1)
        
        fig.update_xaxes(title_text="風向 (度)", row=2, col=1)
        fig.update_yaxes(title_text="速度 (ノット)", row=2, col=1)
        
        return fig
    
    def create_tack_analysis_plot(self, data, boat_name=None):
        """
        タック（方向転換）の分析プロットを作成します
        
        Parameters:
        -----------
        data : pandas.DataFrame
            コース、速度、タイムスタンプを含むデータフレーム
        boat_name : str, optional
            ボートの名前
            
        Returns:
        --------
        plotly.graph_objects.Figure
            タック分析の図オブジェクト
        """
        # 必要な列の確認
        required_columns = ['timestamp', 'speed', 'course']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"データには次の列が必要です: {', '.join(missing_columns)}")
        
        # タックの検出（コースの大きな変化を検出）
        data = data.copy()  # 元のデータを変更しないためにコピー
        data['course_diff'] = data['course'].diff().abs()
        
        # コース差が90度以上の場合をタックとして検出
        tack_threshold = 90
        tacks = data[data['course_diff'] > tack_threshold].copy()
        
        if len(tacks) == 0:
            print("タックが検出されませんでした")
            return self.create_speed_vs_time_plot(data, boat_name)
        
        # タックの前後のデータを抽出（タックの前後30秒間）
        tack_windows = []
        window_size = pd.Timedelta(seconds=30)
        
        for idx, tack in tacks.iterrows():
            tack_time = tack['timestamp']
            window_start = tack_time - window_size
            window_end = tack_time + window_size
            
            window_data = data[
                (data['timestamp'] >= window_start) & 
                (data['timestamp'] <= window_end)
            ].copy()
            
            if not window_data.empty:
                # 相対時間の計算（タック時点を0秒とする）
                window_data['relative_time'] = (window_data['timestamp'] - tack_time).dt.total_seconds()
                window_data['tack_id'] = idx
                tack_windows.append(window_data)
        
        if not tack_windows:
            print("タック前後のデータが抽出できませんでした")
            return self.create_speed_vs_time_plot(data, boat_name)
        
        # 全タックデータの結合
        all_tack_data = pd.concat(tack_windows)
        
        # タック分析プロットの作成
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("タック前後の速度変化", "タック前後のコース変化")
        )
        
        # 各タックごとにプロット
        for tack_id, group in all_tack_data.groupby('tack_id'):
            color_idx = hash(str(tack_id)) % len(self.color_sequence)
            color = self.color_sequence[color_idx]
            
            # 速度プロット
            fig.add_trace(
                go.Scatter(
                    x=group['relative_time'],
                    y=group['speed'],
                    mode='lines',
                    name=f'タック {tack_id}',
                    line=dict(color=color),
                    showlegend=False  # 凡例は一度だけ表示
                ),
                row=1, col=1
            )
            
            # コースプロット
            fig.add_trace(
                go.Scatter(
                    x=group['relative_time'],
                    y=group['course'],
                    mode='lines',
                    name=f'タック {tack_id}',
                    line=dict(color=color)
                ),
                row=2, col=1
            )
        
        # 平均値のプロット
        avg_data = all_tack_data.groupby('relative_time').agg({
            'speed': 'mean',
            'course': 'mean'
        }).reset_index()
        
        # 平均速度プロット
        fig.add_trace(
            go.Scatter(
                x=avg_data['relative_time'],
                y=avg_data['speed'],
                mode='lines',
                name='平均速度',
                line=dict(color='black', width=3)
            ),
            row=1, col=1
        )
        
        # タック時点に垂直線を追加
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=0, y1=1,
            yref="paper",
            xref="x",
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=0, y1=1,
            yref="paper",
            xref="x",
            line=dict(color="red", width=2, dash="dash"),
            row=2, col=1
        )
        
        # レイアウト設定
        title = f"{boat_name} タック分析" if boat_name else "タック分析"
        fig.update_layout(
            title_text=title,
            template=self.template,
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # X軸とY軸のラベル設定
        fig.update_xaxes(title_text="タックからの相対時間 (秒)", row=2, col=1)
        fig.update_yaxes(title_text="速度 (ノット)", row=1, col=1)
        fig.update_yaxes(title_text="コース (度)", row=2, col=1)
        
        return fig
    
    def create_timeline_visualization(self, data_dict, event_markers=None):
        """
        ボートの時間ベースのビジュアライゼーションを作成します
        
        Parameters:
        -----------
        data_dict : dict
            ボート名をキー、データフレームを値とする辞書
        event_markers : list of dict, optional
            イベントを示すマーカーのリスト
            例: [{'time': datetime, 'name': 'スタート', 'description': '...'}]
            
        Returns:
        --------
        plotly.graph_objects.Figure
            タイムラインの図オブジェクト
        """
        fig = make_subplots(
            rows=len(data_dict) + 1, cols=1,  # ボート数+イベント行
            shared_xaxes=True,
            row_heights=[0.2] + [0.8/len(data_dict)] * len(data_dict),
            subplot_titles=["イベント"] + list(data_dict.keys())
        )
        
        # イベントマーカーの追加（イベントがあれば）
        if event_markers:
            for event in event_markers:
                fig.add_trace(
                    go.Scatter(
                        x=[event['time']],
                        y=[0],
                        mode='markers+text',
                        text=[event['name']],
                        textposition="top center",
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='red'
                        ),
                        hoverinfo='text',
                        hovertext=event.get('description', event['name']),
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # 各ボートのデータを追加
        for i, (boat_name, data) in enumerate(data_dict.items(), start=1):
            if 'timestamp' not in data.columns or 'speed' not in data.columns:
                print(f"警告: {boat_name} のデータには必要な列がありません。スキップします。")
                continue
            
            color = self.color_sequence[i % len(self.color_sequence)]
            
            # 速度データの追加
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['speed'],
                    mode='lines',
                    name=f'{boat_name} - 速度',
                    line=dict(color=color),
                    showlegend=True
                ),
                row=i+1, col=1
            )
            
            # 風向データがあれば追加
            if 'wind_direction' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['wind_direction'],
                        mode='lines',
                        name=f'{boat_name} - 風向',
                        line=dict(color=color, dash='dot'),
                        yaxis=f'y{i+1}',
                        showlegend=True
                    ),
                    row=i+1, col=1
                )
        
        # レイアウト設定
        fig.update_layout(
            title_text="セーリングタイムライン",
            template=self.template,
            height=200 + 300 * len(data_dict),  # 高さを動的に調整
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )
        
        # Y軸のラベル設定
        fig.update_yaxes(title_text="", row=1, col=1, showticklabels=False)
        
        for i in range(len(data_dict)):
            fig.update_yaxes(title_text="値", row=i+2, col=1)
        
        # X軸のラベル設定（最後の行のみ）
        fig.update_xaxes(title_text="時間", row=len(data_dict)+1, col=1)
        
        return fig

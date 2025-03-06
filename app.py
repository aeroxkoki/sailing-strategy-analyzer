"""
セーリング戦略分析システム - Streamlitアプリケーション

このモジュールは、Streamlitを使用してセーリングデータの可視化と分析のための
Webアプリケーションを提供します。

作成日: 2025-03-05
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
from io import StringIO
import json

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# 自作モジュールのインポート
from visualization.sailing_visualizer import SailingVisualizer
from visualization.map_display import SailingMapDisplay
from visualization.performance_plots import SailingPerformancePlots
import visualization.visualization_utils as viz_utils

# ページ設定
st.set_page_config(
    page_title="セーリング戦略分析システム",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# アプリケーションのタイトル
st.title('セーリング戦略分析システム')

# セッション状態の初期化
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = SailingVisualizer()
if 'map_display' not in st.session_state:
    st.session_state.map_display = SailingMapDisplay()
if 'performance_plots' not in st.session_state:
    st.session_state.performance_plots = SailingPerformancePlots()
if 'boats_data' not in st.session_state:
    st.session_state.boats_data = {}

# サンプルデータ生成関数
def generate_sample_data():
    """テスト用のサンプルデータを生成"""
    # ボート1のデータ
    boat1_timestamps = pd.date_range(start='2025-03-01 10:00:00', periods=100, freq='10S')
    
    boat1_data = pd.DataFrame({
        'timestamp': boat1_timestamps,
        'latitude': 35.45 + np.cumsum(np.random.normal(0, 0.0001, 100)),
        'longitude': 139.65 + np.cumsum(np.random.normal(0, 0.0001, 100)),
        'speed': 5 + np.random.normal(0, 0.5, 100),
        'course': 45 + np.random.normal(0, 5, 100),
        'wind_direction': 90 + np.random.normal(0, 3, 100)
    })
    
    # ボート2のデータ
    boat2_timestamps = pd.date_range(start='2025-03-01 10:02:00', periods=100, freq='10S')
    
    boat2_data = pd.DataFrame({
        'timestamp': boat2_timestamps,
        'latitude': 35.45 + np.cumsum(np.random.normal(0, 0.00012, 100)),
        'longitude': 139.65 + np.cumsum(np.random.normal(0, 0.00012, 100)),
        'speed': 5.2 + np.random.normal(0, 0.5, 100),
        'course': 50 + np.random.normal(0, 5, 100),
        'wind_direction': 90 + np.random.normal(0, 3, 100)
    })
    
    return {'ボート1': boat1_data, 'ボート2': boat2_data}

# CSV処理関数
def process_csv(file):
    """CSVファイルを処理してデータフレームに変換"""
    try:
        # CSVファイルの読み込み
        df = pd.read_csv(file)
        
        # 必須カラムの確認
        required_columns = ['latitude', 'longitude']
        for col in required_columns:
            if col not in df.columns:
                return None, f"必須カラム '{col}' がCSVファイルに見つかりません"
        
        # タイムスタンプ列の処理
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                return None, "タイムスタンプ列の形式が無効です"
        else:
            # タイムスタンプがない場合はインデックスから生成
            df['timestamp'] = pd.date_range(start='2025-03-01', periods=len(df), freq='10S')
        
        return df, None
        
    except Exception as e:
        return None, f"CSVファイルの処理中にエラーが発生しました: {str(e)}"

# サイドバーのナビゲーション
page = st.sidebar.selectbox(
    'ページ選択',
    ['ホーム', 'データアップロード', 'マップビュー', 'パフォーマンス分析', 'ヘルプ']
)

# ページの表示
if page == 'ホーム':
    st.header('セーリング戦略分析システムへようこそ')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        ### セーリング戦略分析システムとは
        
        このアプリケーションは、セーリング競技のGPSデータを分析し、戦略立案をサポートするためのツールです。
        舵手や監督がレース後にパフォーマンスを分析したり、次のレースの戦略を検討するために使用できます。
        
        ### 主な機能
        
        - **航跡の可視化**: GPSデータから航跡を地図上に表示
        - **複数艇の比較**: 異なる艇の航跡やパフォーマンスを比較
        - **時間同期表示**: 異なる時間帯のデータを同期して表示
        - **パフォーマンス分析**: 速度、風向、VMGなどの分析グラフ
        
        ### 使い方
        
        1. 「データアップロード」ページでGPSデータを読み込む
        2. 「マップビュー」ページで航跡を地図上に表示する
        3. 「パフォーマンス分析」ページで詳細な分析を行う
        """)
    
    with col2:
        st.write("### クイックスタート")
        if st.button('サンプルデータをロード'):
            with st.spinner('サンプルデータを生成中...'):
                sample_data = generate_sample_data()
                for boat_name, data in sample_data.items():
                    st.session_state.boats_data[boat_name] = data
                    st.session_state.visualizer.boats_data[boat_name] = data
                st.success('サンプルデータを読み込みました！')
                st.info('「マップビュー」ページに移動して、データを表示できます')
        
        # 読み込み済みのデータ一覧
        if st.session_state.boats_data:
            st.write("### 読み込み済みのデータ")
            for boat, data in st.session_state.boats_data.items():
                st.write(f"- {boat}: {len(data)} データポイント")

elif page == 'データアップロード':
    st.header('データアップロード')
    
    upload_tab, manage_tab = st.tabs(["データのアップロード", "データの管理"])
    
    with upload_tab:
        # ファイルアップロード
        uploaded_file = st.file_uploader("GPXまたはCSVファイルをアップロードしてください", type=['gpx', 'csv'])
        
        if uploaded_file is not None:
            # ファイル種別を判定
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # ボート名入力
            boat_name = st.text_input('ボート名を入力してください:', uploaded_file.name.split('.')[0])
            
            if st.button('処理開始'):
                with st.spinner('データを処理中...'):
                    try:
                        if file_extension == 'csv':
                            df, error = process_csv(uploaded_file)
                            if error:
                                st.error(error)
                                st.stop()
                        else:
                            # GPXファイルの処理（データ処理モジュールが必要）
                            st.error('GPXファイルの処理モジュールがまだ実装されていません')
                            st.stop()
                        
                        # セッションにデータを保存
                        st.session_state.boats_data[boat_name] = df
                        
                        # 可視化クラスにデータをロード
                        st.session_state.visualizer.boats_data[boat_name] = df
                        
                        st.success(f'{boat_name} のデータを正常に読み込みました！')
                        
                        # データプレビュー表示
                        st.subheader('データプレビュー')
                        st.dataframe(df.head())
                        
                        # 基本統計情報
                        st.subheader('基本統計情報')
                        if 'speed' in df.columns:
                            st.write(f"平均速度: {df['speed'].mean():.2f} ノット")
                        if 'timestamp' in df.columns:
                            st.write(f"データ期間: {df['timestamp'].min()} から {df['timestamp'].max()}")
                        st.write(f"データポイント数: {len(df)}")
                        
                    except Exception as e:
                        st.error(f'エラーが発生しました: {e}')
    
    with manage_tab:
        st.subheader('読み込み済みのボートデータ')
        
        if not st.session_state.boats_data:
            st.info('データがまだアップロードされていません')
        else:
            for boat_name, data in st.session_state.boats_data.items():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(f"**{boat_name}** ({len(data)} データポイント)")
                
                with col2:
                    if st.button(f"プレビュー", key=f"preview_{boat_name}"):
                        st.dataframe(data.head())
                
                with col3:
                    if st.button(f"情報", key=f"info_{boat_name}"):
                        st.write(f"列: {', '.join(data.columns)}")
                        if 'timestamp' in data.columns:
                            st.write(f"期間: {data['timestamp'].min()} - {data['timestamp'].max()}")
                
                with col4:
                    if st.button(f"削除", key=f"delete_{boat_name}"):
                        del st.session_state.boats_data[boat_name]
                        del st.session_state.visualizer.boats_data[boat_name]
                        st.success(f"{boat_name} のデータを削除しました")
                        st.experimental_rerun()
            
            if st.button("すべてのデータを削除"):
                st.session_state.boats_data = {}
                st.session_state.visualizer.boats_data = {}
                st.success("すべてのデータを削除しました")
                st.experimental_rerun()

elif page == 'マップビュー':
    st.header('航跡マップ')
    
    if not st.session_state.boats_data:
        st.warning('データがまだアップロードされていません。「データアップロード」ページでデータを追加してください。')
        if st.button('サンプルデータを生成してテスト'):
            with st.spinner('サンプルデータを生成中...'):
                sample_data = generate_sample_data()
                for boat_name, data in sample_data.items():
                    st.session_state.boats_data[boat_name] = data
                    st.session_state.visualizer.boats_data[boat_name] = data
                st.success('サンプルデータを読み込みました！')
                st.experimental_rerun()
        st.stop()
    
    # 表示オプションと艇選択を並べて表示
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader('表示オプション')
        # マップ表示オプション
        map_tile = st.selectbox('地図スタイル', 
                               options=list(st.session_state.map_display.available_tiles.keys()))
        
        show_labels = st.checkbox('艇名ラベルを表示', value=True)
        show_course = st.checkbox('コース情報を表示', value=True)
        sync_time = st.checkbox('時間を同期して表示', value=False)
    
    with col2:
        # 表示するボートの選択
        st.subheader('表示する艇の選択')
        boat_options = list(st.session_state.boats_data.keys())
        selected_boats = st.multiselect('表示するボートを選択:', boat_options, default=boat_options)
    
    # マップの作成と表示
    if st.button('マップを表示', type="primary"):
        if not selected_boats:
            st.warning('表示するボートを少なくとも1つ選択してください')
        else:
            with st.spinner('マップを生成中...'):
                try:
                    # マップオブジェクトの作成
                    map_display = st.session_state.map_display
                    map_object = map_display.create_map(tile=map_tile)
                    
                    # 複数艇表示機能を使用
                    map_object = st.session_state.visualizer.visualize_multiple_boats(
                        boat_names=selected_boats,
                        map_object=map_object,
                        show_labels=show_labels, 
                        sync_time=sync_time
                    )
                    
                    # 地図を表示
                    st.subheader('セーリング航跡マップ')
                    folium_static(map_object, width=1000)
                    
                    # マップをHTMLファイルとして保存するオプション
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button('マップをHTMLファイルとして保存'):
                            map_display.save_map('sailing_map.html')
                            st.success('マップを保存しました: sailing_map.html')
                    
                    with col2:
                        # ダウンロードボタンの代わりに説明を表示
                        st.info('HTMLファイルはサーバー上に保存されました。実際の環境ではダウンロードボタンが表示されます。')
                
                except Exception as e:
                    st.error(f'マップ生成中にエラーが発生しました: {e}')

elif page == 'パフォーマンス分析':
    st.header('パフォーマンス分析')
    
    if not st.session_state.boats_data:
        st.warning('データがまだアップロードされていません。「データアップロード」ページでデータを追加してください。')
        if st.button('サンプルデータを生成してテスト'):
            with st.spinner('サンプルデータを生成中...'):
                sample_data = generate_sample_data()
                for boat_name, data in sample_data.items():
                    st.session_state.boats_data[boat_name] = data
                    st.session_state.visualizer.boats_data[boat_name] = data
                st.success('サンプルデータを読み込みました！')
                st.experimental_rerun()
        st.stop()
    
    tabs = st.tabs(["単艇分析", "複数艇比較", "パフォーマンスサマリー"])
    
    with tabs[0]:
        st.subheader('単艇分析')
        
        # 分析するボートの選択
        boat_options = list(st.session_state.boats_data.keys())
        selected_boat = st.selectbox('分析するボートを選択:', boat_options)
        
        # グラフの種類を選択
        plot_type = st.selectbox(
            'グラフの種類:',
            ['速度の時系列', '速度ポーラーチャート', 'タック分析', 'パフォーマンスダッシュボード']
        )
        
        # 選択したグラフを表示
        if st.button('グラフを生成', key='single_boat_graph'):
            with st.spinner('グラフを生成中...'):
                try:
                    performance_plots = st.session_state.performance_plots
                    boat_data = st.session_state.boats_data[selected_boat]
                    
                    if plot_type == '速度の時系列':
                        fig = performance_plots.create_speed_vs_time_plot(boat_data, selected_boat)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif plot_type == '速度ポーラーチャート':
                        if 'wind_direction' not in boat_data.columns or 'speed' not in boat_data.columns:
                            st.error('このグラフには風向と速度のデータが必要です')
                        else:
                            fig = performance_plots.create_speed_polar_plot(boat_data, selected_boat)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif plot_type == 'タック分析':
                        if 'course' not in boat_data.columns or 'speed' not in boat_data.columns:
                            st.error('このグラフにはコースと速度のデータが必要です')
                        else:
                            fig = performance_plots.create_tack_analysis_plot(boat_data, selected_boat)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif plot_type == 'パフォーマンスダッシュボード':
                        # 必要なカラムのチェック
                        missing_columns = []
                        for col in ['timestamp', 'speed', 'wind_direction']:
                            if col not in boat_data.columns:
                                missing_columns.append(col)
                        
                        if missing_columns:
                            st.error(f'このグラフには次の列が必要です: {", ".join(missing_columns)}')
                        else:
                            fig = performance_plots.create_performance_dashboard(boat_data, selected_boat)
                            st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f'グラフ生成中にエラーが発生しました: {e}')
    
    with tabs[1]:
        st.subheader('複数艇比較')
        
        # 比較するボートを選択
        comparison_boats = st.multiselect('比較するボートを選択:', boat_options, default=boat_options[:2] if len(boat_options) >= 2 else [])
        
        if comparison_boats and st.button('比較グラフを生成'):
            with st.spinner('比較グラフを生成中...'):
                try:
                    # 選択したボートのデータを辞書に格納
                    data_dict = {}
                    for boat in comparison_boats:
                        data_dict[boat] = st.session_state.boats_data[boat]
                    
                    # 複数ボート比較グラフの生成
                    fig = st.session_state.performance_plots.create_multi_boat_speed_comparison(data_dict)
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f'比較グラフ生成中にエラーが発生しました: {e}')
    
    with tabs[2]:
        st.subheader('パフォーマンスサマリー')
        
        # サマリーを表示するボートの選択
        summary_boat = st.selectbox('サマリーを表示するボートを選択:', boat_options, key='summary_boat')
        
        if st.button('パフォーマンスサマリーを生成'):
            with st.spinner('サマリーを生成中...'):
                try:
                    # パフォーマンスサマリーの取得
                    summary = st.session_state.visualizer.create_performance_summary(summary_boat)
                    
                    if summary:
                        # サマリー情報を表示
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("### 基本統計")
                            if 'speed' in summary:
                                st.write("#### 速度情報")
                                speed_stats = summary['speed']
                                st.write(f"最高速度: {speed_stats['max']:.2f} ノット")
                                st.write(f"平均速度: {speed_stats['avg']:.2f} ノット")
                                st.write(f"最低速度: {speed_stats['min']:.2f} ノット")
                            
                            if 'total_distance_nm' in summary:
                                st.write(f"走行距離: {summary['total_distance_nm']:.2f} 海里")
                            
                            if 'duration_seconds' in summary:
                                duration_min = summary['duration_seconds'] / 60
                                st.write(f"走行時間: {duration_min:.1f} 分")
                        
                        with col2:
                            if 'tack_count' in summary:
                                st.write("### タック分析")
                                st.write(f"タック回数: {summary['tack_count']}")
                                
                                if 'tack_analysis' in summary:
                                    tack_analysis = summary['tack_analysis']
                                    st.write(f"平均速度損失: {tack_analysis['avg_loss_knots']:.2f} ノット ({tack_analysis['avg_loss_percent']:.1f}%)")
                            
                            if 'vmg' in summary:
                                st.write("### VMG分析")
                                vmg = summary['vmg']
                                st.write(f"平均VMG: {vmg['avg_vmg']:.2f}")
                                st.write(f"最大VMG: {vmg['max_vmg']:.2f}")
                    else:
                        st.warning(f"{summary_boat} のパフォーマンスサマリーを生成できませんでした")
                
                except Exception as e:
                    st.error(f'サマリー生成中にエラーが発生しました: {e}')

elif page == 'ヘルプ':
    st.header('ヘルプ & 使用方法')
    
    help_topics = [
        "基本的な使い方", 
        "データ形式について", 
        "マップ表示機能", 
        "パフォーマンス分析機能", 
        "よくある問題"
    ]
    
    selected_topic = st.selectbox("トピックを選択", help_topics)
    
    if selected_topic == "基本的な使い方":
        st.write("""
        ## 基本的な使い方
        
        ### 1. データのアップロード
        * 「データアップロード」ページでGPXまたはCSVファイルをアップロードします
        * ボート名を入力して「処理開始」ボタンをクリックします
        * 複数のボートデータをアップロードできます
        
        ### 2. マップビュー
        * 「マップビュー」ページで航跡を地図上に表示します
        * 表示するボートと表示オプションを選択できます
        * 作成した地図はHTMLファイルとして保存できます
        
        ### 3. パフォーマンス分析
        * 「パフォーマンス分析」ページでさまざまなグラフを生成します
        * 速度の時系列、ポーラーチャート、タック分析などのグラフが利用可能です
        * 複数のボートデータを比較することもできます
        """)
    
    elif selected_topic == "データ形式について":
        st.write("""
        ## データ形式について
        
        ### CSVファイルの形式
        CSVファイルには少なくとも以下の列が必要です:
        
        * `latitude`: 緯度（度）
        * `longitude`: 経度（度）
        
        以下の列があるとより詳細な分析が可能になります:
        
        * `timestamp`: 時刻（日時形式）
        * `speed`: 速度（ノット）
        * `course`: 進行方向（度）
        * `wind_direction`: 風向（度）
        
        ### GPXファイル
        GPXファイルは標準的なGPSトラックデータ形式で、多くのGPSデバイスやアプリケーションから出力できます。
        
        ### サンプルCSVファイル
        ```
        timestamp,latitude,longitude,speed,course,wind_direction
        2025-03-01 10:00:00,35.45,139.65,5.1,45,90
        2025-03-01 10:00:10,35.4501,139.6502,5.2,46,90
        2025-03-01 10:00:20,35.4502,139.6504,5.3,47,91
        ```
        """)
    
    elif selected_topic == "マップ表示機能":
        st.write("""
        ## マップ表示機能
        
        ### 地図スタイル
        * ポジトロン: 明るい背景の地図（デフォルト）
        * オープンストリートマップ: 標準的な地図
        * 衛星写真: 衛星画像
        * 地形図: 白黒の地図
        * 暗い背景: ダークモードの地図
        
        ### 表示オプション
        * 艇名ラベルを表示: 各艇の軌跡上に艇名を表示します
        * コース情報を表示: コースマークなどの情報を表示します（コースデータがある場合）
        * 時間を同期して表示: 複数艇のデータを時間軸で同期します
        
        ### マップの操作
        * マウスのホイールでズームイン/アウト
        * マウスのドラッグで地図を移動
        * マーカーやラインにカーソルを合わせると情報が表示されます
        """)
    
    elif selected_topic == "パフォーマンス分析機能":
        st.write("""
        ## パフォーマンス分析機能
        
        ### 単艇分析
        * 速度の時系列: 時間に対する速度の変化を表示
        * 速度ポーラーチャート: 風向に対する速度の分布を表示
        * タック分析: タック（方向転換）前後の速度変化を分析
        * パフォーマンスダッシュボード: 複数の分析グラフを一画面に表示
        
        ### 複数艇比較
        複数のボートの速度や他のパラメータを時系列で比較できます。
        
        ### パフォーマンスサマリー
        速度統計、タック分析、VMG（Velocity Made Good）など、パフォーマンスの要約情報を表示します。
        """)
    
    elif selected_topic == "よくある問題":
        st.write("""
        ## よくある問題
        
        ### データが読み込めない
        * ファイル形式が正しいことを確認してください（CSV or GPX）
        * CSVファイルの場合、必須カラム（latitude, longitude）が含まれていることを確認してください
        * データの区切り文字がカンマであることを確認してください
        
        ### グラフが生成できない
        * 選択したグラフの種類に必要なデータ列があることを確認してください
        * 例えば、「速度ポーラーチャート」には speed と wind_direction の列が必要です
        
        ### 地図が表示されない
        * 少なくとも1つのボートを選択していることを確認してください
        * データに有効な緯度・経度が含まれていることを確認してください

　　　　　### パフォーマンスが悪い
        * 大量のデータポイントがある場合は、データの間引きを検討してください
        * 複雑なグラフは生成に時間がかかることがあります
        
        ### その他の問題
        技術的な問題やバグがある場合は、開発チームにお問い合わせください。
        """)

# フッター
st.sidebar.markdown('---')
st.sidebar.info('セーリング戦略分析システム v1.0')
st.sidebar.write('© 2025 Sailing Analytics Project')

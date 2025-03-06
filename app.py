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

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 自作モジュールのインポート
from visualization.sailing_visualizer import SailingVisualizer
from visualization.map_display import SailingMapDisplay
from visualization.performance_plots import SailingPerformancePlots
import visualization.visualization_utils as viz_utils

# アプリケーションのタイトル
st.title('セーリング戦略分析システム')

# サイドバーのナビゲーション
page = st.sidebar.selectbox(
    'ページ選択',
    ['ホーム', 'データアップロード', 'マップビュー', 'パフォーマンス分析', 'ヘルプ']
)

# セッション状態の初期化
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = SailingVisualizer()
if 'map_display' not in st.session_state:
    st.session_state.map_display = SailingMapDisplay()
if 'performance_plots' not in st.session_state:
    st.session_state.performance_plots = SailingPerformancePlots()
if 'boats_data' not in st.session_state:
    st.session_state.boats_data = {}

# ページの表示
if page == 'ホーム':
    st.header('セーリング戦略分析システムへようこそ')
    st.write("""
    このアプリケーションは、セーリング競技のGPSデータを分析し、戦略立案をサポートするためのツールです。
    
    主な機能:
    - GPSデータのアップロードと処理
    - 航跡の地図表示
    - パフォーマンス指標の可視化
    - 複数艇の比較分析
    
    左側のサイドバーからページを選択して、分析を開始しましょう。
    """)
    
    # サンプル画像の代わりにテキスト表示
    st.info('Week 2: 複数艇表示機能とStreamlit統合中')

elif page == 'データアップロード':
    st.header('データアップロード')
    
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
                    # 実際のデータ処理はここに実装
                    # （データ処理モジュールが必要）
                    
                    # 仮のデータフレーム作成（実際には処理されたデータを使用）
                    if file_extension == 'csv':
                        df = pd.read_csv(uploaded_file)
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
    
    # 読み込み済みのデータ一覧
    st.subheader('読み込み済みのボートデータ')
    
    if st.session_state.boats_data:
        for boat, data in st.session_state.boats_data.items():
            st.write(f"- {boat}: {len(data)} データポイント")
    else:
        st.info('データがまだアップロードされていません')

elif page == 'マップビュー':
    st.header('航跡マップ')
    
    if not st.session_state.boats_data:
        st.warning('データがまだアップロードされていません。「データアップロード」ページでデータを追加してください。')
        st.stop()
    
    # 表示するボートの選択
    boat_options = list(st.session_state.boats_data.keys())
    selected_boats = st.multiselect('表示するボートを選択:', boat_options, default=boat_options)
    
    # マップ表示オプション
    st.subheader('表示オプション')
    col1, col2 = st.columns(2)
    
    with col1:
        show_labels = st.checkbox('艇名ラベルを表示', value=True)
        show_course = st.checkbox('コース情報を表示', value=True)
    
    with col2:
        sync_time = st.checkbox('時間を同期して表示', value=False)
        map_tile = st.selectbox('地図スタイル', options=list(st.session_state.map_display.available_tiles.keys()))
    
    # マップの作成と表示
    if st.button('マップを表示'):
        with st.spinner('マップを生成中...'):
            try:
                # マップオブジェクトの作成
                map_display = st.session_state.map_display
                map_object = map_display.create_map(tile=map_tile)
                
                # 選択したボートの表示
                for i, boat_name in enumerate(selected_boats):
                    color = map_display.colors[i % len(map_display.colors)]
                    boat_data = st.session_state.boats_data[boat_name]
                    map_display.add_track(boat_data, boat_name, color)
                
                # 地図を表示
                folium_static(map_object)
                
                # マップをHTMLファイルとして保存するオプション
                if st.button('マップをHTMLファイルとして保存'):
                    map_display.save_map('sailing_map.html')
                    st.success('マップを保存しました: sailing_map.html')
            
            except Exception as e:
                st.error(f'マップ生成中にエラーが発生しました: {e}')

elif page == 'パフォーマンス分析':
    st.header('パフォーマンス分析')
    
    if not st.session_state.boats_data:
        st.warning('データがまだアップロードされていません。「データアップロード」ページでデータを追加してください。')
        st.stop()
    
    # 分析するボートの選択
    boat_options = list(st.session_state.boats_data.keys())
    selected_boat = st.selectbox('分析するボートを選択:', boat_options)
    
    # グラフの種類を選択
    plot_type = st.selectbox(
        'グラフの種類:',
        ['速度の時系列', '速度ポーラーチャート', 'タック分析', 'パフォーマンスダッシュボード']
    )
    
    # 選択したグラフを表示
    if st.button('グラフを生成'):
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
    
    # 複数艇比較セクション
    st.subheader('複数艇の比較')
    comparison_boats = st.multiselect('比較するボートを選択:', boat_options, default=[])
    
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

elif page == 'ヘルプ':
    st.header('ヘルプ & 使用方法')
    st.write("""
    ## セーリング戦略分析システムの使用方法
    
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
    
    ### データ形式について
    * CSVファイルには少なくとも以下の列が必要です:
      * latitude: 緯度（度）
      * longitude: 経度（度）
      * timestamp: 時刻（日時形式）
      * speed: 速度（ノット）
    * 風向、コースなどの追加データがあるとより詳細な分析が可能です
    
    ### 問題が発生した場合
    * データ形式が正しいことを確認してください
    * 必要なカラムが揃っていることを確認してください
    * エラーメッセージを確認し、指示に従ってください
    """)

# フッター
st.sidebar.markdown('---')
st.sidebar.info('セーリング戦略分析システム v1

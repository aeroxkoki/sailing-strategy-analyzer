import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import io
import base64
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sailing_data_processor import SailingDataProcessor

def main():
    # ページ設定
    st.set_page_config(
        page_title="セーリング戦略分析システム", 
        page_icon="🌬️", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    # タイトルとイントロダクション
    st.title('セーリングGPSデータ処理モジュール')
    st.markdown("""
    このモジュールは、セーリングレースのGPSデータを処理するためのツールです。
    複数艇のGPSデータを同時に読み込み、前処理、時刻同期、異常値検出などを行います。
    """)
    
    # サイドバーにファイルアップロード機能
    st.sidebar.header('データ入力')
    
    # 複数艇データのアップロード
    uploaded_files = st.sidebar.file_uploader(
        "GPX/CSVファイルをアップロード (最大19艇)", 
        type=['gpx', 'csv'],
        accept_multiple_files=True
    )
    
    # 艇ID割り当てモード
    id_mode = st.sidebar.radio(
        "艇ID割り当てモード",
        options=["自動（ファイル名から生成）", "手動"],
        index=0
    )
    
    # 手動ID入力
    manual_ids = []
    if id_mode == "手動" and uploaded_files:
        st.sidebar.subheader("艇IDを入力")
        for i, file in enumerate(uploaded_files):
            default_id = f"Boat{i+1}"
            manual_id = st.sidebar.text_input(f"ファイル {file.name} の艇ID", value=default_id, key=f"id_{i}")
            manual_ids.append(manual_id)
    
    # データ処理オプション
    st.sidebar.header('処理オプション')
    
    # 最大速度設定（異常値検出用）
    max_speed = st.sidebar.slider(
        '最大想定速度 (ノット)', 
        min_value=10.0, 
        max_value=50.0, 
        value=30.0,
        step=0.5
    )
    
    # サンプリングレート設定
    sampling_rate = st.sidebar.selectbox(
        "リサンプリング頻度",
        options=["1秒", "2秒", "5秒", "10秒"],
        index=0
    )
    
    # サンプリングレートを文字列に変換
    sampling_map = {
        "1秒": "1s", 
        "2秒": "2s", 
        "5秒": "5s", 
        "10秒": "10s"
    }
    sampling_freq = sampling_map[sampling_rate]
    
    # 時刻同期オプション
    sync_option = st.sidebar.radio(
        "時刻同期オプション",
        options=["自動（共通時間枠を検出）", "手動（範囲指定）"],
        index=0
    )
    
    # 手動時間範囲指定
    start_time = None
    end_time = None
    
    if sync_option == "手動（範囲指定）":
        # デフォルト値の設定
        default_start = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
        default_end = default_start + timedelta(hours=2)
        
        start_date = st.sidebar.date_input("開始日", value=default_start.date())
        start_hour = st.sidebar.time_input("開始時刻", value=default_start.time())
        end_date = st.sidebar.date_input("終了日", value=default_end.date())
        end_hour = st.sidebar.time_input("終了時刻", value=default_end.time())
        
        start_time = datetime.combine(start_date, start_hour)
        end_time = datetime.combine(end_date, end_hour)
    
    # 処理ボタン
    process_button = st.sidebar.button("データ処理開始")
    
    # データプロセッサーのインスタンス作成
    processor = SailingDataProcessor()
    
    # データ処理と結果表示
    if uploaded_files and process_button:
        with st.spinner('データ処理中...'):
            # ファイルコンテンツの準備
            file_contents = []
            for file in uploaded_files:
                filename = file.name
                content = file.read()
                filetype = filename.split('.')[-1].lower()
                file_contents.append((filename, content, filetype))
            
            # データ読み込み
            use_manual_ids = (id_mode == "手動")
            processor.load_multiple_files(file_contents, auto_id=(not use_manual_ids), manual_ids=manual_ids if use_manual_ids else None)
            
            # 全艇データのクリーニング
            processor.clean_all_boat_data(max_speed_knots=max_speed)
            
            # 時刻同期
            if sync_option == "自動（共通時間枠を検出）":
                auto_start, auto_end = processor.get_common_timeframe()
                if auto_start and auto_end:
                    processor.synchronize_time(target_freq=sampling_freq, start_time=auto_start, end_time=auto_end)
                else:
                    st.error("共通の時間枠が見つかりませんでした。データの時間範囲が重なっていない可能性があります。")
            else:
                processor.synchronize_time(target_freq=sampling_freq, start_time=start_time, end_time=end_time)
            
            # データ品質レポート
            quality_report = processor.get_data_quality_report()
        
        # 処理結果の表示
        show_processing_results(processor, quality_report)
        
    # サンプルデータの使用オプション
    use_sample_data = st.sidebar.checkbox('サンプルデータを使用', value=False)
    
    if use_sample_data and st.sidebar.button("サンプルデータを読み込む"):
        with st.spinner('サンプルデータを生成中...'):
            sample_data = generate_sample_data(3)  # 3艇分のサンプルデータ
            
            # ファイルコンテンツの準備
            file_contents = []
            for boat_id, df in sample_data.items():
                # CSVに変換
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_content = csv_buffer.getvalue().encode('utf-8')
                
                filename = f"{boat_id}.csv"
                file_contents.append((filename, csv_content, 'csv'))
            
            # データ読み込み
            processor.load_multiple_files(file_contents, auto_id=True)
            
            # データクリーニングと時刻同期
            processor.clean_all_boat_data(max_speed_knots=max_speed)
            processor.synchronize_time(target_freq=sampling_freq)
            
            # データ品質レポート
            quality_report = processor.get_data_quality_report()
        
        # 処理結果の表示
        show_processing_results(processor, quality_report)

def show_processing_results(processor, quality_report):
    """処理結果をStreamlitで表示"""
    
    st.header("データ処理結果")
    
    # タブ設定
    tab1, tab2, tab3 = st.tabs(["データ概要", "品質レポート", "可視化"])
    
    # タブ1: データ概要
    with tab1:
        # 各艇のデータ概要を表示
        for boat_id in processor.synced_data:
            st.subheader(f"艇 {boat_id}")
            
            df = processor.synced_data[boat_id]
            
            # 基本情報
            points_count = len(df)
            duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
            avg_speed = df['speed'].mean() * 1.94384  # m/s → ノット
            max_speed = df['speed'].max() * 1.94384
            
            # 情報表示
            st.write(f"データポイント数: {points_count}")
            st.write(f"計測時間: {duration:.1f} 秒 ({duration/60:.1f} 分)")
            st.write(f"平均速度: {avg_speed:.2f} ノット")
            st.write(f"最高速度: {max_speed:.2f} ノット")
            
            # データサンプルを表示
            st.write("データサンプル:")
            st.dataframe(df.head())
            
            # 区切り線
            st.markdown("---")
    
    # タブ2: 品質レポート
    with tab2:
        # 品質レポートのテーブル作成
        report_data = []
        
        for boat_id, report in quality_report.items():
            report_data.append({
                "艇ID": boat_id,
                "データ点数": report['total_points'],
                "計測時間(秒)": f"{report['duration_seconds']:.1f}",
                "サンプリングレート(Hz)": f"{report['avg_sampling_rate']:.2f}",
                "最大ギャップ(秒)": f"{report['max_gap_seconds']:.1f}",
                "ギャップ数": report['gap_count'],
                "速度異常値数": report['speed_outliers'],
                "品質スコア": f"{report['quality_score']:.1f}",
                "品質評価": report['quality_rating']
            })
        
        if report_data:
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True)
        else:
            st.warning("品質レポートがありません")
    
    # タブ3: 可視化
    with tab3:
        if processor.synced_data:
            st.subheader("GPSトラック可視化")
            
            # 地図の準備
            gps_map = visualize_gps_tracks(processor.synced_data)
            folium_static(gps_map)
            
            # 速度グラフ
            st.subheader("速度比較")
            speed_fig = plot_speed_comparison(processor.synced_data)
            st.plotly_chart(speed_fig, use_container_width=True)
        else:
            st.warning("可視化するための同期データがありません")
    
    # データダウンロード機能
    st.header("処理済みデータのダウンロード")
    
    for boat_id in processor.synced_data:
        # CSVとJSONのダウンロードボタン
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = processor.export_processed_data(boat_id, format_type='csv')
            if csv_data:
                csv_b64 = base64.b64encode(csv_data).decode()
                href = f'<a href="data:text/csv;base64,{csv_b64}" download="{boat_id}_processed.csv" class="button">CSVダウンロード</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            json_data = processor.export_processed_data(boat_id, format_type='json')
            if json_data:
                json_b64 = base64.b64encode(json_data).decode()
                href = f'<a href="data:application/json;base64,{json_b64}" download="{boat_id}_processed.json" class="button">JSONダウンロード</a>'
                st.markdown(href, unsafe_allow_html=True)

def visualize_gps_tracks(gps_data_dict):
    """GPSデータを地図上に表示する関数"""
    # 全ての艇のデータを統合
    all_lats = []
    all_lons = []
    
    for boat_id, df in gps_data_dict.items():
        all_lats.extend(df['latitude'].tolist())
        all_lons.extend(df['longitude'].tolist())
    
    # 地図の中心を計算
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    
    # 地図を作成
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # 艇ごとに異なる色で表示
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkblue', 'darkgreen', 'cadetblue', 
              'darkred', 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black',
              'lightgray', 'lightred', 'beige', 'darkgreen']
    
    for i, (boat_id, df) in enumerate(gps_data_dict.items()):
        color = colors[i % len(colors)]
        
        # 航跡を追加
        points = list(zip(df['latitude'], df['longitude']))
        folium.PolyLine(
            points,
            color=color,
            weight=3,
            opacity=0.8,
            tooltip=boat_id
        ).add_to(m)
        
        # スタート地点にマーカーを追加
        folium.Marker(
            location=[df['latitude'].iloc[0], df['longitude'].iloc[0]],
            popup=f'{boat_id} Start',
            icon=folium.Icon(color=color, icon='play'),
        ).add_to(m)
        
        # フィニッシュ地点にマーカーを追加
        folium.Marker(
            location=[df['latitude'].iloc[-1], df['longitude'].iloc[-1]],
            popup=f'{boat_id} Finish',
            icon=folium.Icon(color=color, icon='stop'),
        ).add_to(m)
    
    # タイトルを追加
    title_html = '<h3 align="center" style="font-size:16px"><b>GPS航跡</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

def plot_speed_comparison(gps_data_dict):
    """複数艇の速度を比較するプロットを作成"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, (boat_id, df) in enumerate(gps_data_dict.items()):
        color = colors[i % len(colors)]
        
        # 時間軸の正規化（最初を0とする）
        t0 = df['timestamp'].iloc[0]
        time_mins = [(t - t0).total_seconds() / 60 for t in df['timestamp']]
        
        # 速度をノットに変換
        speed_knots = df['speed'] * 1.94384
        
        # 平均速度
        avg_speed = speed_knots.mean()
        
        fig.add_trace(go.Scatter(
            x=time_mins,
            y=speed_knots,
            mode='lines',
            name=f'{boat_id} (Avg: {avg_speed:.2f} kt)',
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title='艇速度比較',
        xaxis_title='時間 (分)',
        yaxis_title='速度 (ノット)',
        legend_title='艇',
        template='plotly_white'
    )
    
    return fig

def generate_sample_data(num_boats=3):
    """テスト用のサンプルGPSデータを生成"""
    # 東京湾でのセーリングレースを想定した座標
    base_lat, base_lon = 35.620, 139.770
    
    # 各艇のデータを格納
    all_boats_data = {}
    
    for boat_id in range(1, num_boats + 1):
        # 時間間隔（秒）
        time_interval = 1  # 1秒間隔
        
        # データポイント数
        num_points = 1800  # 30分分（1秒間隔）
        
        # タイムスタンプの作成
        start_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0) - timedelta(days=1)
        start_time = start_time + timedelta(seconds=(boat_id-1)*5)  # 各艇の開始時間を少しずらす
        timestamps = [start_time + timedelta(seconds=i*time_interval) for i in range(num_points)]
        
        # 艇ごとの微小な変動を追加
        lat_var = (boat_id - 1) * 0.001
        lon_var = (boat_id - 1) * 0.002
        
        # 風上/風下のレグを含むコースを模擬
        lats = []
        lons = []
        speeds = []
        
        # 最初の風上レグ
        leg1_points = 450
        for i in range(leg1_points):
            progress = i / leg1_points
            # ジグザグパターン（タック）を追加
            phase = i % 30
            if phase < 15:
                # 左に向かうタック
                lats.append(base_lat + progress * 0.03 + 0.002 * math.sin(i/5) + lat_var)
                lons.append(base_lon + progress * 0.01 + 0.005 + lon_var)
                speeds.append(5.0 + 0.5 * math.sin(i/10) + 0.2 * (boat_id - 1))
            else:
                # 右に向かうタック
                lats.append(base_lat + progress * 0.03 + 0.002 * math.sin(i/5) + lat_var)
                lons.append(base_lon + progress * 0.01 - 0.005 + lon_var)
                speeds.append(5.2 + 0.5 * math.sin(i/10) + 0.2 * (boat_id - 1))
        
        # 風下レグ
        leg2_points = 450
        for i in range(leg2_points):
            progress = i / leg2_points
            # より直線的な動き
            lats.append(base_lat + 0.03 - progress * 0.03 + 0.001 * math.sin(i/10) + lat_var)
            lons.append(base_lon + 0.01 + 0.002 * math.cos(i/8) + lon_var)
            speeds.append(6.0 + 0.3 * math.sin(i/15) + 0.2 * (boat_id - 1))
        
        # 2回目の風上レグ
        leg3_points = 450
        for i in range(leg3_points):
            progress = i / leg3_points
            # ジグザグパターン（タック）を追加
            phase = i % 25
            if phase < 12:
                # 左に向かうタック
                lats.append(base_lat + progress * 0.03 + 0.002 * math.sin(i/6) + lat_var)
                lons.append(base_lon - 0.01 + progress * 0.02 + 0.004 + lon_var)
                speeds.append(4.8 + 0.4 * math.sin(i/12) + 0.2 * (boat_id - 1))
            else:
                # 右に向かうタック
                lats.append(base_lat + progress * 0.03 + 0.002 * math.sin(i/6) + lat_var)
                lons.append(base_lon - 0.01 + progress * 0.02 - 0.004 + lon_var)
                speeds.append(5.0 + 0.4 * math.sin(i/12) + 0.2 * (boat_id - 1))
        
        # 最終レグ
        leg4_points = 450
        for i in range(leg4_points):
            progress = i / leg4_points
            # フィニッシュに向かう
            lats.append(base_lat + 0.03 - progress * 0.02 + 0.001 * math.sin(i/7) + lat_var)
            lons.append(base_lon + 0.01 - progress * 0.01 + 0.001 * math.cos(i/9) + lon_var)
            speeds.append(5.5 + 0.3 * math.sin(i/20) + 0.2 * (boat_id - 1))
        
        # データフレーム作成
        data = {
            'timestamp': timestamps[:num_points],  # 配列の長さを合わせる
            'latitude': lats[:num_points],
            'longitude': lons[:num_points],
            'speed': np.array(speeds[:num_points]) * 0.514444,  # ノット -> m/s
            'boat_id': [f"Boat{boat_id}"] * num_points
        }
        
        df = pd.DataFrame(data)
        
        # 進行方向（ベアリング）の計算
        import math
        
        df['bearing'] = 0.0
        for i in range(1, len(df)):
            lat1, lon1 = math.radians(df.iloc[i-1]['latitude']), math.radians(df.iloc[i-1]['longitude'])
            lat2, lon2 = math.radians(df.iloc[i]['latitude']), math.radians(df.iloc[i]['longitude'])
            
            # ベアリング計算
            y = math.sin(lon2 - lon1) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
            bearing = math.degrees(math.atan2(y, x))
            
            # 0-360度の範囲に正規化
            bearing = (bearing + 360) % 360
            
            df.iloc[i, df.columns.get_loc('bearing')] = bearing
        
        # NaN値を処理
        df = df.fillna(0)
        
        all_boats_data[f"Boat{boat_id}"] = df
    
    return all_boats_data

if __name__ == "__main__":
    main()

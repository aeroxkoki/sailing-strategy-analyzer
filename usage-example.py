"""
セーリングデータ処理モジュールの使用例
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import math
from sailing_data_processor import SailingDataProcessor

def generate_sample_data(num_boats=2):
    """テスト用のサンプルGPSデータを生成"""
    # 東京湾でのセーリングレースを想定した座標
    base_lat, base_lon = 35.620, 139.770
    
    # 各艇のデータを格納
    all_boats_data = {}
    
    for boat_id in range(1, num_boats + 1):
        # 時間間隔（秒）
        time_interval = 1  # 1秒間隔
        
        # データポイント数
        num_points = 600  # 10分分（1秒間隔）
        
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
        
        # 風上レグ
        leg1_points = 300
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
        leg2_points = 300
        for i in range(leg2_points):
            progress = i / leg2_points
            # より直線的な動き
            lats.append(base_lat + 0.03 - progress * 0.03 + 0.001 * math.sin(i/10) + lat_var)
            lons.append(base_lon + 0.01 + 0.002 * math.cos(i/8) + lon_var)
            speeds.append(6.0 + 0.3 * math.sin(i/15) + 0.2 * (boat_id - 1))
        
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

def main():
    """メイン処理"""
    # データプロセッサーのインスタンス作成
    processor = SailingDataProcessor()
    
    print("1. サンプルデータの生成")
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
    
    print("2. データの読み込み")
    # データ読み込み
    processor.load_multiple_files(file_contents, auto_id=True)
    print(f"  読み込み完了: {len(processor.boat_data)}艇のデータ")
    
    print("3. データのクリーニングと異常値検出")
    # 全艇データのクリーニング
    cleaned_data = processor.clean_all_boat_data(max_speed_knots=30.0)
    print(f"  クリーニング完了: {len(cleaned_data)}艇のデータ")
    
    print("4. 時刻同期処理")
    # 共通の時間枠を検出
    start_time, end_time = processor.get_common_timeframe()
    if start_time and end_time:
        print(f"  共通時間枠: {start_time} から {end_time}")
        # 同期（2秒間隔）
        synced_data = processor.synchronize_time(target_freq='2s', start_time=start_time, end_time=end_time)
        print(f"  同期完了: {len(synced_data)}艇のデータ")
    else:
        print("  共通の時間枠が見つかりませんでした")
    
    print("5. データ品質レポート")
    # データ品質レポート
    quality_report = processor.get_data_quality_report()
    for boat_id, report in quality_report.items():
        print(f"  艇 {boat_id}:")
        print(f"    データ点数: {report['total_points']}")
        print(f"    計測時間: {report['duration_seconds']:.1f}秒")
        print(f"    サンプリングレート: {report['avg_sampling_rate']:.2f}Hz")
        print(f"    品質スコア: {report['quality_score']:.1f} ({report['quality_rating']})")
    
    print("6. 速度グラフのプロット")
    # 速度グラフのプロット
    plt.figure(figsize=(10, 6))
    
    for boat_id, df in processor.synced_data.items():
        # 時間軸の正規化
        t0 = df['timestamp'].iloc[0]
        time_mins = [(t - t0).total_seconds() / 60 for t in df['timestamp']]
        
        # 速度をノットに変換
        speed_knots = df['speed'] * 1.94384
        
        plt.plot(time_mins, speed_knots, label=f"{boat_id}")
    
    plt.xlabel('Time (minutes)')
    plt.ylabel('Speed (knots)')
    plt.title('Boat Speed Comparison')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # グラフをファイルに保存
    plt.savefig('boat_speeds.png')
    print("  速度グラフを boat_speeds.png に保存しました")
    
    print("7. 処理済みデータのエクスポート")
    # CSVエクスポート
    for boat_id in processor.synced_data:
        csv_data = processor.export_processed_data(boat_id, format_type='csv')
        if csv_data:
            with open(f"{boat_id}_processed.csv", 'wb') as f:
                f.write(csv_data)
            print(f"  {boat_id} のデータを {boat_id}_processed.csv に保存しました")

if __name__ == "__main__":
    main()
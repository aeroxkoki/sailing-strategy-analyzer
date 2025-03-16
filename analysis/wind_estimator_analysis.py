"""
WindEstimatorクラスの詳細分析を行うスクリプト
"""

import pandas as pd
import numpy as np
import os
import sys
import math  # これが不足していました
from datetime import datetime, timedelta
import inspect
import matplotlib.pyplot as plt

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# WindEstimatorをインポート
from sailing_data_processor.wind_estimator import WindEstimator

def analyze_wind_estimator_structure():
    """WindEstimatorクラスの構造を分析"""
    print("=== WindEstimatorクラスの構造分析 ===")
    
    # WindEstimatorのインスタンス化
    estimator = WindEstimator()
    
    # 公開メソッドの抽出
    public_methods = [method for method in dir(estimator) 
                     if callable(getattr(estimator, method)) 
                     and not method.startswith('_')]
    
    print(f"\n公開メソッド ({len(public_methods)}):")
    for method in sorted(public_methods):
        # 引数情報を取得
        signature = inspect.signature(getattr(estimator, method))
        print(f"  - {method}{signature}")
    
    # 内部メソッドの抽出
    private_methods = [method for method in dir(estimator) 
                      if callable(getattr(estimator, method)) 
                      and method.startswith('_') 
                      and not method.startswith('__')]
    
    print(f"\n内部メソッド ({len(private_methods)}):")
    for method in sorted(private_methods):
        # 引数情報を取得
        signature = inspect.signature(getattr(estimator, method))
        print(f"  - {method}{signature}")
    
    # 状態変数の抽出
    state_vars = [var for var in dir(estimator) 
                 if not callable(getattr(estimator, var)) 
                 and not var.startswith('__')]
    
    print(f"\n状態変数 ({len(state_vars)}):")
    for var in sorted(state_vars):
        value = getattr(estimator, var)
        var_type = type(value).__name__
        
        # 辞書型の場合はキーを表示
        if isinstance(value, dict):
            keys_info = f"キー数: {len(value)}, キー: {list(value.keys())[:3] + ['...'] if len(value) > 3 else list(value.keys())}"
            print(f"  - {var} ({var_type}): {keys_info}")
        # リスト型の場合は長さを表示
        elif isinstance(value, list):
            print(f"  - {var} ({var_type}): 長さ {len(value)}")
        # その他の型は値を表示
        else:
            print(f"  - {var} ({var_type}): {value}")
    
    return {
        "public_methods": public_methods,
        "private_methods": private_methods,
        "state_vars": state_vars
    }

def analyze_critical_path():
    """WindEstimatorの重要パスを分析"""
    print("\n=== 重要パスの分析 ===")
    
    # 風向風速推定の主要なパス
    critical_path = [
        "estimate_wind_from_single_boat",  # 単一艇からの風推定（メイン機能）
        "_ensure_columns",                # カラム存在確認
        "_detect_tacks",                  # タック検出
        "_calculate_bearing",             # 方位計算
        "_angle_difference",              # 角度差分計算
        "_get_optimal_twa",               # 最適風向角取得
        "_weighted_angle_average",        # 角度の重み付き平均
        "_interpolate_boat_speed"         # 艇速補間
    ]
    
    print("風向風速推定の重要パス:")
    for path in critical_path:
        print(f"  - {path}")
    
    # アルゴリズム上の重要ポイントと影響
    key_algorithms = {
        "タック検出": "セーリングポイントの切り替わりを検出。風向推定の基礎となる。",
        "風向角計算": "風上・風下レグの方向から風向を推定。",
        "風速計算": "艇の速度とポーラー曲線から風速を逆算。",
        "ベイズ推定": "時間的に変化する風の状態を推定。"
    }
    
    print("\n重要アルゴリズム:")
    for algo, desc in key_algorithms.items():
        print(f"  - {algo}: {desc}")
    
    return {
        "critical_path": critical_path,
        "key_algorithms": key_algorithms
    }

def analyze_accuracy_factors():
    """精度に影響する要因を分析"""
    print("\n=== 精度影響要因の分析 ===")
    
    accuracy_factors = {
        "min_tack_angle": "タック検出の閾値。小さすぎると誤検出、大きすぎると見逃しの原因。",
        "boat_coefficients": "艇種ごとの係数。不正確だと風速推定に影響。",
        "min_valid_duration": "有効な風推定に必要な最小データ期間。",
        "min_valid_points": "有効な風推定に必要な最小データポイント数。",
        "confidence_threshold": "風推定の信頼度閾値。高すぎると有効データ減少。",
        "upwind_ratio/downwind_ratio": "風上/風下の速度比率。艇速から風速を推定する係数。"
    }
    
    print("精度に影響する主要因子:")
    for factor, desc in accuracy_factors.items():
        print(f"  - {factor}: {desc}")
    
    return {
        "accuracy_factors": accuracy_factors
    }

def create_sample_data():
    """分析用のサンプルデータを作成"""
    print("\n=== サンプルデータの作成 ===")
    
    # タイムスタンプの作成
    timestamps = pd.date_range(start='2025-03-01 10:00:00', periods=200, freq='5s')  # 'S'から's'に変更
    
    # 基本のコースパターン（風上/風下の往復）
    # 風向を270度（西風）と仮定
    assumed_wind_direction = 270
    
    # 風上走行（タック）を模したジグザグコース
    bearings = []
    
    # 2つのタックを作成
    for i in range(200):
        if i < 50:  # 最初のタック（北西方向）
            bearings.append(315)
        elif i < 60:  # タック変更中
            bearings.append(315 - (i-50) * 9)  # 徐々に変える
        elif i < 110:  # 2番目のタック（北東方向）
            bearings.append(225)
        elif i < 120:  # タック変更中
            bearings.append(225 + (i-110) * 9)  # 徐々に変える
        elif i < 170:  # 3番目のタック（北西方向）
            bearings.append(315)
        else:  # タック変更中
            bearings.append(315 - (i-170) * 9)  # 徐々に変える
    
    # 位置の計算
    lats = [35.45]
    lons = [139.65]
    
    # 毎秒0.5mで移動すると仮定
    speed_ms = 3.0  # 3m/s ≈ 6ノット
    
    for i in range(1, 200):
        bearing_rad = math.radians(bearings[i])
        # 緯度1度 ≈ 111km, 経度1度 ≈ 111km * cos(latitude)
        lat_change = math.cos(bearing_rad) * speed_ms * 5 / 111000  # 5秒間の変化
        lon_change = math.sin(bearing_rad) * speed_ms * 5 / (111000 * math.cos(math.radians(lats[-1])))
        
        lats.append(lats[-1] + lat_change)
        lons.append(lons[-1] + lon_change)
    
    # DataFrameを作成
    df = pd.DataFrame({
        'timestamp': timestamps,
        'latitude': lats,
        'longitude': lons,
        'bearing': bearings,
        'speed': [speed_ms] * 200  # 一定速度と仮定
    })
    
    print(f"サンプルデータ作成完了: {len(df)}行")
    print(f"データ期間: {df['timestamp'].min()} から {df['timestamp'].max()}")
    
    return df

def test_wind_estimation(sample_data):
    """サンプルデータを使用して風推定をテスト"""
    print("\n=== 風推定テスト ===")
    
    estimator = WindEstimator()
    
    # 風向風速を推定
    print("風向風速の推定を実行中...")
    results = estimator.estimate_wind_from_single_boat(
        gps_data=sample_data,
        min_tack_angle=30.0,
        boat_type='default',
        use_bayesian=True
    )
    
    if results is not None:
        print(f"推定結果: {len(results)}行のデータ")
        print(f"推定された風向の平均: {results['wind_direction'].mean():.1f}度")
        print(f"推定された風速の平均: {results.get('wind_speed_knots', results.get('wind_speed', 0)).mean():.1f}ノット")
        print(f"推定の平均信頼度: {results['confidence'].mean():.2f}")
        
        # 実際の風向（270度）との比較
        error = abs(angle_difference(results['wind_direction'].mean(), 270))
        print(f"実際の風向との誤差: {error:.1f}度")
    else:
        print("風推定に失敗しました")
    
    return results

def angle_difference(angle1, angle2):
    """2つの角度間の最小差分を計算"""
    return ((angle1 - angle2 + 180) % 360) - 180

def main():
    """メイン実行関数"""
    print("WindEstimator分析ツール - 開始")
    
    # WindEstimatorの構造分析
    analyze_wind_estimator_structure()
    
    # 重要パスの分析
    analyze_critical_path()
    
    # 精度要因の分析
    analyze_accuracy_factors()
    
    # サンプルデータの作成
    sample_data = create_sample_data()
    
    # 風推定のテスト
    test_results = test_wind_estimation(sample_data)
    
    print("\nWindEstimator分析ツール - 完了")

if __name__ == "__main__":
    main()

"""
WindEstimatorクラスの詳細分析を行うスクリプト
"""

import pandas as pd
import numpy as np
import os
import sys
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
        "min_valid_points": "有効な風推定に必要な最小データポイ

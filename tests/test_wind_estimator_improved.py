# -*- coding: utf-8 -*-
"""
WindEstimatorクラスの改良部分をテストするスクリプト
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import os
import sys
import pytest
import unittest
import logging

logging.basicConfig(level=logging.DEBUG)

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# テスト対象のクラスをインポート
from sailing_data_processor.wind.wind_estimator import WindEstimator
from sailing_data_processor.wind.wind_estimator_maneuvers import categorize_maneuver, determine_point_state

@pytest.fixture
def estimator():
    """テスト用のWindEstimatorインスタンスを返す"""
    return WindEstimator()

@pytest.fixture
def test_data():
    """シンプルなタックを含むテストデータを返す"""
    base_time = datetime(2024, 3, 1, 10, 0, 0)
    points = 100
    timestamps = [base_time + timedelta(seconds=i*5) for i in range(points)]
    
    # 方位データ - より明確なタックパターン（急激な変化）を作成
    bearings = [30] * 45  # 最初の45ポイントは30度
    # より急激な変化を持つタックパターンを作成
    bearings.append(45)  # 少し方向を変える
    bearings.append(90)  # さらに方向が変わる
    bearings.append(150)  # 大きく変わる - これでタックが検出されるはず
    bearings += [150] * (points - len(bearings))  # 残りのポイントは150度
    
    # 緯度・経度データ（シンプルな直線）
    base_lat, base_lon = 35.6, 139.7
    lats = [base_lat + i * 0.0001 for i in range(points)]
    lons = [base_lon + i * 0.0001 for i in range(points)]
    
    # 速度データ（タック時に減速）
    speeds = []
    for i in range(points):
        if 45 <= i < 55:  # タック中は減速
            speeds.append(3.0)
        else:
            speeds.append(5.0)
    
    # 配列の長さを確認
    assert len(timestamps) == points
    assert len(bearings) == points
    assert len(lats) == points
    assert len(lons) == points
    assert len(speeds) == points
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'latitude': lats,
        'longitude': lons,
        'speed': np.array(speeds) * 0.514444,  # ノット→m/s変換
        'bearing': bearings,
        'heading': bearings,  # headingカラムを追加
        'boat_id': ['TestBoat'] * points
    })
    
    return df

class TestWindEstimatorImproved:
    """WindEstimatorクラスの改良部分をテスト"""
    
    def test_categorize_maneuver(self):
        """マニューバー分類機能のテスト"""
        
        # テストシナリオの設定
        test_cases = [
            # (before_bearing, after_bearing, wind_direction, expected_type)
            # タックのテスト
            (30, 330, 0, 'tack'),
            (330, 30, 0, 'tack'),
            # ジャイブのテスト
            (150, 210, 0, 'jibe'),
            (210, 150, 0, 'jibe'),
            # ベアアウェイのテスト
            (30, 150, 0, 'bear_away'),
            # ヘッドアップのテスト
            (150, 30, 0, 'head_up'),
        ]
        
        # 風上・風下の閾値
        upwind_threshold = 45.0
        downwind_threshold = 120.0
        
        for i, (before, after, wind_dir, expected) in enumerate(test_cases):
            result = categorize_maneuver(before, after, wind_dir, upwind_threshold, downwind_threshold)
            assert result['maneuver_type'] == expected, f"テスト{i+1}失敗: 期待されるマニューバタイプ（{expected}）と実際の結果（{result['maneuver_type']}）が一致しません"
            
            # 信頼度が0-1の範囲内にあることを確認
            assert 0 <= result['confidence'] <= 1, f"テスト{i+1}失敗: 信頼度（{result['confidence']}）が範囲外です"
            
            # 状態が正しく設定されていることを確認
            assert result['before_state'] in ['upwind', 'downwind', 'reaching'], f"転換前の状態（{result['before_state']}）が無効です"
            assert result['after_state'] in ['upwind', 'downwind', 'reaching'], f"転換後の状態（{result['after_state']}）が無効です"
    
    def test_determine_point_state(self):
        """風に対する状態判定のテスト"""
        
        # テストケース
        test_cases = [
            # (相対角度, upwind_range, downwind_range, 期待される状態)
            (0, 80, 100, 'upwind'),
            (80, 80, 100, 'upwind'),
            (359, 80, 100, 'upwind'),
            (100, 80, 100, 'downwind'),
            (180, 80, 100, 'downwind'),
            (260, 80, 100, 'downwind'),
            (85, 80, 100, 'reaching'),
            (95, 80, 100, 'reaching'),
            (275, 80, 100, 'reaching'),
        ]
        
        for i, (rel_angle, upwind, downwind, expected) in enumerate(test_cases):
            result = determine_point_state(rel_angle, upwind, downwind)
            assert result == expected, f"テスト{i+1}失敗: 相対角度{rel_angle}°の期待される状態（{expected}）と実際の結果（{result}）が一致しません"
    
    def test_detect_maneuvers(self, estimator, test_data):
        """マニューバー検出機能のテスト"""
        
        # 方位カラムの確認と追加（test_dataがbearingを持っているがheadingを持っていない場合、headingを追加）
        if 'bearing' in test_data.columns and 'heading' not in test_data.columns:
            test_data['heading'] = test_data['bearing']
        
        # 風向を設定（テストデータでは風向0度を想定）
        wind_direction = 0.0
        
        # マニューバー検出
        maneuvers = estimator.detect_maneuvers(test_data, wind_direction)
        
        # 検出結果の確認
        assert not maneuvers.empty, "マニューバが検出されていません"
        
        # 検出されたマニューバーの内容を確認
        first_maneuver = maneuvers.iloc[0]
        assert first_maneuver['maneuver_type'] in ['tack', 'jibe', 'bear_away', 'head_up'], "検出されたマニューバタイプが無効です"
        assert 0 <= first_maneuver['maneuver_confidence'] <= 1, "信頼度が範囲外です"
        assert 'angle_change' in first_maneuver, "角度変化が含まれていません"
        assert 'before_bearing' in first_maneuver and 'after_bearing' in first_maneuver, "方位情報が不足しています"
        assert 'before_state' in first_maneuver and 'after_state' in first_maneuver, "状態情報が不足しています"

# Directly run pytest if this file is executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

"""
セーリング戦略分析システム - 戦略的判断ポイント検出モジュール

セーリングレース中の重要な戦略的判断ポイントを検出し、
それらの評価とリスク分析を行う機能を提供します。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import warnings

# 内部モジュールのインポート
from .strategy.points import StrategyPoint, WindShiftPoint, TackPoint, LaylinePoint, StrategyAlternative
from .strategy.detector import StrategyDetector
from .strategy.evaluator import StrategyEvaluator
from .strategy.visualizer import StrategyVisualizer

try:
    from .utilities.math_utils import normalize_angle, angle_difference
except ImportError:
    # スタンドアロン実行用
    def normalize_angle(angle):
        """角度を0-360度の範囲に正規化"""
        return angle % 360
        
    def angle_difference(angle1, angle2):
        """2つの角度間の最小差分を計算"""
        return ((angle1 - angle2 + 180) % 360) - 180


class StrategyPointDetector:
    """
    戦略的判断ポイントを検出し評価するクラス
    
    このクラスは、風向風速データと最適VMG計算結果を基に、
    セーリングレース中の重要な戦略的判断ポイントを特定し、
    それらのリスクと代替戦略を評価します。
    """
    
    def __init__(self, wind_field_interpolator=None, vmg_calculator=None):
        """
        初期化
        
        Parameters:
        -----------
        wind_field_interpolator : WindFieldInterpolator
            風の場補間器（オプション）
        vmg_calculator : OptimalVMGCalculator
            VMG計算機（オプション）
        """
        # 内部コンポーネントの初期化
        self.detector = StrategyDetector(wind_field_interpolator, vmg_calculator)
        self.evaluator = StrategyEvaluator()
        self.visualizer = StrategyVisualizer()
        
        # 外部参照用に保持
        self.wind_field_interpolator = wind_field_interpolator
        self.vmg_calculator = vmg_calculator
        
        # 検出された戦略ポイント
        self.detected_points = []
        
        # 設定
        self.config = {
            # 全般設定
            'min_confidence_threshold': 0.6,    # 信頼度の最小閾値
            'max_strategy_points': 8,           # 最大戦略ポイント数
            
            # 風向シフト検出設定
            'min_wind_shift_angle': 5.0,        # 最小風向シフト角度（度）
            'wind_forecast_interval': 300,      # 風予測間隔（秒）
            'max_wind_forecast_horizon': 1800,  # 最大風予測期間（秒）
            
            # タック検出設定
            'tack_search_radius': 500,          # タック探索半径（メートル）
            'min_vmg_improvement': 0.05,        # 最小VMG改善閾値（比率）
            'max_tacks_per_leg': 3,             # レグあたり最大タック数
            
            # レイライン検出設定
            'layline_safety_margin': 10.0,      # レイライン安全マージン（度）
            'min_mark_distance': 100,           # マークからの最小検出距離（メートル）
            
            # 重要度評価設定
            'time_weight': 0.5,                 # 時間要素の重み
            'risk_weight': 0.3,                 # リスク要素の重み
            'gain_weight': 0.2,                 # 利得要素の重み
        }
    
    def update_config(self, **kwargs):
        """
        設定を更新
        
        Parameters:
        -----------
        **kwargs
            更新する設定のキーと値
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                
                # 内部コンポーネントにも設定を伝播
                if hasattr(self.detector, 'config') and key in self.detector.config:
                    self.detector.config[key] = value
                if hasattr(self.evaluator, 'config') and key in self.evaluator.config:
                    self.evaluator.config[key] = value
            else:
                warnings.warn(f"未知の設定キー: {key}")
    
    def detect_critical_points(self, course_data: Dict[str, Any], wind_field: Dict[str, Any], 
                             target_time: datetime = None) -> List[StrategyPoint]:
        """
        コースデータと風の場から戦略的判断ポイントを検出
        
        Parameters:
        -----------
        course_data : Dict[str, Any]
            OptimalVMGCalculatorのコース計算結果
        wind_field : Dict[str, Any]
            WindFieldInterpolatorからの風の場データ
        target_time : datetime, optional
            戦略ポイント検出の対象時間
            
        Returns:
        --------
        List[StrategyPoint]
            検出された戦略的判断ポイントのリスト
        """
        # 設定を伝播
        self.detector.config = self.config
        self.evaluator.config = self.config
        
        # 各タイプの戦略ポイントを検出
        wind_shift_points = self.detector.detect_wind_shifts(course_data, wind_field, target_time)
        tack_points = self.detector.detect_optimal_tacks(course_data, wind_field)
        layline_points = self.detector.detect_laylines(course_data, wind_field)
        
        # すべてのポイントを統合
        all_points = []
        all_points.extend(wind_shift_points)
        all_points.extend(tack_points)
        all_points.extend(layline_points)
        
        # リスク評価と代替戦略の生成
        evaluated_points = self.evaluator.evaluate_risk_factors(all_points, wind_field)
        points_with_alternatives = self.evaluator.generate_alternative_strategies(evaluated_points, wind_field)
        
        # 重要度によるランク付けとフィルタリング
        ranked_points = self.evaluator.rank_strategy_points(points_with_alternatives)
        filtered_points = self.evaluator.filter_by_importance(
            ranked_points, 
            self.config['min_confidence_threshold'],
            self.config['max_strategy_points']
        )
        
        # 検出結果を保存
        self.detected_points = filtered_points
        return filtered_points
    
    def detect_critical_points_optimized(self, course_data: Dict[str, Any], wind_field: Dict[str, Any], 
                                      target_time: datetime = None) -> List[StrategyPoint]:
        """
        大規模コースデータの効率的な戦略ポイント検出
        
        Parameters:
        -----------
        course_data : Dict[str, Any]
            コース計算結果
        wind_field : Dict[str, Any]
            風の場データ
        target_time : datetime, optional
            戦略ポイント検出の対象時間
            
        Returns:
        --------
        List[StrategyPoint]
            検出された戦略的判断ポイントのリスト
        """
        return self.detector.detect_critical_points_optimized(
            course_data, wind_field, target_time
        )
    
    def visualize_strategy_points(self, background_map=None, save_path=None):
        """
        戦略ポイントを可視化
        
        Parameters:
        -----------
        background_map : Any, optional
            背景マップ（オプション）
        save_path : str, optional
            保存先パス
            
        Returns:
        --------
        matplotlib.figure.Figure
            プロット図
        """
        return self.visualizer.visualize_strategy_points(
            self.detected_points, background_map, save_path
        )
    
    def get_strategy_points_by_type(self, point_type: str) -> List[StrategyPoint]:
        """
        特定タイプの戦略ポイントを取得
        
        Parameters:
        -----------
        point_type : str
            取得するポイントタイプ ("wind_shift", "tack", "layline")
            
        Returns:
        --------
        List[StrategyPoint]
            指定タイプの戦略ポイントリスト
        """
        return [p for p in self.detected_points if p.point_type == point_type]
    
    def get_prioritized_strategy_points(self, current_position: Tuple[float, float], 
                                     current_time: float, lookahead: float = 600) -> List[StrategyPoint]:
        """
        優先度付けされた戦略ポイントを取得
        
        Parameters:
        -----------
        current_position : Tuple[float, float]
            現在位置 (lat, lon)
        current_time : float
            現在時刻
        lookahead : float
            先読み時間（秒）
            
        Returns:
        --------
        List[StrategyPoint]
            優先度付けされた戦略ポイントリスト
        """
        return self.evaluator.prioritize_strategy_points(
            self.detected_points, current_position, current_time, lookahead
        )

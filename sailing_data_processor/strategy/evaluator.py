"""
sailing_data_processor.strategy.evaluator モジュール

戦略的判断ポイントの評価、リスク分析、代替戦略生成を実装しています。
見つかった戦略ポイントの重要度ランキングと最適な代替オプションを提供します。
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# 内部モジュールのインポート
from .points import (
    StrategyPoint, WindShiftPoint, TackPoint, LaylinePoint, StrategyAlternative
)

class StrategyEvaluator:
    """戦略的判断ポイントの評価と分析を行うクラス"""
    
    def __init__(self, vmg_calculator=None):
        """
        評価器の初期化
        
        Parameters:
        -----------
        vmg_calculator : OptimalVMGCalculator, optional
            VMG計算機（オプション）
        """
        self.vmg_calculator = vmg_calculator
        
        # 評価設定
        self.config = {
            # 重要度評価設定
            'time_weight': 0.5,                 # 時間要素の重み
            'risk_weight': 0.3,                 # リスク要素の重み
            'gain_weight': 0.2,                 # 利得要素の重み
            
            # フィルタリング設定
            'min_confidence_threshold': 0.6,    # 信頼度の最小閾値
            'max_strategy_points': 8,           # 最大戦略ポイント数
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
    
    def evaluate_risk_factors(self, strategy_points: List[StrategyPoint], 
                            wind_field: Dict[str, Any]) -> List[StrategyPoint]:
        """
        各戦略ポイントのリスク要因を評価
        
        Parameters:
        -----------
        strategy_points : List[StrategyPoint]
            評価する戦略ポイントのリスト
        wind_field : Dict[str, Any]
            風の場データ
            
        Returns:
        --------
        List[StrategyPoint]
            リスク評価が追加された戦略ポイントのリスト
        """
        for point in strategy_points:
            # 各ポイント種類のリスク評価メソッドを呼び出す
            point.evaluate_risk()
            
            # リスク評価に失敗した場合の代替計算
            if point.risk_score <= 0:
                point.risk_score = self._calculate_fallback_risk(point)
            
            # 信頼度の確認
            if point.confidence <= 0:
                point.confidence = self._estimate_confidence(point, wind_field)
        
        return strategy_points
    
    def _calculate_fallback_risk(self, point: StrategyPoint) -> float:
        """
        リスク評価の代替計算（ポイント個別の評価に失敗した場合）
        
        Parameters:
        -----------
        point : StrategyPoint
            評価対象のポイント
            
        Returns:
        --------
        float
            リスクスコア（0-100）
        """
        # ポイント種類ごとのデフォルトリスク
        default_risks = {
            'wind_shift': 60,  # 風向シフトは中程度のリスク
            'tack': 50,        # タックは中程度のリスク
            'layline': 70      # レイラインは高リスク
        }
        
        # 基本リスク
        base_risk = default_risks.get(point.point_type, 50)
        
        # 風の変動性による調整
        if hasattr(point, 'wind_info') and 'variability' in point.wind_info:
            variability = point.wind_info['variability']
            variability_factor = 1.0 + variability * 0.5  # 変動性が高いほどリスク増加
        else:
            variability_factor = 1.0
        
        # 種類固有の調整
        type_factor = 1.0
        
        if point.point_type == 'wind_shift':
            # 風向シフトポイントの場合
            if hasattr(point, 'shift_angle'):
                # 大きなシフトほどリスクが高い
                type_factor = 1.0 + min(0.5, abs(point.shift_angle) / 45.0)
        
        elif point.point_type == 'tack':
            # タックポイントの場合
            if hasattr(point, 'timing_sensitivity'):
                # タイミング感度が高いほどリスクが高い
                type_factor = 1.0 + point.timing_sensitivity * 0.5
        
        elif point.point_type == 'layline':
            # レイラインポイントの場合
            if hasattr(point, 'mark_distance'):
                # マークが遠いほどリスクが高い（風向変化の影響を受けやすい）
                dist_factor = min(0.5, point.mark_distance / 2000.0)  # 2000mで最大+50%
                type_factor = 1.0 + dist_factor
                
                # 交通量が多いほどリスクが高い
                if hasattr(point, 'traffic_factor'):
                    type_factor += point.traffic_factor * 0.3
        
        # 最終リスクスコア
        risk_score = base_risk * variability_factor * type_factor
        
        # 範囲制限
        return min(100, max(0, risk_score))
    
    def _estimate_confidence(self, point: StrategyPoint, wind_field: Dict[str, Any]) -> float:
        """
        ポイントの信頼度を推定
        
        Parameters:
        -----------
        point : StrategyPoint
            評価対象のポイント
        wind_field : Dict[str, Any]
            風の場データ
            
        Returns:
        --------
        float
            信頼度（0-1）
        """
        # 基本信頼度
        base_confidence = 0.7
        
        # 風情報がある場合は風の信頼度を考慮
        if hasattr(point, 'wind_info') and 'confidence' in point.wind_info:
            base_confidence = point.wind_info['confidence']
        
        # 種類固有の調整
        type_factor = 1.0
        
        if point.point_type == 'wind_shift':
            # 風向シフトの場合
            if hasattr(point, 'shift_probability'):
                return point.shift_probability  # 専用の信頼度がある場合はそれを使用
            
            # シフト角度が小さいほど信頼度は低い
            if hasattr(point, 'shift_angle'):
                type_factor = min(1.0, 0.6 + abs(point.shift_angle) / 30.0)
        
        elif point.point_type == 'tack':
            # タックポイントの場合
            # VMG利得が大きいほど信頼度が高い
            if hasattr(point, 'vmg_gain'):
                type_factor = min(1.0, 0.7 + point.vmg_gain * 1.5)
        
        elif point.point_type == 'layline':
            # レイラインポイントの場合
            # マークに近いほど信頼度が高い
            if hasattr(point, 'mark_distance'):
                type_factor = min(1.0, 0.6 + 500.0 / max(1, point.mark_distance))
        
        # 最終信頼度
        confidence = base_confidence * type_factor
        
        # 範囲制限
        return min(1.0, max(0.0, confidence))
    
    def generate_alternative_strategies(self, strategy_points: List[StrategyPoint], 
                                      wind_field: Dict[str, Any]) -> List[StrategyPoint]:
        """
        各戦略ポイントの代替戦略を生成
        
        Parameters:
        -----------
        strategy_points : List[StrategyPoint]
            戦略ポイントのリスト
        wind_field : Dict[str, Any]
            風の場データ
            
        Returns:
        --------
        List[StrategyPoint]
            代替戦略が追加された戦略ポイントのリスト
        """
        for point in strategy_points:
            alternatives = []
            
            # ポイント種類ごとの代替戦略生成
            if point.point_type == "wind_shift":
                alternatives = self._generate_wind_shift_alternatives(point)
            elif point.point_type == "tack":
                alternatives = self._generate_tack_alternatives(point)
            elif point.point_type == "layline":
                alternatives = self._generate_layline_alternatives(point)
            
            # 代替戦略の信頼度を設定
            for alt in alternatives:
                alt.confidence = point.confidence * 0.9  # 代替戦略は基本戦略より信頼度低め
            
            # 戦略ポイントに代替戦略を追加
            point.alternatives = alternatives
        
        return strategy_points
    
    def _generate_wind_shift_alternatives(self, point: WindShiftPoint) -> List[StrategyAlternative]:
        """
        風向シフトポイントの代替戦略を生成
        
        Parameters:
        -----------
        point : WindShiftPoint
            風向シフトポイント
            
        Returns:
        --------
        List[StrategyAlternative]
            代替戦略のリスト
        """
        alternatives = []
        
        if hasattr(point, 'favorable') and hasattr(point, 'shift_angle'):
            if point.favorable:
                # 有利なシフトの場合
                alt1 = StrategyAlternative("wait_for_shift")
                alt1.advantage = "Maximize gain from favorable shift"
                alt1.disadvantage = "May lose distance waiting for shift"
                alt1.expected_gain = point.shift_angle * 0.2  # 簡易推定
                
                alt2 = StrategyAlternative("ignore_shift")
                alt2.advantage = "Continue on optimal VMG path"
                alt2.disadvantage = "Miss opportunity to gain from shift"
                alt2.expected_gain = 0
                
                alternatives = [alt1, alt2]
            else:
                # 不利なシフトの場合
                alt1 = StrategyAlternative("early_tack")
                alt1.advantage = "Minimize impact of unfavorable shift"
                alt1.disadvantage = "May tack too early with VMG loss"
                alt1.expected_gain = abs(point.shift_angle) * 0.15  # 簡易推定
                
                alt2 = StrategyAlternative("delay_decision")
                alt2.advantage = "Maintain optimal VMG until shift occurs"
                alt2.disadvantage = "May face maximum impact of shift"
                alt2.expected_gain = -abs(point.shift_angle) * 0.1  # 簡易推定
                
                alternatives = [alt1, alt2]
                
        return alternatives
    
    def _generate_tack_alternatives(self, point: TackPoint) -> List[StrategyAlternative]:
        """
        タックポイントの代替戦略を生成
        
        Parameters:
        -----------
        point : TackPoint
            タックポイント
            
        Returns:
        --------
        List[StrategyAlternative]
            代替戦略のリスト
        """
        alternatives = []
        
        # 早めのタック
        alt1 = StrategyAlternative("early_tack")
        alt1.time_delta = -30  # 30秒早め
        alt1.advantage = "Potentially better position for next shift"
        alt1.disadvantage = "VMG loss compared to optimal timing"
        alt1.expected_gain = -2.0  # 秒数でのコスト
        
        # 遅めのタック
        alt2 = StrategyAlternative("delay_tack")
        alt2.time_delta = 30  # 30秒遅め
        alt2.advantage = "May find better wind conditions"
        alt2.disadvantage = "Possible VMG loss and overstanding risk"
        alt2.expected_gain = -3.0  # 秒数でのコスト
        
        alternatives = [alt1, alt2]
        
        return alternatives
    
    def _generate_layline_alternatives(self, point: LaylinePoint) -> List[StrategyAlternative]:
        """
        レイラインポイントの代替戦略を生成
        
        Parameters:
        -----------
        point : LaylinePoint
            レイラインポイント
            
        Returns:
        --------
        List[StrategyAlternative]
            代替戦略のリスト
        """
        alternatives = []
        
        # 早めのレイライン到達
        alt1 = StrategyAlternative("early_layline")
        alt1.advantage = "Secure mark rounding, less traffic risk"
        alt1.disadvantage = "Sailing extra distance if wind shifts unfavorably"
        alt1.expected_gain = -5.0  # 秒数でのコスト
        
        # 遅めのレイライン到達
        alt2 = StrategyAlternative("late_layline")
        alt2.advantage = "Shorter distance if wind remains steady"
        alt2.disadvantage = "Risk of missing mark if wind shifts adversely"
        
        if hasattr(point, 'mark_distance'):
            alt2.expected_gain = point.mark_distance * 0.01  # 距離に比例した利得
        else:
            alt2.expected_gain = 3.0
            
        alt2.risk_delta = 20.0  # リスク増加
        
        alternatives = [alt1, alt2]
        
        return alternatives
    
    def rank_strategy_points(self, strategy_points: List[StrategyPoint]) -> List[StrategyPoint]:
        """
        戦略ポイントの重要度によるランク付け
        
        Parameters:
        -----------
        strategy_points : List[StrategyPoint]
            戦略ポイントのリスト
            
        Returns:
        --------
        List[StrategyPoint]
            重要度でランク付けされた戦略ポイントのリスト
        """
        if not strategy_points:
            return []
        
        # 重要度計算のための重み
        time_weight = self.config['time_weight']
        risk_weight = self.config['risk_weight']
        gain_weight = self.config['gain_weight']
        
        # 時間基準を設定（最初のポイントの時間）
        base_time = min(point.time_estimate for point in strategy_points)
        max_time = max(point.time_estimate for point in strategy_points)
        time_range = max(1, max_time - base_time)  # ゼロ除算防止
        
        # リスクの最大値を取得（正規化用）
        max_risk = max(point.risk_score for point in strategy_points) if strategy_points else 100
        
        for point in strategy_points:
            # 時間要素（近いほど重要）
            time_factor = 1.0 - ((point.time_estimate - base_time) / time_range)
            
            # リスク要素（リスクが高いほど重要）
            risk_factor = point.risk_score / max_risk if max_risk > 0 else 0.5
            
            # 利得要素（タイプに応じて計算）
            gain_factor = 0.0
            
            if point.point_type == "wind_shift":
                # 風向シフトの利得
                if hasattr(point, 'shift_angle'):
                    gain_factor = min(1.0, abs(point.shift_angle) / 30)  # 30度を1.0とする
                    # 有利なシフトは重要度が高い
                    if hasattr(point, 'favorable') and point.favorable:
                        gain_factor *= 1.2
            elif point.point_type == "tack":
                # タックの利得
                if hasattr(point, 'vmg_gain'):
                    gain_factor = min(1.0, point.vmg_gain * 5)  # 20%のVMG向上を1.0とする
            elif point.point_type == "layline":
                # レイラインの利得（マークに近いほど重要）
                if hasattr(point, 'mark_distance'):
                    gain_factor = min(1.0, 1000 / max(1, point.mark_distance))  # 1000mを基準
                    # トラフィック要素を考慮
                    if hasattr(point, 'traffic_factor'):
                        gain_factor *= (1.0 + point.traffic_factor)  # 交通量が多いほど重要
            
            # 重要度のスコア計算
            point.importance = (
                time_factor * time_weight +
                risk_factor * risk_weight +
                gain_factor * gain_weight
            )
            
            # 信頼度による調整
            point.importance *= point.confidence
        
        # 重要度でソート
        sorted_points = sorted(strategy_points, key=lambda p: p.importance, reverse=True)
        
        return sorted_points
    
    def filter_by_importance(self, ranked_points: List[StrategyPoint]) -> List[StrategyPoint]:
        """
        重要度に基づいて戦略ポイントをフィルタリング
        
        Parameters:
        -----------
        ranked_points : List[StrategyPoint]
            重要度でランク付けされた戦略ポイントのリスト
            
        Returns:
        --------
        List[StrategyPoint]
            フィルタリング後のリスト
        """
        # 最小信頼度によるフィルタリング
        min_confidence = self.config['min_confidence_threshold']
        confidence_filtered = [p for p in ranked_points if p.confidence >= min_confidence]
        
        # 最大ポイント数でフィルタリング
        max_points = self.config['max_strategy_points']
        
        return confidence_filtered[:max_points]
    
    def prioritize_for_real_time(self, strategy_points: List[StrategyPoint], 
                               current_time: float, lookahead: float = 1800) -> List[StrategyPoint]:
        """
        リアルタイム表示のための優先順位付け
        
        Parameters:
        -----------
        strategy_points : List[StrategyPoint]
            戦略ポイントのリスト
        current_time : float
            現在時刻
        lookahead : float, optional
            先読み時間（秒）、デフォルトは30分
            
        Returns:
        --------
        List[StrategyPoint]
            優先順位付けされたリスト
        """
        # 時間フィルタリング - 指定時間内のポイントのみを考慮
        time_filtered = [
            p for p in strategy_points 
            if current_time <= p.time_estimate <= current_time + lookahead
        ]
        
        # 各ポイントの優先度スコアを計算
        for point in time_filtered:
            # 時間的な近さ（近いほど重要）
            time_factor = 1.0 - ((point.time_estimate - current_time) / lookahead)
            
            # 影響度（リスクや利得から算出）
            impact_factor = point.importance * 1.5  # 重要度を基本とする
            
            # 総合スコア計算
            point.priority_score = time_factor * 0.7 + impact_factor * 0.3
        
        # スコアでソート
        prioritized = sorted(time_filtered, key=lambda p: p.priority_score, reverse=True)
        
        # トップN件のみ返す（表示制限など）
        top_n = min(5, len(prioritized))  # 最大5件
        return prioritized[:top_n]
    
    def evaluate_strategic_scenario(self, strategy_points: List[StrategyPoint]) -> Dict[str, Any]:
        """
        戦略ポイント群から総合的な戦略シナリオを評価
        
        Parameters:
        -----------
        strategy_points : List[StrategyPoint]
            戦略ポイントのリスト
            
        Returns:
        --------
        Dict[str, Any]
            戦略シナリオの評価結果
        """
        if not strategy_points:
            return {
                'scenario_risk': 0,
                'critical_points': [],
                'tactical_opportunities': [],
                'overall_assessment': "No strategic points detected."
            }
        
        # 種類ごとにポイントを分類
        wind_shifts = [p for p in strategy_points if p.point_type == "wind_shift"]
        tack_points = [p for p in strategy_points if p.point_type == "tack"]
        layline_points = [p for p in strategy_points if p.point_type == "layline"]
        
        # リスクの高いポイントを特定
        high_risk_points = [p for p in strategy_points if p.risk_score > 70]
        
        # 戦術的チャンスポイント（高い重要度、高い信頼度）
        opportunities = [p for p in strategy_points if p.importance > 0.7 and p.confidence > 0.7]
        
        # 総合リスクスコア
        if strategy_points:
            # 重要度加重平均
            total_importance = sum(p.importance for p in strategy_points)
            if total_importance > 0:
                scenario_risk = sum(p.risk_score * p.importance for p in strategy_points) / total_importance
            else:
                scenario_risk = sum(p.risk_score for p in strategy_points) / len(strategy_points)
        else:
            scenario_risk = 0
        
        # 総合評価
        if scenario_risk > 80:
            assessment = "High risk scenario requiring careful navigation."
        elif scenario_risk > 60:
            assessment = "Moderate risk scenario with some challenging points."
        elif scenario_risk > 40:
            assessment = "Manageable scenario with normal tactical considerations."
        else:
            assessment = "Low risk scenario with standard sailing conditions."
        
        # 風の安定性評価
        if wind_shifts:
            avg_shift = sum(abs(p.shift_angle) for p in wind_shifts if hasattr(p, 'shift_angle')) / len(wind_shifts)
            if avg_shift > 20:
                assessment += " Wind conditions are unstable with significant shifts."
            elif avg_shift > 10:
                assessment += " Wind conditions show moderate variability."
            else:
                assessment += " Wind conditions appear relatively stable."
        
        return {
            'scenario_risk': scenario_risk,
            'critical_points': high_risk_points[:3],  # 上位3つのみ
            'tactical_opportunities': opportunities[:3],  # 上位3つのみ
            'wind_shifts_count': len(wind_shifts),
            'tack_points_count': len(tack_points),
            'layline_points_count': len(layline_points),
            'overall_assessment': assessment
        }

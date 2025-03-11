"""
セーリング戦略分析システム - 戦略ポイントクラス

戦略的判断ポイントを表現するクラス群を提供します。
基底クラス StrategyPoint とその派生クラスが含まれます。
"""

from typing import Dict, List, Tuple, Any, Optional


class StrategyPoint:
    """戦略的判断ポイントを表現する基本クラス"""
    
    def __init__(self, point_type: str, position: Tuple[float, float], time_estimate: float):
        """
        戦略ポイントの初期化
        
        Parameters:
        -----------
        point_type : str
            ポイントの種類 ("tack", "wind_shift", "layline" など)
        position : Tuple[float, float]
            位置 (latitude, longitude)
        time_estimate : float
            到達予想時間（秒）
        """
        # 基本情報
        self.position = position            # (lat, lon)
        self.time_estimate = time_estimate  # 到達予想時間
        self.point_type = point_type        # "tack", "wind_shift", "layline" など
        
        # 評価情報
        self.risk_score = 0.0               # リスク評価スコア（0-100）
        self.confidence = 0.0               # 信頼度（0-1）
        self.importance = 0.0               # 重要度スコア
        
        # 説明・代替案
        self.description = ""               # 説明文
        self.recommendation = ""            # 推奨アクション
        self.alternatives = []              # 代替戦略オプション
        
        # 環境情報
        self.wind_info = {                  # 風情報
            'direction': 0.0,               # 風向（度）
            'speed': 0.0,                   # 風速（ノット）
            'direction_trend': 0.0,         # 風向変化トレンド（度/分）
            'speed_trend': 0.0,             # 風速変化トレンド（ノット/分）
            'variability': 0.0,             # 風の変動性（0-1）
            'confidence': 0.0               # 風情報の信頼度（0-1）
        }
        
    def evaluate_risk(self) -> float:
        """
        リスク評価を実行（サブクラスでオーバーライド）
        
        Returns:
        --------
        float
            リスクスコア（0-100）
        """
        return self.risk_score
        
    def generate_alternatives(self) -> List[Any]:
        """
        代替戦略を生成（サブクラスでオーバーライド）
        
        Returns:
        --------
        List[StrategyAlternative]
            代替戦略のリスト
        """
        return self.alternatives
        
    def get_description(self) -> str:
        """
        人間が理解できる説明を生成
        
        Returns:
        --------
        str
            説明文
        """
        return self.description

    def __str__(self) -> str:
        """文字列表現"""
        return f"{self.point_type.capitalize()} Point at {self.position}, Time: {self.time_estimate}, Risk: {self.risk_score:.1f}"
    
    def __repr__(self) -> str:
        """詳細な文字列表現"""
        return f"{self.__class__.__name__}(position={self.position}, time={self.time_estimate}, risk={self.risk_score:.1f}, importance={self.importance:.2f})"


class TackPoint(StrategyPoint):
    """タックまたはジャイブのポイント"""
    
    def __init__(self, position: Tuple[float, float], time_estimate: float):
        """
        タックポイントの初期化
        
        Parameters:
        -----------
        position : Tuple[float, float]
            位置 (latitude, longitude)
        time_estimate : float
            到達予想時間（秒）
        """
        super().__init__("tack", position, time_estimate)
        self.tack_angle = 0.0               # タック角度
        self.vmg_gain = 0.0                 # タックによるVMG向上量
        self.timing_sensitivity = 0.0       # タイミングの感度（値が高いほど精密なタイミングが必要）
        
    def evaluate_risk(self) -> float:
        """
        タックリスクの評価
        
        Returns:
        --------
        float
            リスクスコア（0-100）
        """
        # タック失敗リスク、風の不確実性リスクなどを評価
        risk_factors = {
            'wind_variability': min(100, self.wind_info.get('variability', 0.2) * 100),
            'timing_sensitivity': self.timing_sensitivity * 50,
            'execution_complexity': 40 if self.point_type == "tack" else 60  # ジャイブの方が複雑
        }
        
        # 重み付き合計
        self.risk_score = (
            risk_factors['wind_variability'] * 0.4 +
            risk_factors['timing_sensitivity'] * 0.3 +
            risk_factors['execution_complexity'] * 0.3
        )
        
        return self.risk_score


class WindShiftPoint(StrategyPoint):
    """風向シフトのポイント"""
    
    def __init__(self, position: Tuple[float, float], time_estimate: float):
        """
        風向シフトポイントの初期化
        
        Parameters:
        -----------
        position : Tuple[float, float]
            位置 (latitude, longitude)
        time_estimate : float
            到達予想時間（秒）
        """
        super().__init__("wind_shift", position, time_estimate)
        self.shift_angle = 0.0              # シフト角度（度）
        self.shift_duration = 0.0           # シフト持続時間（秒）
        self.shift_probability = 0.0        # シフト発生確率（0-1）
        self.favorable = False              # 有利なシフトかどうか
        
    def evaluate_risk(self) -> float:
        """
        風向シフトリスクの評価
        
        Returns:
        --------
        float
            リスクスコア（0-100）
        """
        # シフト検出失敗リスク、シフト対応リスクなどを評価
        risk_factors = {
            'shift_uncertainty': (1 - self.shift_probability) * 100,
            'shift_magnitude': min(100, abs(self.shift_angle) * 1.5),
            'response_time': max(0, min(100, 100 - self.shift_duration / 10))  # 持続時間が短いほどリスク大
        }
        
        # 重み付き合計
        self.risk_score = (
            risk_factors['shift_uncertainty'] * 0.5 +
            risk_factors['shift_magnitude'] * 0.3 +
            risk_factors['response_time'] * 0.2
        )
        
        return self.risk_score


class LaylinePoint(StrategyPoint):
    """レイライン（マークへの最終アプローチ）のポイント"""
    
    def __init__(self, position: Tuple[float, float], time_estimate: float):
        """
        レイラインポイントの初期化
        
        Parameters:
        -----------
        position : Tuple[float, float]
            位置 (latitude, longitude)
        time_estimate : float
            到達予想時間（秒）
        """
        super().__init__("layline", position, time_estimate)
        self.mark_distance = 0.0            # マークまでの距離（メートル）
        self.layline_angle = 0.0            # レイライン角度
        self.traffic_factor = 0.0           # 交通量係数（0-1、高いほど混雑）
        
    def evaluate_risk(self) -> float:
        """
        レイラインリスクの評価
        
        Returns:
        --------
        float
            リスクスコア（0-100）
        """
        # 超過リスク、風変化リスク、交通リスクなどを評価
        risk_factors = {
            'wind_shift_vulnerability': min(100, self.mark_distance / 50),  # 距離が長いほど風向変化の影響を受けやすい
            'traffic_congestion': self.traffic_factor * 100,
            'overshoot_risk': min(100, max(0, 90 - abs(self.layline_angle)))  # レイライン角度が鋭角なほどリスク大
        }
        
        # 重み付き合計
        self.risk_score = (
            risk_factors['wind_shift_vulnerability'] * 0.4 +
            risk_factors['traffic_congestion'] * 0.3 +
            risk_factors['overshoot_risk'] * 0.3
        )
        
        return self.risk_score


class StrategyAlternative:
    """代替戦略オプションを表現するクラス"""
    
    def __init__(self, action: str, position: Tuple[float, float] = None):
        """
        代替戦略の初期化
        
        Parameters:
        -----------
        action : str
            アクション種類（"early_tack", "delay_tack", "continue" など）
        position : Tuple[float, float], optional
            実行位置（位置が関連する場合）
        """
        self.action = action                # アクション種類
        self.position = position            # 実行位置（位置が関連する場合）
        self.time_delta = 0                 # 基本戦略からの時間差（秒）
        self.advantage = ""                 # 利点説明
        self.disadvantage = ""              # 欠点説明
        self.risk_delta = 0.0               # リスク変化量
        self.expected_gain = 0.0            # 期待利得（秒、メートルなど）
        self.confidence = 0.0               # 信頼度（0-1）
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"Alternative: {self.action}, Expected Gain: {self.expected_gain:.1f}, Risk Delta: {self.risk_delta:.1f}"

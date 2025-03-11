"""
sailing_data_processor.strategy パッケージ

セーリングレースにおける戦略的判断ポイントの検出、評価、可視化機能を提供します。
"""

from .points import StrategyPoint, WindShiftPoint, TackPoint, LaylinePoint, StrategyAlternative
from .detector import StrategyDetector
from .evaluator import StrategyEvaluator
from .visualizer import StrategyVisualizer

__version__ = '1.0.0'
__all__ = [
    'StrategyPoint',
    'WindShiftPoint',
    'TackPoint',
    'LaylinePoint',
    'StrategyAlternative',
    'StrategyDetector',
    'StrategyEvaluator',
    'StrategyVisualizer'
]

def get_version():
    """パッケージのバージョンを返します"""
    return __version__

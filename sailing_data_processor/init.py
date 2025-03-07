# sailing_data_processor/__init__.py
"""
セーリング戦略分析システム - データ処理モジュール

GPSデータから風向風速を推定し、セーリングレースの戦略分析を支援するモジュール
"""

# バージョン情報
__version__ = '2.0.0'

# 後方互換性のために元のクラスをインポート可能にする
from .core import SailingDataProcessor

# 新しいクラスもインポート可能にする
from .wind_estimator import WindEstimator
from .performance_optimizer import PerformanceOptimizer
from .boat_data_fusion import BoatDataFusionModel
from .wind_field_interpolator import WindFieldInterpolator

# 将来的に警告を表示するための準備
import warnings

# デフォルトでエクスポートするシンボル
__all__ = [
    'SailingDataProcessor',
    'WindEstimator',
    'PerformanceOptimizer',
    'BoatDataFusionModel',
    'WindFieldInterpolator'
]

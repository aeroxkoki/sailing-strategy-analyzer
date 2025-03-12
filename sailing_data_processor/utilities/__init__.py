"""
セーリング戦略分析システム - ユーティリティモジュール

GPS計算、数学関数、可視化機能などの共通ユーティリティを提供します。
"""

# バージョン情報
__version__ = '1.0.0'

# よく使用される数学ユーティリティ関数をインポート
from .math_utils import normalize_angle, angle_difference, average_angle, angle_dispersion

# エクスポートするシンボル
__all__ = [
    'normalize_angle',
    'angle_difference',
    'average_angle',
    'angle_dispersion'
]

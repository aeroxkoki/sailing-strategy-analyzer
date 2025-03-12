"""
セーリング戦略分析システム - 可視化モジュール

セーリングデータの地図表示、グラフ生成、パフォーマンス分析の可視化機能を提供します。
"""

# バージョン情報
__version__ = '1.0.0'

# 主要モジュールをインポート
from .sailing_visualizer import SailingVisualizer

# エクスポートするシンボル
__all__ = [
    'SailingVisualizer'
]

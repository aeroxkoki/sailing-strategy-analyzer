from setuptools import setup, find_packages

setup(
    name="sailing_strategy_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.38.0",
        "pandas>=2.2.0",
        "numpy>=1.26.3",
        "matplotlib>=3.8.2",
        "folium>=0.15.1",
        "streamlit-folium>=0.17.0",
        "gpxpy>=1.6.0",
        "geopy>=2.4.1",
        "scikit-learn>=1.4.0",
        "plotly>=5.18.0",
        "scipy>=1.11.0",
    ],
    description="セーリング戦略分析システム - GPSデータからの風向風速推定と最適コース戦略提案",
    long_description="""
    セーリング戦略分析システムは、GPSデータをベースにAI技術を活用して風向風速を推定し、
    最適なコース戦略を提案することで、セーリング競技者の意思決定を支援するツールです。
    
    主な機能:
    - GPSデータの読み込みと前処理
    - 風向風速の推定
    - コース戦略の最適化
    - データの可視化
    """,
    author="Sailing Strategy Team",
    author_email="contact@example.com",
    url="https://github.com/username/sailing-strategy-analyzer",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)

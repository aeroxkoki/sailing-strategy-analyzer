from setuptools import setup, find_packages

setup(
    name="sailing_strategy_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "folium",
        "streamlit-folium",
        "gpxpy",
        "geopy",
        "scikit-learn",
        "plotly",
    ],
    description="セーリング戦略分析システム",
    author="Your Name",
)

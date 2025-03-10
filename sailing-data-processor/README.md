# OptimalVMGCalculator - セーリング戦略分析システム

OptimalVMGCalculator は、風向風速データを基に最適なセーリング戦略を計算するエンジンです。風向・風速・艇種特性を考慮し、レース中の最適な進路やタイミングを計算します。

## 主な機能

- 異なる艇種の極座標データ(polar data)に基づく艇速計算
- 風向風速データを用いた最適VMG（Velocity Made Good）計算
- 出発地点から目標地点までの最適パス計算
- コース全体に対する最適戦略立案
- 風向変化やスタート位置の影響分析
- 戦略リスク評価と安全性チェック

## 動作要件

- Python 3.7以上
- NumPy
- Pandas
- Matplotlib
- GeoPy (距離計算)

## インストール方法

```bash
pip install -r requirements.txt
```

または sailing_data_processor パッケージの一部として:

```bash
pip install sailing-data-processor
```

## 基本的な使い方

### 1. 初期化

```python
from sailing_data_processor.optimal_vmg_calculator import OptimalVMGCalculator

# インスタンス作成（標準艇種がロードされます）
calculator = OptimalVMGCalculator()
```

### 2. 風向風速データの設定

```python
# 風向風速データを設定
calculator.set_wind_field(wind_field_data)
```

### 3. 最適VMGの計算

```python
# 特定地点から目標への最適VMGを計算
result = calculator.calculate_optimal_vmg(
    boat_type='laser',      # 艇種
    lat=35.605,             # 現在位置 (緯度)
    lon=139.775,            # 現在位置 (経度)
    target_lat=35.615,      # 目標位置 (緯度)
    target_lon=139.775      # 目標位置 (経度)
)

print(f"最適進路: {result['optimal_course']}°")
print(f"艇速: {result['boat_speed']} ノット")
print(f"VMG: {result['vmg']} ノット")
print(f"タックが必要: {'はい' if result['tack_needed'] else 'いいえ'}")
```

### 4. 最適パスの計算

```python
# 出発地点から目標までの最適パスを計算
path = calculator.find_optimal_path(
    boat_type='laser',       # 艇種
    start_lat=35.605,        # 出発位置 (緯度)
    start_lon=139.775,       # 出発位置 (経度)
    target_lat=35.625,       # 目標位置 (緯度)
    target_lon=139.775,      # 目標位置 (経度)
    max_tacks=3              # 最大タック数
)

print(f"パスポイント数: {len(path['path_points'])}")
print(f"タック数: {path['tack_count']}")
print(f"総距離: {path['total_distance']} メートル")
print(f"所要時間: {path['total_time'] / 60} 分")
```

### 5. コース全体の最適戦略計算

```python
# コース (ウェイポイントのリスト)
course = [
    {'name': 'Start', 'lat': 35.605, 'lon': 139.775, 'type': 'start'},
    {'name': 'Mark1', 'lat': 35.625, 'lon': 139.775, 'type': 'rounding'},
    {'name': 'Mark2', 'lat': 35.615, 'lon': 139.795, 'type': 'rounding'},
    {'name': 'Finish', 'lat': 35.605, 'lon': 139.775, 'type': 'finish'}
]

# コース全体の最適戦略を計算
strategy = calculator.calculate_optimal_route_for_course('laser', course)

print(f"レッグ数: {len(strategy['legs'])}")
print(f"総タック数: {strategy['total_tack_count']}")
print(f"総所要時間: {strategy['total_time'] / 60} 分")
```

### 6. 可視化

```python
# 最適パスを可視化
fig = calculator.visualize_optimal_path(path, show_wind=True)
plt.title("Optimal Path")
plt.savefig("optimal_path.png")

# コース戦略を可視化
fig = calculator.visualize_course_strategy(strategy, show_wind=True)
plt.title("Course Strategy")
plt.savefig("course_strategy.png")
```

## 詳細設定

計算パラメータをカスタマイズするには:

```python
calculator.update_config(
    path_resolution=100,     # パス計算の解像度(メートル)
    min_distance=20,         # 目標到達判定距離(メートル)
    use_parallel=True,       # 並列計算の使用
    use_vectorization=True   # ベクトル化計算の使用
)
```

## 戦略分析

戦略の安全性とリスクを評価するには:

```python
# パスの安全性をチェック
safety = calculator.check_path_safety(
    path_data=path,
    safety_margin_meters=50.0,
    obstacle_points=[(35.615, 139.780)]  # 障害物の位置
)

print(f"安全か: {'はい' if safety['is_safe'] else 'いいえ'}")
if not safety['is_safe']:
    print(f"警告数: {len(safety['warnings'])}")
    print(f"危険ポイント数: {len(safety['dangerous_points'])}")

# 戦略のリスク評価
risk = calculator.evaluate_strategy_risk(
    route_data=strategy,
    wind_variability=0.2,            # 風の変動性 (0-1)
    tactical_difficulty='medium'     # 戦術的難易度
)

print(f"総合リスクスコア: {risk['overall_risk_score']}/100")
print(f"リスク要因: {risk['risk_factors']}")
```

## 実装サンプル

詳細な実装例については、`optimal_vmg_example_complete.py` を参照してください。このサンプルでは:

1. 風向風速データの読み込み
2. 様々な艇種の比較
3. 風向シフトの影響分析
4. スタート位置の影響分析
5. 結果のレポート生成と可視化

を行います。

## 注意事項

- 実際のレースでは、このツールによる計算結果を参考情報として使用し、最終的な判断は乗員が行ってください。
- 現実の海況、潮流、障害物などはこのモデルでは完全には考慮されていません。
- 風の変動性が高い場合、計算結果の信頼性は低くなります。

## ライセンス

このプロジェクトは [LICENSE] のもとで配布されています。

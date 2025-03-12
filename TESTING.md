# セーリング戦略分析システム - テスト実行手順

## 前提条件
- Python 3.8以上がインストールされていること
- 依存パッケージがインストールされていること（`pip install -r requirements.txt`）
- 開発用の依存パッケージがインストールされていること（`pip install -r requirements-dev.txt`）

## テスト環境のセットアップ

```bash
# プロジェクトルートディレクトリで実行
pip install -e .  # 開発モードでインストール
pip install pytest pytest-cov  # テストツールのインストール

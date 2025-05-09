# インポートウィザード使用ガイド

このガイドでは、セーリング戦略分析システムのデータインポート機能の使い方を解説します。

## 目次

1. [インポートウィザードの概要](#インポートウィザードの概要)
2. [対応ファイル形式](#対応ファイル形式)
3. [基本的なインポート手順](#基本的なインポート手順)
4. [カラムマッピング](#カラムマッピング)
5. [拡張インポートウィザード](#拡張インポートウィザード)
6. [バッチインポート](#バッチインポート)
7. [トラブルシューティング](#トラブルシューティング)

## インポートウィザードの概要

インポートウィザードは、GPS位置データを様々なフォーマットからシステムに取り込むための機能です。ステップバイステップでデータを読み込み、必要な前処理や設定を行うことができます。

システムでは以下の3種類のインポート方法が提供されています：

1. **基本インポートウィザード**: シンプルな操作でひとつのファイルをインポート
2. **拡張インポートウィザード**: 詳細な設定と高度なプレビュー機能を備えたインポート
3. **バッチインポート**: 複数ファイルを一度にインポート

## 対応ファイル形式

インポートウィザードは以下のファイル形式に対応しています：

| 形式 | 拡張子 | 説明 |
|------|--------|------|
| CSV  | .csv   | カンマ区切りのテキストファイル。様々なGPSデバイスやアプリからエクスポートされる一般的な形式 |
| GPX  | .gpx   | GPS Exchange Format。GPS情報を保存するためのXMLベースの標準形式 |
| TCX  | .tcx   | Training Center XML。Garmin社のトレーニングデータ形式 |
| FIT  | .fit   | Flexible and Interoperable Data Transfer。コンパクトなバイナリ形式 |

## 基本的なインポート手順

基本的なインポートの流れは以下の通りです：

1. **ファイルのアップロード**: サポートされている形式のファイルを選択
2. **ファイル形式の検出**: システムが自動的に形式を検出
3. **列マッピング**: CSVファイルの場合、列名と必要なフィールドのマッピングを設定
4. **メタデータの入力**: データに関する追加情報を入力
5. **プレビュー**: インポート前にデータ内容を確認
6. **インポート完了**: データが正常に取り込まれる

## カラムマッピング

CSVファイルをインポートする場合、カラムマッピングが必要になることがあります。これは、CSVファイルの列名とシステムが必要とするフィールド名が異なる場合に、対応関係を指定する機能です。

### 必須フィールド

以下のフィールドは必須です：

- **timestamp**: 時刻情報
- **latitude**: 緯度
- **longitude**: 経度

### オプションフィールド

以下のフィールドはオプションですが、あると分析の幅が広がります：

- **speed**: 速度
- **course**: 方位（進行方向）
- **elevation**: 高度
- **heart_rate**: 心拍数
- **cadence**: ケイデンス
- **power**: パワー
- **distance**: 距離
- **temperature**: 温度
- **wind_speed**: 風速
- **wind_direction**: 風向

### マッピング設定の保存

よく使うマッピング設定は保存しておくことができます。保存したマッピング設定は、次回以降のインポート時に再利用できます。

## 拡張インポートウィザード

拡張インポートウィザードは、基本インポートウィザードの機能に加えて、以下の追加機能を提供します：

1. **詳細なファイル形式選択**: 利用可能なファイル形式を視覚的に選択
2. **インポート設定のカスタマイズ**: ファイル形式ごとの詳細設定
3. **高度なデータプレビュー**: 統計情報やグラフを含む詳細なデータプレビュー
4. **データ検証**: インポート前にデータの整合性チェック
5. **位置データの可視化**: インポート結果をマップ上で確認

### 推奨される使用シーン

以下のような場合は拡張インポートウィザードの使用をお勧めします：

- 特殊な形式のデータをインポートする場合
- データに問題がある可能性がある場合
- より詳細なデータプレビューが必要な場合
- インポート設定を細かくカスタマイズしたい場合

## バッチインポート

バッチインポートは、複数のファイルを一度にインポートする機能です。以下の特徴があります：

1. **複数ファイル選択**: 複数のファイルを一度に選択
2. **形式ごとの設定**: 各ファイル形式ごとの設定を一括で適用
3. **並列処理**: インポート処理を並列化して高速化
4. **一括マージ**: 必要に応じて全データを統合
5. **詳細な結果レポート**: 成功・失敗・警告の詳細情報

### インポートモード

バッチインポートには以下の2つのモードがあります：

- **個別処理モード**: 各ファイルを別々のデータセットとしてインポート
- **マージモード**: すべてのファイルのデータを統合して1つのデータセットとしてインポート

### 並列処理設定

システムの性能に応じて、並列処理の有無と並列数を設定できます。一般的には：

- 小規模なファイル（〜10ファイル）: 並列処理を有効化し、並列数を2〜4に設定
- 大規模なファイル（10ファイル以上）: 並列処理を有効化し、並列数をCPUコア数程度に設定
- メモリ制約がある場合: 並列処理を無効化

## トラブルシューティング

### よくある問題と解決策

1. **ファイル形式が認識されない**
   - ファイル拡張子が正しいか確認してください
   - ファイルが破損していないか確認してください
   - テキストエディタでファイルを開いて内容を確認してください

2. **CSVの列マッピングでエラーが出る**
   - CSV内の必須フィールド（timestamp, latitude, longitude）が存在するか確認してください
   - 列名に特殊文字が含まれていないか確認してください
   - 列のデータ型が適切か確認してください

3. **インポートに時間がかかる**
   - ファイルサイズが大きい場合は処理に時間がかかる場合があります
   - バッチインポートでは並列数を調整してみてください
   - システムのメモリ使用量を確認してください

4. **データが正しくインポートされない**
   - タイムスタンプの形式が認識されているか確認してください
   - 緯度・経度の値が適切な範囲内か確認してください
   - インポート前のプレビューで内容を確認してください

### エラーメッセージ一覧

| エラーメッセージ | 原因 | 対処法 |
|----------------|------|--------|
| "必須列がありません" | CSVに必要な列が含まれていない | 列マッピングで正しい列を指定するか、CSVファイルを修正 |
| "タイムスタンプの変換に失敗しました" | 日付形式が認識できない | 日付形式を指定するか、CSVファイルの日付形式を修正 |
| "ファイル形式を認識できません" | サポートされていない形式またはファイル破損 | ファイル形式を確認し、必要に応じて変換 |
| "GPX/TCXの解析に失敗しました" | XMLファイルの構造が不正 | XMLファイルを確認し、必要に応じて修正 |
| "FITファイルの解析に失敗しました" | FITファイルの構造が不正または非対応バージョン | 別のツールでFITファイルをGPXやCSVに変換してみる |

### サポートされていないファイル形式の変換

サポートされていないファイル形式は、以下のツールを使って変換できます：

- [GPSBabel](https://www.gpsbabel.org/): 様々なGPS形式の相互変換が可能
- [Garmin BaseCamp](https://www.garmin.com/): Garmin製品のデータ変換
- [GPX Editor](https://gpx-editor.app/): GPXファイルの編集と変換

---

このガイドが、セーリング戦略分析システムでのデータインポートに役立つことを願っています。さらに詳細な情報やサポートが必要な場合は、システム管理者にお問い合わせください。

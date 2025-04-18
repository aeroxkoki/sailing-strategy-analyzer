# セーリング戦略分析システム - エンドツーエンドテスト計画

## 概要

このドキュメントは、セーリング戦略分析システムの統合テストフェーズにおけるエンドツーエンドテストの計画です。フロントエンドとバックエンドの完全な統合を検証し、主要なユーザーフローが期待通りに動作することを確認します。

## テスト環境

- **バックエンドURL**: http://localhost:8000
- **フロントエンドURL**: http://localhost:3000
- **テストデータ**: `/test_data/` ディレクトリ内のサンプルCSVとGPXファイル
- **テストユーザーアカウント**: `test@example.com` / `testpassword123`

## テスト実行手順

1. バックエンドサーバー起動:
   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. フロントエンドサーバー起動:
   ```bash
   cd frontend
   npm run dev
   ```

3. 以下のテストケースを順番に実行し、結果を記録する

## テストケース

### 1. ユーザー認証フロー

#### 1.1 新規ユーザー登録

**手順**:
1. ホームページを開く
2. 「登録」リンクをクリック
3. 名前、メールアドレス、パスワードを入力
4. 「登録」ボタンをクリック

**期待結果**:
- 登録成功メッセージが表示される
- ダッシュボードページにリダイレクトされる
- ユーザー名がヘッダーに表示される

**検証項目**:
- [ ] 登録成功
- [ ] リダイレクト動作
- [ ] ユーザー情報表示

#### 1.2 ログアウト

**手順**:
1. ヘッダーのユーザーメニューを開く
2. 「ログアウト」をクリック

**期待結果**:
- ログアウト成功メッセージが表示される
- ホームページにリダイレクトされる
- ログイン/登録リンクが表示される

**検証項目**:
- [ ] ログアウト成功
- [ ] リダイレクト動作
- [ ] UI状態の変化

#### 1.3 ログイン

**手順**:
1. ホームページを開く
2. 「ログイン」リンクをクリック
3. メールアドレスとパスワードを入力
4. 「ログイン」ボタンをクリック

**期待結果**:
- ログイン成功メッセージが表示される
- ダッシュボードページにリダイレクトされる
- ユーザー名がヘッダーに表示される

**検証項目**:
- [ ] ログイン成功
- [ ] リダイレクト動作
- [ ] ユーザー情報表示

#### 1.4 保護されたリソースへのアクセス

**手順**:
1. ログアウト状態で「プロジェクト」ページへのアクセスを試みる

**期待結果**:
- ログインページにリダイレクトされる
- アクセス制限メッセージが表示される

**検証項目**:
- [ ] リダイレクト動作
- [ ] エラーメッセージ表示

### 2. プロジェクト管理フロー

#### 2.1 プロジェクト作成

**手順**:
1. ダッシュボードから「新規プロジェクト」ボタンをクリック
2. プロジェクト名「テスト走行分析」、説明「エンドツーエンドテスト用プロジェクト」を入力
3. 「作成」ボタンをクリック

**期待結果**:
- プロジェクト作成成功メッセージが表示される
- プロジェクト詳細ページにリダイレクトされる
- 入力した内容が正しく表示される

**検証項目**:
- [ ] プロジェクト作成成功
- [ ] リダイレクト動作
- [ ] プロジェクト情報表示
- [ ] 日本語表示の正常性

#### 2.2 プロジェクト一覧表示

**手順**:
1. サイドバーから「プロジェクト」リンクをクリック

**期待結果**:
- プロジェクト一覧ページが表示される
- 作成したプロジェクトが一覧に表示される
- プロジェクトカードに名前と説明が正しく表示される

**検証項目**:
- [ ] プロジェクト一覧表示
- [ ] プロジェクトカード表示
- [ ] 日本語表示の正常性

#### 2.3 プロジェクト編集

**手順**:
1. プロジェクト詳細ページから「編集」ボタンをクリック
2. プロジェクト名を「テスト走行分析（更新）」に変更
3. 「保存」ボタンをクリック

**期待結果**:
- 更新成功メッセージが表示される
- 変更内容が反映される

**検証項目**:
- [ ] プロジェクト更新成功
- [ ] 更新内容の反映
- [ ] 日本語表示の正常性

### 3. セッション管理フロー

#### 3.1 セッション作成

**手順**:
1. プロジェクト詳細ページから「新規セッション」ボタンをクリック
2. セッション名「テストセッション」、説明「エンドツーエンドテスト用セッション」を入力
3. 「作成」ボタンをクリック

**期待結果**:
- セッション作成成功メッセージが表示される
- セッション詳細ページにリダイレクトされる
- 入力した内容が正しく表示される

**検証項目**:
- [ ] セッション作成成功
- [ ] リダイレクト動作
- [ ] セッション情報表示
- [ ] 日本語表示の正常性

#### 3.2 GPSデータインポート

**手順**:
1. セッション詳細ページから「データインポート」ボタンをクリック
2. テストデータから適切なCSVファイルを選択
3. 「インポート」ボタンをクリック

**期待結果**:
- インポート成功メッセージが表示される
- セッション詳細ページにデータサマリーが表示される
- GPSトラックのプレビューが表示される

**検証項目**:
- [ ] データインポート成功
- [ ] データサマリー表示
- [ ] プレビュー表示
- [ ] 日本語表示の正常性

#### 3.3 セッションタグ管理

**手順**:
1. セッション詳細ページから「タグ編集」ボタンをクリック
2. 「テスト」「走行」タグを追加
3. 「保存」ボタンをクリック

**期待結果**:
- タグ更新成功メッセージが表示される
- 追加したタグがセッション詳細ページに表示される

**検証項目**:
- [ ] タグ更新成功
- [ ] タグ表示
- [ ] 日本語表示の正常性

### 4. データ分析フロー

#### 4.1 風向推定実行

**手順**:
1. セッション詳細ページから「風向推定」タブをクリック
2. 「推定実行」ボタンをクリック

**期待結果**:
- 処理中の進捗インジケーターが表示される
- 推定完了メッセージが表示される
- 推定結果のチャートが表示される

**検証項目**:
- [ ] 風向推定処理成功
- [ ] 進捗表示
- [ ] 結果表示

#### 4.2 戦略ポイント検出

**手順**:
1. セッション詳細ページから「戦略分析」タブをクリック
2. 「検出実行」ボタンをクリック

**期待結果**:
- 処理中の進捗インジケーターが表示される
- 検出完了メッセージが表示される
- 検出された戦略ポイントが地図上にマーカーとして表示される
- ポイントリストに検出結果が表示される

**検証項目**:
- [ ] 戦略ポイント検出成功
- [ ] 進捗表示
- [ ] マップ表示
- [ ] リスト表示
- [ ] 日本語表示の正常性

#### 4.3 分析結果エクスポート

**手順**:
1. セッション詳細ページから「エクスポート」ボタンをクリック
2. 「CSV」フォーマットを選択
3. 「エクスポート」ボタンをクリック

**期待結果**:
- エクスポート成功メッセージが表示される
- CSVファイルのダウンロードが開始される
- ダウンロードされたファイルが正しいデータ形式である

**検証項目**:
- [ ] エクスポート成功
- [ ] ファイルダウンロード
- [ ] ファイル内容の正確性
- [ ] 日本語表示の正常性

### 5. レスポンシブデザインテスト

#### 5.1 デスクトップ表示

**手順**:
1. デスクトップブラウザ（1920x1080）でアプリケーションにアクセス
2. 主要ページを順にナビゲート
3. 各UIコンポーネントの表示を確認

**期待結果**:
- レイアウトが崩れることなく表示される
- すべてのコンポーネントが正しく機能する

**検証項目**:
- [ ] レイアウト表示
- [ ] コンポーネント機能
- [ ] 日本語表示の正常性

#### 5.2 タブレット表示

**手順**:
1. タブレットサイズ（768x1024）でアプリケーションにアクセス
2. 主要ページを順にナビゲート
3. 各UIコンポーネントの表示を確認

**期待結果**:
- レスポンシブレイアウトが適用される
- すべてのコンポーネントが正しく機能する
- メニューが適切に折りたたまれる

**検証項目**:
- [ ] レスポンシブレイアウト
- [ ] コンポーネント機能
- [ ] ナビゲーション操作
- [ ] 日本語表示の正常性

#### 5.3 モバイル表示

**手順**:
1. モバイルサイズ（375x667）でアプリケーションにアクセス
2. 主要ページを順にナビゲート
3. 各UIコンポーネントの表示を確認

**期待結果**:
- モバイルレイアウトが適用される
- すべてのコンポーネントが利用可能である
- ハンバーガーメニューが使用される

**検証項目**:
- [ ] モバイルレイアウト
- [ ] コンポーネント機能
- [ ] ハンバーガーメニュー
- [ ] 日本語表示の正常性

## テスト結果記録フォーマット

各テストケースについて以下の情報を記録します：

- テスト実行日時
- テスト実行者
- テスト結果（成功/失敗）
- 問題が発生した場合はスクリーンショットと詳細説明
- 対応策（問題が発生した場合）

## 優先度の高い問題

以下の問題は特に優先度が高いと考えられます：

1. 認証系の問題（ログイン、ログアウト、保護されたリソースへのアクセス）
2. データの永続化に関する問題
3. 日本語エンコーディングに関する問題
4. API通信エラー
5. データの整合性問題

これらの問題が発生した場合は、他のテストを継続する前に解決するか、回避策を確立しておく必要があります。

## テスト後の対応

テスト結果に基づいて、以下の対応を行います：

1. 優先度に基づいた問題リストの作成
2. 本番リリース前に修正が必要な重大な問題の特定
3. 後続リリースで対応する問題の特定
4. 問題解決のためのチケット作成とアサイン
5. 再テスト計画の策定

## 本番環境デプロイ前の最終チェックリスト

テストが完了し、すべての重大な問題が解決された後、本番環境へのデプロイ前に以下の最終確認を行います：

1. 環境変数とシークレットの確認
2. 本番環境のデータベースマイグレーションの検証
3. セキュリティチェック（APIキーの露出がないか等）
4. パフォーマンスの最終確認
5. CORSの設定確認
6. ビルドプロセスの確認
7. ロギングの確認
8. エラーハンドリングの確認
9. バックアッププロセスの確認
10. ロールバック手順の確認

---

**作成日**: 2025年4月18日  
**作成者**: セーリング戦略分析システム開発チーム

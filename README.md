# Nursery Face Attendance System

**⚠️ 注意：本プロジェクトは現在「開発段階（β版）」です。機能の追加や仕様変更が頻繁に行われる可能性があります。**

## 概要
StreamlitとInsightFaceを活用した、保育施設向けの顔認証入退室管理システムです。
登園・退園の記録を顔認証で自動化し、現場の事務負担を軽減することを目的としています。
写真や動画による「なりすまし」を防ぐ生体検知（Liveness Detection）機能を備えており、セキュアな受付業務を実現します。

## 主な機能
- **リアルタイム顔認証スキャン**: WebRTCを利用し、ブラウザ上で高速な顔照合を行います。
- **デュアル認証ロジック**: InsightFaceとface_recognitionの2つのアルゴリズムを組み合わせ、精度の高い判定を行います。
- **生体検知 (Liveness Detection)**: MiniFASNetV2モデルを使用し、写真による不正打刻を防止します（閾値: 0.70）。
- **管理者ダッシュボード**: 園児や職員の顔写真登録、データの削除、および認証ログの確認が可能です。
- **環境変数によるセキュリティ**: 管理画面のログインパスワードは環境変数で安全に管理されます。

## 技術スタック
- **言語**: Python
- **GUI / Web**: Streamlit, streamlit-webrtc
- **AI / 画像処理**:
  - InsightFace (buffalo_s)
  - face_recognition (dlib)
  - ONNX Runtime (生体検知モデルの推論)
  - OpenCV
- **その他**: python-dotenv

## セットアップ

### 1. 必要ファイルの用意
生体検知用のモデルファイル `MiniFASNetV2.onnx` をプロジェクトルートに配置してください。

### 2. 環境変数の設定
プロジェクトルートに `.env` ファイルを作成し、管理画面用のパスワードを設定します。

```
ADMIN_PASSWORD=your_secure_password
```

### 3. ライブラリのインストール

```
pip install streamlit streamlit-webrtc opencv-python numpy insightface face_recognition onnxruntime av python-dotenv
```

### 4. 実行

```
streamlit run face_recognition.py
```

## 現在のステータスと今後の予定
現在はコアとなる認証ロジックとUIの実装が完了した**開発初期段階**です。今後は以下のアップデートを予定しています。

- 認証ログ保存機能。
- 入退室時刻の自動集計。
- Discordへのリアルタイム通知連携、物理ロックの自動解錠。

## ライセンス・プライバシーについて
`registered_faces/` ディレクトリに保存される顔写真は、プライバシーに関わる非常に重要な個人情報です。
本リポジトリの `.gitignore` 設定により、画像データが外部（GitHub等）に公開されないよう厳重に管理してください。
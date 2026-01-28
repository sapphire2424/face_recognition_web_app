# Nursery Face Attendance System

StreamlitとInsightFaceを活用した、保育施設向けの顔認証入退室管理システムです。

## 概要
登園・退園の記録を顔認証で自動化します。写真による不正（スプーフィング）を防ぐための生体検知（Liveness Detection）機能を搭載しており、安全かつ迅速な受付業務を実現します。

## 主な機能
- **リアルタイム顔認証**: InsightFaceとface_recognitionを組み合わせた高精度な照合。
- **生体検知 (Liveness Detection)**: MiniFASNetV2を用いた、写真や動画によるなりすまし防止。
- **管理者画面**: 園児・職員の顔写真登録および削除管理。
- **ログ管理**: 受付履歴の自動表示。

## 技術スタック
- **言語**: Python
- **フレームワーク**: Streamlit
- **画像処理/AI**: OpenCV, InsightFace, face_recognition, ONNX Runtime
- **ストリーミング**: streamlit-webrtc

## 実装のこだわり
現場での利用を想定し、逆光やボケへの耐性を高めるための前処理ロジックを実装しています。また、WebRTCを利用することでブラウザのみで動作し、特別なデバイスを必要としない構成にしました。
import warnings
import logging
import os
import cv2
import numpy as np
import av
import time
import queue
import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from insightface.app import FaceAnalysis
import face_recognition
import onnxruntime as ort
from dotenv import load_dotenv

# ==========================================
# 0. 設定・定数
# ==========================================
warnings.filterwarnings("ignore")
logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

INSIGHT_THRESHOLD = 0.55 # 高いほど厳しい (MAX 1.0)
DLIB_THRESHOLD = 0.47    # 低いほど厳しい (MIN 0.0)
BLUR_THRESHOLD = 40

# 生体検知の閾値
LIVENESS_THRESHOLD = 0.70
MODEL_PATH = "MiniFASNetV2.onnx"

#顔登録・削除の際のパスワード
admin_password = os.getenv("ADMIN_PASSWORD", "password")

# ==========================================
# 1. セッション管理
# ==========================================
if 'app_mode' not in st.session_state: st.session_state.app_mode = 'MENU'
if 'action_type' not in st.session_state: st.session_state.action_type = ''
if 'detected_name' not in st.session_state: st.session_state.detected_name = ''
if 'streamer_key' not in st.session_state: st.session_state.streamer_key = "scanner-init"
if 'admin_authenticated' not in st.session_state: st.session_state.admin_authenticated = False
if 'captured_frame' not in st.session_state: st.session_state.captured_frame = None
if 'auth_logs' not in st.session_state: st.session_state.auth_logs = []

# データ格納
if 'db_faces' not in st.session_state: st.session_state.db_faces = []
if 'db_embeds' not in st.session_state: st.session_state.db_embeds = None
if 'db_names' not in st.session_state: st.session_state.db_names = []

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
FACES_DIR = "registered_faces"
if not os.path.exists(FACES_DIR): os.makedirs(FACES_DIR)

@st.cache_resource
def get_result_queue(): return queue.Queue()
result_queue = get_result_queue()

def clear_queue():
    while not result_queue.empty():
        try: result_queue.get_nowait()
        except queue.Empty: break

# ==========================================
# 2. モデルロード
# ==========================================
@st.cache_resource(show_spinner=False)
def load_insightface_model():
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640)) 
    return app

try:
    face_app = load_insightface_model()
except Exception as e:
    st.error(f"InsightFace Error: {e}")
    st.stop()

# ==========================================
# 3. 生体検知モデル (公式実装完全再現版)
# ==========================================
class RealLivenessModel:
    def __init__(self, model_path):
        self.session = None
        if os.path.exists(model_path):
            try:
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
            except Exception as e:
                print(f"Liveness Model Error: {e}")

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def predict(self, img, bbox):
        """
        img: BGR画像 (0-255)
        bbox: [x1, y1, x2, y2]
        """
        if self.session is None: return True, 1.0, [0,0,0]

        h_img, w_img, _ = img.shape
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        # --- Scale 2.7 (公式仕様) ---
        scale = min((h_img - 1) / box_h, (w_img - 1) / box_w, 2.7)
        
        new_w = box_w * scale
        new_h = box_h * scale
        
        center_x = x1 + box_w / 2
        center_y = y1 + box_h / 2
        
        # --- Clipping (パディングなし) ---
        crop_x1 = max(0, int(center_x - new_w / 2))
        crop_y1 = max(0, int(center_y - new_h / 2))
        crop_x2 = min(w_img - 1, int(center_x + new_w / 2))
        crop_y2 = min(h_img - 1, int(center_y + new_h / 2))
        
        face_img = img[crop_y1 : crop_y2 + 1, crop_x1 : crop_x2 + 1]
        
        # リサイズ (80x80)
        try:
            resized = cv2.resize(face_img, (80, 80))
        except: return False, 0.0, [0,0,0]

        # --- 前処理 (BGR, float32, No-Norm) ---
        face_blob = resized.astype(np.float32)
        face_blob = np.transpose(face_blob, (2, 0, 1))
        face_blob = np.expand_dims(face_blob, axis=0)

        # --- 推論 ---
        try:
            outputs = self.session.run([self.output_name], {self.input_name: face_blob})
            logits = outputs[0]
            probs = self._softmax(logits)
            
            # label_idx = 1 が "Real"
            real_score = float(probs[0, 1]) 
            is_real = real_score > LIVENESS_THRESHOLD
            
            return is_real, real_score, probs[0].tolist()
            
        except Exception as e:
            print(f"Inference Error: {e}")
            return False, 0.0, [0,0,0]

@st.cache_resource(show_spinner=False)
def load_liveness_model():
    return RealLivenessModel(MODEL_PATH)

liveness_model = load_liveness_model()

# ==========================================
# 4. データ管理
# ==========================================
def load_face_data_from_disk():
    loaded_data = []
    if not os.path.exists(FACES_DIR): return []
    for filename in os.listdir(FACES_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(FACES_DIR, filename)
            try:
                img_bgr = cv2.imread(filepath)
                if img_bgr is None: continue
                # 登録用はRGB変換
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                insight_faces = face_app.get(img_bgr)
                if len(insight_faces) != 1: continue
                insight_embed = insight_faces[0].normed_embedding
                dlib_encodings = face_recognition.face_encodings(img_rgb)
                if len(dlib_encodings) != 1: continue
                dlib_enc = dlib_encodings[0]
                loaded_data.append({
                    "name": os.path.splitext(filename)[0],
                    "insight": insight_embed,
                    "dlib": dlib_enc
                })
            except Exception: pass
    return loaded_data

def update_session_state():
    data = load_face_data_from_disk()
    st.session_state.db_faces = data
    if data:
        st.session_state.db_embeds = np.array([d["insight"] for d in data])
        st.session_state.db_names = [d["name"] for d in data]
    else:
        st.session_state.db_embeds = None
        st.session_state.db_names = []

if not st.session_state.db_faces and len(os.listdir(FACES_DIR)) > 0:
    if st.session_state.db_names == []:
        update_session_state()

def check_and_register_face():
    if st.session_state.captured_frame is None: return
    frame = np.ascontiguousarray(st.session_state.captured_frame)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        faces_insight = face_app.get(frame)
        faces_dlib = face_recognition.face_locations(img_rgb)
    except Exception: return
    
    if len(faces_insight) == 1 and len(faces_dlib) == 1:
        st.success("✅ 顔を正常に検知しました。登録可能です。")
        with st.form("register_form"):
            new_name = st.text_input("名前を入力 (例: tanaka)")
            if st.form_submit_button("登録する", type="primary") and new_name:
                cv2.imwrite(os.path.join(FACES_DIR, f"{new_name}.jpg"), st.session_state.captured_frame)
                st.success(f"✨ {new_name} さんを登録しました！")
                st.session_state.captured_frame = None
                update_session_state()
                time.sleep(1)
                st.rerun()
    elif len(faces_insight) == 0: st.error("❌ 顔が検出されませんでした。")
    else: st.error("❌ 複数の顔が検出されました。")

# ==========================================
# 5. 映像処理クラス (連続判定削除・詳細ログ追加)
# ==========================================
class DualAuthProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_process_time = 0
        self.process_interval = 0.3
        self.queue = get_result_queue()
        self.start_time = time.time()
        self.unknown_count = 0
        
        self.face_embeds = None
        self.face_names = []
        self.face_db = []
        self.liveness_model = liveness_model
        
        self.status_color = (255, 0, 0)
        self.status_text = ""

    def _log(self, msg):
        try: self.queue.put_nowait(("LOG", f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"))
        except: pass

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img_bgr = frame.to_ndarray(format="bgr24")
            if img_bgr is None: return frame
            
            # ズーム処理
            h, w, _ = img_bgr.shape
            ZOOM_LEVEL = 1.3
            cy, cx = h // 2, w // 2
            nh, nw = int(h / ZOOM_LEVEL), int(w / ZOOM_LEVEL)
            y1, x1 = max(0, cy - nh // 2), max(0, cx - nw // 2)
            y2, x2 = min(h, cy + nh // 2), min(w, cx + nw // 2)
            process_img = np.ascontiguousarray(img_bgr[y1:y2, x1:x2])
            
            if time.time() - self.start_time > 45:
                try: self.queue.put(("ERROR", "TIMEOUT"))
                except: pass
                return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

            if time.time() - self.last_process_time > self.process_interval:
                self.status_color = (100, 100, 100)
                gray = cv2.cvtColor(process_img, cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(gray, cv2.CV_64F).var() >= BLUR_THRESHOLD:
                    try: faces = face_app.get(process_img)
                    except: faces = []

                    if len(faces) > 0:
                        target_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
                        bbox = target_face.bbox.astype(int)
                        
                        # --- 1. 生体検知 ---
                        is_real, live_score, prob_list = self.liveness_model.predict(process_img, bbox)
                        
                        # デバッグ数値をUIへ送信
                        try: self.queue.put_nowait(("DEBUG_PROBS", prob_list))
                        except: pass

                        if not is_real:
                            self._log(f"⚠️ Spoof: {live_score:.2f}")
                            self.status_color = (0, 0, 255)
                            self.status_text = f"SPOOF {live_score:.2f}"
                        else:
                            # --- 2. 顔認証 (連続判定なしで即実行) ---
                            self._log(f"🛡️ Liveness OK! ({live_score:.2f})")
                            
                            candidate = "Unknown"
                            insight_score = 0.0
                            best_idx = -1
                            
                            if self.face_embeds is not None and len(self.face_embeds) > 0:
                                sims = np.dot(self.face_embeds, target_face.normed_embedding)
                                best_idx = np.argmax(sims)
                                insight_score = sims[best_idx]
                                candidate = self.face_names[best_idx]

                            if insight_score >= INSIGHT_THRESHOLD:
                                # dlib詳細チェック
                                try:
                                    bx1, by1, bx2, by2 = bbox
                                    css = (by1, bx2, by2, bx1)
                                    rgb = cv2.cvtColor(process_img, cv2.COLOR_BGR2RGB)
                                    encs = face_recognition.face_encodings(rgb, known_face_locations=[css])
                                    
                                    if encs:
                                        dlib_dist = face_recognition.face_distance([self.face_db[best_idx]["dlib"]], encs[0])[0]
                                        
                                        # ★ ここで両方のスコアをログ出力 ★
                                        log_msg = f"判定中: {candidate} (Ins:{insight_score:.3f}, Dlib:{dlib_dist:.3f})"
                                        self._log(log_msg)
                                        
                                        if dlib_dist <= DLIB_THRESHOLD:
                                            self._log(f"✅ 認証成功: {candidate}")
                                            self.queue.put(("SUCCESS", candidate))
                                            self.status_color = (0, 255, 0)
                                            self.status_text = "OK"
                                        else:
                                            self.status_color = (0, 0, 255)
                                            self.status_text = "Diff Person"
                                            self.unknown_count += 1
                                    else:
                                        self._log("dlib特徴抽出失敗")
                                        self.unknown_count += 1
                                except Exception as e: 
                                    self.unknown_count += 1
                            else:
                                self._log(f"類似度不足 (Ins:{insight_score:.3f})")
                                self.status_text = "Unknown"
                                self.unknown_count += 1
                self.last_process_time = time.time()
            
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), self.status_color, 3)
            if self.status_text:
                cv2.putText(img_bgr, self.status_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.status_color, 2)
            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")
        except: return frame

class CaptureProcessor(VideoProcessorBase):
    def __init__(self): self.latest_frame = None
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = np.ascontiguousarray(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 6. UI
# ==========================================
st.set_page_config(page_title="保育園システム", layout="wide") 
st.markdown("""<style>.element-container div { margin-bottom: 0px; }</style>""", unsafe_allow_html=True)

def reset_to_menu():
    st.session_state.app_mode = 'MENU'
    st.session_state.captured_frame = None
    st.rerun()

if st.session_state.app_mode == 'MENU':
    st.title("保育園 入退室システム")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("☀️入室", type="primary", use_container_width=True):
            clear_queue()
            st.session_state.streamer_key = f"scanner-{time.time()}"
            st.session_state.action_type = "入室"
            st.session_state.app_mode = 'SCAN'
            st.rerun()
    with c2:
        if st.button("🌙退室", type="secondary", use_container_width=True):
            clear_queue()
            st.session_state.streamer_key = f"scanner-{time.time()}"
            st.session_state.action_type = "退室"
            st.session_state.app_mode = 'SCAN'
            st.rerun()
    st.divider()
    st.subheader("📋 ログ")
    st.text_area("Log", "\n".join(st.session_state.auth_logs), height=150)
    st.divider()
    c3, c4 = st.columns(2)
    with c3: 
        if st.button("🔒 管理画面"): st.session_state.app_mode = 'ADMIN_LOGIN'; st.rerun()
    with c4:
        if st.button("⚙️ 設定"): st.session_state.app_mode = 'SETTINGS'; st.rerun()

elif st.session_state.app_mode == 'SCAN':
    st.title(f"{st.session_state.action_type} 受付")
    
    col_main, col_side = st.columns([3, 1])
    
    with col_main:
        if not os.path.exists(MODEL_PATH): st.error("⚠️ MiniFASNetV2.onnx がありません！")
        
        ctx = webrtc_streamer(
            key=st.session_state.streamer_key,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=DualAuthProcessor,
            desired_playing_state=True,
            media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
            video_html_attrs={"style": {"width": "100%", "borderRadius": "10px"}, "autoPlay": True, "muted": True},
        )
        
        if ctx.video_processor:
            ctx.video_processor.face_embeds = st.session_state.db_embeds if st.session_state.db_embeds is not None else np.array([])
            ctx.video_processor.face_names = st.session_state.db_names
            ctx.video_processor.face_db = st.session_state.db_faces
            ctx.video_processor.liveness_model = liveness_model

        if st.button("キャンセル", use_container_width=True): reset_to_menu()

    with col_side:
        st.write("### 👁️ 生体検知詳細")
        st.info("AI判定の生データ (Real > 0.70)")
        debug_placeholder = st.empty()
        
        st.write("### 📝 ログ")
        log_placeholder = st.empty()

    if ctx.state.playing:
        while True:
            try:
                item = result_queue.get(timeout=0.1)
                if isinstance(item, tuple):
                    status, data = item
                    if status == "SUCCESS":
                        st.session_state.detected_name = data
                        st.session_state.app_mode = 'RESULT'
                        st.rerun()
                        break
                    elif status == "ERROR":
                        st.session_state.app_mode = 'ERROR'
                        st.rerun()
                        break
                    elif status == "LOG":
                        st.session_state.auth_logs.insert(0, data)
                        st.session_state.auth_logs = st.session_state.auth_logs[:20]
                        log_placeholder.text_area("Log", "\n".join(st.session_state.auth_logs), height=200, label_visibility="collapsed")
                    elif status == "DEBUG_PROBS":
                        # 数値の詳細を表示
                        probs = data 
                        debug_text = f"""
                        - **Spoof**: {probs[0]:.4f}
                        - **Real**:  **{probs[1]:.4f}**
                        """
                        debug_placeholder.markdown(debug_text)

            except queue.Empty:
                if not ctx.state.playing: break
                continue

# --- RESULT ---
elif st.session_state.app_mode == 'RESULT':
    st.success(f"OK! {st.session_state.detected_name} 様 {st.session_state.action_type}")
    time.sleep(2)
    reset_to_menu()

# --- ERROR ---
elif st.session_state.app_mode == 'ERROR':
    st.error("認証できませんでした")
    if st.button("戻る"): reset_to_menu()

# --- ADMIN/SETTINGS ---
elif st.session_state.app_mode == 'ADMIN_LOGIN':
    st.subheader("管理者ログイン")
    if st.button("戻る"): reset_to_menu()
    pwd = st.text_input("Password", type="password")
    if st.button("Login") and pwd == admin_password:
        st.session_state.admin_authenticated = True
        st.session_state.app_mode = 'ADMIN_DASHBOARD'
        st.rerun()

elif st.session_state.app_mode == 'ADMIN_DASHBOARD':
    st.subheader("管理画面")
    if st.button("ログアウト"): st.session_state.admin_authenticated = False; reset_to_menu()
    
    tab1, tab2 = st.tabs(["登録", "削除"])
    with tab1:
        method = st.radio("Mode", ["Camera", "File"], horizontal=True)
        if method == "Camera":
            ctx = webrtc_streamer(key="adm", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_processor_factory=CaptureProcessor, media_stream_constraints={"video": True, "audio": False})
            if ctx.state.playing and st.button("撮影"):
                if ctx.video_processor and ctx.video_processor.latest_frame is not None:
                    st.session_state.captured_frame = ctx.video_processor.latest_frame
                    st.rerun()
            if st.session_state.captured_frame is not None:
                st.image(st.session_state.captured_frame, channels="BGR")
                check_and_register_face()
                if st.button("クリア"): st.session_state.captured_frame = None; st.rerun()
        else:
            uf = st.file_uploader("Img")
            if uf:
                img = cv2.imdecode(np.frombuffer(uf.read(), np.uint8), 1)
                st.session_state.captured_frame = img
                st.image(img, channels="BGR")
                check_and_register_face()

    with tab2:
        for f in os.listdir(FACES_DIR):
            if f.endswith('jpg'):
                c1, c2 = st.columns([4,1])
                c1.write(f)
                if c2.button("Del", key=f): os.remove(os.path.join(FACES_DIR, f)); update_session_state(); st.rerun()

elif st.session_state.app_mode == 'SETTINGS':
    st.subheader("カメラ確認")
    webrtc_streamer(key="set", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION)
    if st.button("戻る"): reset_to_menu()
# app.py

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# ---------------------- Page Config & Styles ----------------------
st.set_page_config(page_title="🤟 Sign Language to Text Converter", layout="wide")
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #43cea2, #185a9d);
        color: white;
    }
    .main {
        background: transparent;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✨ Real-Time Sign Language ➜ Text Converter")
st.markdown("""
Try these hand signs:
- 👍 Thumbs Up
- 👎 Thumbs Down
- ✌️ Peace
- 👌 OK
- 🤙 Call Me
- 👋 Hello
- ✊ Fist
- 🖐️ Palm
""")

run = st.checkbox('✅ Start Camera', value=True)

# ---------------------- Mediapipe Setup ----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ---------------------- Emoji Map ----------------------
SIGN_GIF_MAP = {
    "👍 Thumbs Up": "https://media.giphy.com/media/26BRuo6sLetdllPAQ/giphy.gif",
    "👎 Thumbs Down": "https://media.giphy.com/media/l2Je0GS8yOjJTP9yY/giphy.gif",
    "✌️ Peace": "https://media.giphy.com/media/3o6ZsYLrQh4Ff7DQ6s/giphy.gif",
    "👌 OK": "https://media.giphy.com/media/l3q2K5jinAlChoCLS/giphy.gif",
    "🤙 Call Me": "https://media.giphy.com/media/8FenU2A44a3WTqjLeU/giphy.gif",
    "👋 Hello": "https://media.giphy.com/media/ASd0Ukj0y3qMM/giphy.gif",
    "✊ Fist": "https://media.giphy.com/media/l0MYC0LajbaPoEADu/giphy.gif",
    "🖐️ Open Palm": "https://media.giphy.com/media/1hAXk1JpN96D7DbDnb/giphy.gif",
    "No Sign Detected": "https://media.giphy.com/media/3o7buirYcmV5nSwIRW/giphy.gif"
}

# ---------------------- Helper: Finger State ----------------------
def finger_is_open(lm, tip_id, pip_id):
    return lm.landmark[tip_id].y < lm.landmark[pip_id].y

# ---------------------- Gesture Rules ----------------------
def detect_gesture(lm):
    fingers = []
    fingers.append(finger_is_open(lm, mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP))
    fingers.append(finger_is_open(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP))
    fingers.append(finger_is_open(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP))
    fingers.append(finger_is_open(lm, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP))
    fingers.append(finger_is_open(lm, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP))

    if fingers == [1, 0, 0, 0, 0]:
        return "👍 Thumbs Up"
    if fingers == [0, 1, 1, 0, 0]:
        return "✌️ Peace"
    if fingers == [1, 1, 0, 0, 0]:
        return "🤙 Call Me"
    if fingers == [1, 1, 1, 1, 1]:
        return "🖐️ Open Palm"
    if fingers == [0, 0, 0, 0, 0]:
        return "✊ Fist"
    if fingers == [0, 1, 1, 1, 1]:
        return "👋 Hello"
    if fingers == [0, 0, 0, 0, 1]:
        return "👌 OK"
    return "No Sign Detected"

# ---------------------- Session State ----------------------
if 'last_pos' not in st.session_state:
    st.session_state.last_pos = None

MOVEMENT_THRESHOLD = 0.02

# ---------------------- Camera Loop ----------------------
frame_window = st.image([])
display_placeholder = st.empty()
movement_placeholder = st.sidebar.empty()

cap = cv2.VideoCapture(0)
prev_time = 0

while run:
    current_time = time.time()
    if current_time - prev_time < 0.05:
        continue
    prev_time = current_time

    ret, frame = cap.read()
    if not ret:
        st.warning("🚫 Webcam not found.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    sign = "No Sign Detected"
    moving = False

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            sign = detect_gesture(lm)

            wrist = lm.landmark[mp_hands.HandLandmark.WRIST]
            current_pos = np.array([wrist.x, wrist.y])
            if st.session_state.last_pos is not None:
                movement = np.linalg.norm(current_pos - st.session_state.last_pos)
                if movement > MOVEMENT_THRESHOLD:
                    moving = True
            st.session_state.last_pos = current_pos

    frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # Always update the sign text + emoji on RIGHT
    display_placeholder.markdown(
        f"""
        <div style="display: flex; justify-content: flex-end; align-items: center;">
            <img src="{SIGN_GIF_MAP.get(sign, SIGN_GIF_MAP['No Sign Detected'])}" alt="emoji" width="100">
            <h2 style='color: #FFD700; margin-left: 20px; text-align: right;'>{sign}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    movement_text = "🟢 Moving" if moving else "🔵 Still"
    movement_placeholder.markdown(f"### Hand Status: {movement_text}")

cap.release()
cv2.destroyAllWindows()

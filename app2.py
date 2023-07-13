import streamlit as st
import av
import numpy as np
import mediapipe as mp
import cv2
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode

import pandas as pd
import pickle

from landmarks import landmarks

st.title("Deadlift tracker")


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Load the machine learning model
with open('./deadlift.pkl', 'rb') as f:
    model = pickle.load(f)


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.model = model
        self.current_stage = ''
        self.counter = 0
        self.bodylang_prob = np.array([0, 0])
        self.bodylang_class = ''

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        results = pose.process(image)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                                  mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks)
            self.bodylang_prob = self.model.predict_proba(X)[0]
            self.bodylang_class = self.model.predict(X)[0]

            if self.bodylang_class == "down" and self.bodylang_prob[self.bodylang_prob.argmax()] > 0.7:
                if self.current_stage != "down":
                    self.current_stage = "down"
                    self.counter += 1
            elif self.current_stage == "down" and self.bodylang_class == "up" and self.bodylang_prob[self.bodylang_prob.argmax()] > 0.7:
                self.current_stage = "up"

        except Exception as e:
            print(e)

        img = Image.fromarray(image)
        img_with_annotations = np.array(img)
        return img_with_annotations


webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=VideoTransformer,
    async_transform=True,
)

if webrtc_ctx.video_transformer:
    st.write("Started video transformer")

    while True:
        video_transformer = webrtc_ctx.video_transformer
        counter_text = st.empty()
        prob_text = st.empty()
        stage_text = st.empty()

        counter_text.text(f"REPS: {video_transformer.counter}")
        prob_text.text(f"PROB: {video_transformer.bodylang_prob[video_transformer.bodylang_prob.argmax()]}")
        stage_text.text(f"STAGE: {video_transformer.current_stage}")

else:
    st.write("WebRTC not available")

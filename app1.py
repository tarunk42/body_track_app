import streamlit as st
import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
from PIL import Image

from landmarks import landmarks

st.title("Deadlift tracker")
current_stage = ''
counter = 0
bodylang_prob = np.array([0, 0])
bodylang_class = ''

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

with open('./deadlift.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)


def reset_counter():
    global counter
    counter = 0


st.button("RESET", on_click=reset_counter)

frame_placeholder = st.empty()
class_box = st.empty()
counter_box = st.empty()
prob_box = st.empty()

# Set desired dimensions for the video frame
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

while True:
    ret, frame = cap.read()
    
    # Flip the frame horizontally to mirror the camera
    frame = cv2.flip(frame, 1)
    
    # Resize the frame to desired dimensions
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                              mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

    try:
        row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row], columns=landmarks)
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0]

        if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "down"
        elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "up"
            counter += 1

    except Exception as e:
        print(e)

    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    frame_placeholder.image(imgarr)

    counter_box.text(f"REPS: {counter}")
    prob_box.text(f"PROB: {bodylang_prob[bodylang_prob.argmax()]}")
    class_box.text(f"STAGE: {current_stage}")

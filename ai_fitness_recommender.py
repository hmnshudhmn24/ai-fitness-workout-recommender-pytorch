
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
import random

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize Pose Model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Recommendations based on posture
recommendations = [
    "Try squats to strengthen your lower body!",
    "Push-ups can help improve your upper body strength.",
    "Consider planks for core stability.",
    "Great posture! Try jumping jacks for cardio.",
    "Bicep curls could enhance your arm strength."
]

def get_pose_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        return landmarks
    return None

def check_posture_and_recommend(landmarks):
    # Simple heuristic: check visibility of shoulders and hips
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    if all(lm.visibility > 0.5 for lm in [left_shoulder, right_shoulder, left_hip, right_hip]):
        return random.choice(recommendations)
    else:
        return "Please adjust your position for better detection."

def main():
    st.title("🏋️‍♂️ AI Fitness Workout Recommender")
    st.write("Get real-time posture feedback and workout suggestions!")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        success, frame = camera.read()
        if not success:
            st.write("Error reading from webcam.")
            break

        frame = cv2.flip(frame, 1)
        landmarks = get_pose_landmarks(frame)

        if landmarks:
            rec = check_posture_and_recommend(landmarks)
            cv2.putText(frame, rec, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            mp_drawing.draw_landmarks(frame, pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            cv2.putText(frame, "No pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()

if __name__ == "__main__":
    main()

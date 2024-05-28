import cv2
import streamlit as st
import numpy as np
import torch
from super_gradients.training import models
import time
import pygame
from gtts import gTTS
import io
import random
import tempfile
import os

def load_model():
    model = models.get('yolo_nas_s', num_classes=1, checkpoint_path='ckpt_best (1).pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, device

def detect_objects(input_video_path, output_video_path, model, device, confidence_threshold=0.2):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected_frame, num_humans = detect_objects_in_frame(frame, model, device)

        if num_humans > 0:
            # Save frame
            out.write(detected_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def detect_objects_in_frame(frame, model, device, confidence_threshold=0.2):
    detected_frame = frame.copy()

    # Preprocess the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        results = model.predict(img)

    # Process the inference results
    bboxes = results.prediction.bboxes_xyxy
    class_names = results.class_names
    num_humans = 0
    for bbox in bboxes:
        num_humans += 1
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = 'Human'
        cv2.putText(detected_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 255, 254), 2)
    return detected_frame, num_humans

def main():
    st.title('Human Detection in Thermal Videos')
    uploaded_file = st.file_uploader("Upload Thermal Video...", type=["mp4"])

    if uploaded_file is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

        model, device = load_model()

        output_temp_file = tempfile.NamedTemporaryFile(delete=False)
        output_video_path = output_temp_file.name

        # Add a spinner while processing
        with st.spinner('Processing...'):
            detect_objects(video_path, output_video_path, model, device)

        num_humans = count_humans_in_video(output_video_path)
        if num_humans > 0:
            st.video(output_video_path)
        else:
            st.write("No humans detected in the uploaded video.")

        # Clean up temporary files
        temp_video.close()
        output_temp_file.close()
        os.unlink(temp_video.name)
        os.unlink(output_video_path)

def count_humans_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected_frame, num_humans = detect_objects_in_frame(frame, model, device)
        count += num_humans

    cap.release()
    cv2.destroyAllWindows()

    return count

if __name__ == "__main__":
    main()

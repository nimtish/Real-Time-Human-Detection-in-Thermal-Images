import streamlit as st
# Actual content of the app starts here
import cv2
import numpy as np
from super_gradients.training import models
import time
import pygame
from gtts import gTTS
import io
import random

def load_model():
    model = models.get('yolo_nas_s', num_classes=1, checkpoint_path='ckpt_best (1).pth')
    return model

def detect_objects(image, model):
    detected_image = image
    results = model.predict(detected_image)
    bboxes = results.prediction.bboxes_xyxy
    class_names = results.class_names
    num_humans = 0
    for bbox in bboxes:
        num_humans += 1
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(detected_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = 'Human'
        cv2.putText(detected_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 255, 254), 2)
    return detected_image, num_humans

def main():
    st.title('ThermoSense: Human Detection in Thermal Images')
    st.write("Created and Developed By: IT Group 19")
    st.write("Under the Guidance of Dr.Nitin Sharma")
    uploaded_file = st.file_uploader("Upload Thermal Image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Add a spinner while processing the image
        with st.spinner("Detecting humans, please wait..."):
            time.sleep(1)
            detected_image, num_humans = detect_objects(image, load_model())
            st.image(detected_image, caption='Result Image', use_column_width=True)
        
        # Display result
        if num_humans > 0:
            if num_humans == 1:
                tts = gTTS(text=f'{num_humans} human detected!', lang='en')
            else:
                tts = gTTS(text=f'{num_humans} human(s) detected!', lang='en')
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            pygame.mixer.init()
            pygame.mixer.music.load(audio_bytes)
            pygame.mixer.music.play()
            # Wait for the sound to finish playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.stop()
            
            st.write(f'{num_humans} human(s) detected!')
        else:
            st.write("No humans detected.")

if __name__ == "__main__":
    main()

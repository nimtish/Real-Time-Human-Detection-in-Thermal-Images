import cv2
import streamlit as st
import numpy as np
from super_gradients.training import models
def load_model():
    model = models.get('yolo_nas_s', num_classes=1, checkpoint_path='ckpt_best (1).pth')
    return model
def detect_objects(image, model):
    detected_image = image  # Placeholder
    results = model.predict(detected_image)
    bboxes = results.prediction.bboxes_xyxy
    class_names = results.class_names
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(detected_image, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(detected_image, 'Human', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 255, 254), 5)

    return detected_image
def main():
    st.title('Object Detection with YOLO-NAS')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        model = load_model()
        detected_image = detect_objects(image, model)
        st.image(detected_image, caption='Detected Objects', use_column_width=True)
if __name__ == "__main__":
    main()

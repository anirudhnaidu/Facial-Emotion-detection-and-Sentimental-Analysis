# -*- coding: utf-8 -*-
import streamlit as st
from keras.models import load_model
from keras_preprocessing.image import img_to_array
import cv2
import numpy as np
import datetime
from transformers import pipeline  # For sentiment analysis using Hugging Face

# Inject custom CSS for styling
st.markdown("""
    <style>
    /* General Page Styles */
    body {
        background-color: #f0f2f6;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #746;
    }
    .stButton>button {
        background-color: #4CAF50; 
        color: white;
        padding: 12px 20px;
        font-size: 16px;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        cursor: pointer;
    }
    .stTextInput>div>input {
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        transition: box-shadow 0.3s ease;
    }
    .stTextInput>div>input:focus {
        box-shadow: 0px 0px 8px 0px rgba(0, 150, 136, 0.8);
    }
    .stTextInput>div>label {
        font-weight: bold;
    }
    .stMarkdown p {
        font-size: 16px;
        line-height: 1.6;
        color: #444;
    }
    .stMarkdown h1 {
        color: #4CAF50;
        font-size: 36px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stMarkdown h2 {
        color: #4CAF50;
        font-size: 28px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #777;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
emotion_model = load_model('./emotion_detection_model_50epochs.h5')
age_model = load_model('./age_model_3epochs.h5', compile=False)
gender_model = load_model('Age_and_Gender_Detection/gender_detection_model.h5')

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

# Sentiment Analysis model using Hugging Face Transformers
sentiment_analyzer = pipeline('sentiment-analysis')

# Streamlit app
st.markdown("<h1>Live Face Detection and Sentiment Analysis</h1>", unsafe_allow_html=True)
st.write("This app detects face, emotion, gender of a person and predicts the sentiment of input text.")

# Start/Stop buttons for webcam
col1, col2 = st.columns([1, 1])
with col1:
    start_button = st.button('Start Webcam', key='start')
with col2:
    stop_button = st.button('Stop Webcam', key='stop')

# Webcam Status
webcam_on = start_button

if webcam_on:
    # Start webcam
    cap = cv2.VideoCapture(0)
    frameST = st.empty()

    while webcam_on and not stop_button:
        ret, frame = cap.read()
        labels = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            # Get image ready for prediction
            roi = roi_gray.astype('float') / 255.0  # Scaling the image
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)  # Expand dims for prediction (1, 48, 48, 1)

            preds = emotion_model.predict(roi)[0]  # One hot encoded result for 7 classes
            label = class_labels[preds.argmax()]  # Find the label
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Gender
            roi_color = frame[y:y + h, x:x + w]
            roi_color = cv2.resize(roi_color, (200, 200), interpolation=cv2.INTER_AREA)
            gender_predict = gender_model.predict(np.array(roi_color).reshape(-1, 200, 200, 3))
            gender_predict = (gender_predict >= 0.5).astype(int)[:, 0]
            gender_label = gender_labels[gender_predict[0]]
            gender_label_position = (x, y + h + 50)  # 50 pixels below to move the label outside the face
            cv2.putText(frame, gender_label, gender_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Age
            age_predict = age_model.predict(np.array(roi_color).reshape(-1, 200, 200, 3))
            age = round(age_predict[0, 0])
            age_label_position = (x + h, y + h)
            cv2.putText(frame, "Age=" + str(age), age_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        frameST.image(frame, channels="BGR")

        # Update the webcam status
        webcam_on = not stop_button

    cap.release()
    cv2.destroyAllWindows()

# Sentiment Analysis Section
st.header("Sentiment Analysis")
st.write("Enter a sentence to analyze its sentiment:")

# Text input for sentiment analysis
user_input = st.text_input("Type a sentence here...")

if user_input:
    sentiment_result = sentiment_analyzer(user_input)[0]  # Get sentiment prediction
    sentiment_label = sentiment_result['label']  # e.g., 'POSITIVE' or 'NEGATIVE'
    sentiment_score = sentiment_result['score']  # Confidence score

    # Display the sentiment result
    st.write(f"Sentiment: **{sentiment_label}** (Confidence: {sentiment_score:.2f})")

# Write information file of what does this app do
st.header("Information")
st.markdown("<h1>Facial Emotion Detection, Age, and Gender Recognition with OpenCV</h1>", unsafe_allow_html=True)

st.write("""
In this project, a multi-faceted facial analysis system has been implemented, combining state-of-the-art deep learning models with the powerful computer vision library, OpenCV. The system is designed to accurately detect faces in images or video streams and provide insights into facial emotions, age, and gender.
""")

st.header("Project Overview")
st.write("""
- **Face Detection**: Utilizing the robust face detection capabilities of OpenCV, the system efficiently locates and isolates faces within images or video frames.
- **Facial Emotion Detection**: The emotion detection model recognizes a range of human emotions, such as happiness, sadness, anger, and surprise.
- **Age and Gender Recognition**: The models estimate the age and gender of detected faces, adding another layer of demographic information.
- **Sentiment Analysis**: This feature allows users to input text and receive predictions about the sentiment of the text (positive, negative, or neutral).
""")

st.header("Practical Applications")
st.write("""
The system has practical applications in various fields:
- Human-computer interaction
- Sentiment analysis for written text
- Audience engagement measurement
- Security and surveillance
""")

# Footer
st.markdown(f"<div class='footer'>© {datetime.datetime.now().year}.</div>", unsafe_allow_html=True)

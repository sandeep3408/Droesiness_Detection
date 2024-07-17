import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from playsound import playsound

# Load the pre-trained Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load your trained model
new_model = load_model('C:\\Users\\sande\\OneDrive\\Desktop\\drowsiness_detection project\\my_model.h5')


# Initialize count for closed eyes and alarm status
count_closed_eyes = 0
alarm_triggered = False

# Function to play alarm sound
def play_alarm():
    playsound('emergency-alarm-with-reverb-29431 (1).mp3')

# Streamlit app
st.title('Drowsiness Detection with Alarm')

# Checkbox to start/stop webcam
start_webcam = st.checkbox("Start Webcam")

# Initialize the video capture object for webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Main app loop
while start_webcam:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for better processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for eyes
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Display eye status text for each face based on model output
        f_img = cv2.resize(roi_color, (224, 224))
        f_img = np.expand_dims(f_img, axis=0)
        f_img = f_img / 255.0

        model_output = new_model.predict(f_img)
        print(model_output)

        # Determine eye status based on model output and threshold
        # eye_status = "Open" if model_output > 0.2 else "Closed"

        # # Update count for closed eyes and trigger alarm if count exceeds 5
        # if eye_status == "Closed":
        #     count_closed_eyes += 1
        #     if count_closed_eyes > 5 and not alarm_triggered:
        #         play_alarm()
        #         alarm_triggered = True
        # else:
        #     count_closed_eyes = 0
        #     alarm_triggered = False

        # cv2.putText(frame, f'Eye Status: {eye_status}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if len(eyes) == 0:
            eye_status = "Closed"
            count_closed_eyes += 1
            if count_closed_eyes > 5 and not alarm_triggered:
                play_alarm()
                alarm_triggered = True
            else:
                count_closed_eyes = 0
                alarm_triggered = False        

        else:
            eye_status = "Open"

        # Display eye status text for each face
        cv2.putText(frame, f'Eye Status: {eye_status}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Iterate over detected eyes and draw rectangles
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the webcam feed
    cv2.imshow('Webcam', frame)

    # Check for user interrupt (press 'q' to stop webcam)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

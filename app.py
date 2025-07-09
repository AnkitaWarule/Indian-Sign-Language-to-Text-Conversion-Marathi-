from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
import os

app = Flask(__name__)

# Load the pre-trained gesture recognition model
model = load_model("gesture_model.h5")

# Marathi translation mapping
gesture_to_marathi = {
    "hello": "नमस्कार",
    "thank_you": "धन्यवाद",
    "yes": "होय",
    "no": "नाही",
    "have_you_eatten_something": "तुम्ही काही खाल्लं आहे का?",
    "How_are_you" : "कसे आहात?",
    "Please" : "कृपया",
    "what_are_you_doing" : "तू काय करत आहेस?",
    "come_here" : "इकडे ये",
    "Wel_come" : "तुमचे स्वागत आहे.",
    "Bye" : "चला निरोप घेऊया",
    "Zero": "शून्य",
    "One":"एक",
    "Two": "दोन",
    "Three" : "तीन",
    "Four" : "चार",
    "Five" : "पाच",
    "Six" : "सहा",
    "Seven" : "सात",
    "Eight" : "आठ",
    "Nine": "नऊ"
}

gestures = list(gesture_to_marathi.keys())
img_size = (128, 128)

# Start webcam
cap = cv2.VideoCapture(0)

# Global variables to store latest results
latest_gesture = "No Gesture"
latest_translation = "काही नाही"

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    global latest_gesture, latest_translation
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (100, 100), (324, 324), (255, 0, 0), 2)
        roi = frame[100:324, 100:324]  # Region of Interest

        # Prepare image for prediction
        resized = cv2.resize(roi, img_size)
        reshaped = np.expand_dims(resized, axis=0) / 255.0

        # Prediction
        prediction = model.predict(reshaped)
        confidence_threshold = 0.8

        if np.max(prediction) < confidence_threshold:
            gesture = "No Gesture"
            marathi_translation = "काही नाही"
        else:
            gesture_idx = np.argmax(prediction)
            gesture = gestures[gesture_idx]
            marathi_translation = gesture_to_marathi.get(gesture, "काही नाही")

        # Save latest values globally
        latest_gesture = gesture
        latest_translation = marathi_translation

        # Draw Marathi text using PIL
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("Mangal.ttf", 30)
        except IOError:
            font = ImageFont.load_default()

        draw.text((10, 30), f"Gesture: {gesture}", font=font, fill=(0, 255, 0))
        draw.text((10, 60), f"Marathi: {marathi_translation}", font=font, fill=(0, 255, 255))

        frame = np.array(pil_img)

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_translation')
def get_translation():
    return jsonify({
        'gesture': latest_gesture,
        'marathi': latest_translation
    })

if __name__ == "__main__":
    app.run(debug=True)

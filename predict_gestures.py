from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (100, 100), (324, 324), (255, 0, 0), 2)
    roi = frame[100:324, 100:324]  # Region of Interest

    # Prepare the image for prediction
    resized = cv2.resize(roi, img_size)
    reshaped = np.expand_dims(resized, axis=0) / 255.0

    # Predict gesture
    prediction = model.predict(reshaped)

    # Check confidence threshold
    confidence_threshold = 0.8
    if np.max(prediction) < confidence_threshold:
        gesture = "No Gesture"
        marathi_translation = "काही नाही"
    else:
        gesture_idx = np.argmax(prediction)
        gesture = gestures[gesture_idx]
        marathi_translation = gesture_to_marathi.get(gesture, "काही नाही")

    # Convert frame to Image to use PIL for text rendering
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    # Load a font that supports Marathi characters (update path if necessary)
    try:
        font = ImageFont.truetype("Mangal.ttf", 30)  # Update with your font path
    except IOError:
        print("Font file not found!")
        font = ImageFont.load_default()  # Fall back to default font if necessary

    draw.text((10, 30), f"Gesture: {gesture}", font=font, fill=(0, 255, 0))
    draw.text((10, 60), f"Marathi: {marathi_translation}", font=font, fill=(0, 255, 255))

    # Convert PIL image back to OpenCV format
    frame = np.array(pil_img)

    # Show the frame with the prediction
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

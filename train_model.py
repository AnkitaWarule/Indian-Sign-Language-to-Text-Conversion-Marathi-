import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
img_size = (128, 128)  # Image size to resize to
gestures = ["hello", "thank_you", "yes", "no", "have_you_eatten_something","How_are_you", "Please", "what_are_you_doing", "come_here","Wel_come","Bye","Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine" ]
dataset_dir = "..dataset"  # Path to your dataset

# Preparing data
data = []
labels = []

for gesture in gestures:
    gesture_dir = os.path.join(dataset_dir, gesture)
    for img_name in os.listdir(gesture_dir):
        img_path = os.path.join(gesture_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)  # Resize image to fit model input
        data.append(img)
        labels.append(gestures.index(gesture))  # Label the gesture index

data = np.array(data) / 255.0  # Normalize image data to [0, 1]
labels = np.array(labels)

# One-hot encoding of labels
labels = to_categorical(labels, num_classes=len(gestures))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(gestures), activation='softmax')  # Output layer: one per gesture
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Apply data augmentation during training
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test))

# Save the trained model
model.save("gesture_model.h5")
print("Model saved as gesture_model.h5") 
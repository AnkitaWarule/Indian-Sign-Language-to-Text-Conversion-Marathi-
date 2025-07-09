# Indian-Sign-Language-to-Text-Conversion-Marathi-
The primary goal is to develop a system that can recognize and interpret ISL gestures using machine learning or computer vision techniques, enabling real-time translation to bridge the communication gap between the hearing-impaired and the general public.

# Modules in Project

1. Dataset Collection
• Start webcam and draw a fixed ROI box.
• For each gesture: Wait for user to press ’s’ to start. Capture 300 images from within the ROI. Resize images and save in respective gesture folder.

2.Model Training
• Load gesture images and assign labels.
• Preprocess images (resize, normalize).
• Split data into training and validation sets.
• Train a CNN model and save it as .h5 file.

3.Real-Time Prediction
• Open webcam and extract ROI frame from the feed.
• Resize and preprocess the ROI.
• Load the trained model to predict the gesture.
• Display the predicted gesture and its Marathi translation

# Software Requrirements

1. Programming Language: Python
• Python is used for building the core functionalities of the project due to its simplicity
and the wide availability of libraries for image processing, machine learning, and speech
synthesis.

2. Libraries and Frameworks:
• OpenCV: For real-time video capturing, image processing, and gesture detection.
• TensorFlow/Keras: For training and deploying the Convolutional Neural Network
(CNN) model used for gesture recognition.
• NumPy Pandas: For handling numerical data and datasets.
• Matplotlib/Seaborn: For visualizing data during model training and testing phases.
• gTTS (Google Text-to-Speech) For converting the recognized text into speech
output.
• Tkinter: For building the user interface (GUI or web-based).

3.Development Environment
• Visual Studio Code / Jupyter Notebook: IDEs used for coding, testing, and
debugging.
• Git Version control system for managing project updates and collaboration.

4.Operating System:
• Visual Studio Code / Jupyter Notebook: IDEs used for coding, testing, and
debugging.
• VCompatible with Windows 10/11, Linux (Ubuntu).


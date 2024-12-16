# Real-Time Facial Expression Recognition using Deep Learning

This project demonstrates real-time facial expression recognition using a Convolutional Neural Network (CNN) and the FER2013 dataset. The model is trained to recognize 7 facial expressions and is deployed for real-time classification using a webcam.

## Table of Contents

- [Project Description](#project-description)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Real-Time Prediction](#real-time-prediction)
- [How to Run](#how-to-run)
- [License](#license)

## Project Description

This project uses TensorFlow and Keras to build a deep learning model for facial expression recognition. The model is trained on the FER2013 dataset and can classify 7 facial emotions in real-time using a webcam. The face detection is handled using the **MediaPipe** library.

## Dependencies

- TensorFlow >= 2.x
- Keras
- OpenCV
- MediaPipe
- NumPy
- Matplotlib (optional, for plotting training history)
- Scikit-learn (optional, for additional metrics)

## Dataset

The model is trained using the **FER2013** dataset, which contains images of facial expressions labeled with one of the following 7 emotions:
1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

You can download the FER2013 dataset from [Kaggle](https://www.kaggle.com/datasets/jangedoo/fer2013) or use your own dataset.

## Model Architecture

The model consists of the following layers:
- **Conv2D** layers for feature extraction using different filters
- **MaxPooling2D** layers for dimensionality reduction
- **Dropout** layers to prevent overfitting
- **Dense** layers for classification
- Output layer with **Softmax** activation to classify into 7 categories

### CNN Architecture:
1. 2 Conv2D layers (64 filters) followed by MaxPooling and Dropout
2. 2 Conv2D layers (128 filters)
3. 2 Conv2D layers (256 filters) followed by MaxPooling and Dropout
4. Flatten the output and pass it through a Dense layer with L2 regularization
5. Final output layer with 7 neurons corresponding to the 7 facial expressions

## Training the Model

The model is trained using the **FER2013** dataset and a **categorical crossentropy** loss function. The optimizer used is **Adam** with a learning rate of 0.0001. Early stopping is applied during training to avoid overfitting.

To train the model, use the following steps:
1. Download and preprocess the FER2013 dataset.
2. Split the dataset into training and testing sets.
3. Train the model using the following command:
   ```python
   model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test), callbacks=[early_stopping])


##Real-Time Prediction
Once the model is trained and saved, you can use it for real-time facial expression recognition using your webcam. The MediaPipe library is used for detecting faces in the video stream.

## Key Functions:
preprocess_face(face): This function preprocesses the detected face by resizing, converting to grayscale, and normalizing it before passing it into the model for prediction.
The model.predict() function returns a prediction, and the emotion label is displayed on the video feed.
To run the real-time webcam detection, use the following code:

# Initialize webcam and MediaPipe Face Detection
cap = cv2.VideoCapture(0)
while True:
    # Capture and process the frame
    # Predict emotion and display on screen

##How to Run

Install dependencies:

Download and prepare the FER2013 dataset (if you haven't done so already).

Press 'q' to quit the webcam feed.


### **Important Notes:**

- **Model Loading**: Ensure the model path (`fer2013.h5`) is correctly specified when loading the pre-trained model.
- **Face Detection**: MediaPipe's face detection requires a good webcam setup and lighting conditions to work effectively.
- **Real-Time Processing**: This project is designed for real-time usage, so it might be affected by the hardware (e.g., CPU/GPU performance).
  
I hope this helps you set up your GitHub repository and clarify the project. Let me know if you need any further adjustments!

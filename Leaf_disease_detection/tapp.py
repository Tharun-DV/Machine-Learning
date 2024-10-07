import os
import cv2
import numpy as np
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def preprocess_image(image_path, size=(128, 128)):
   image = cv2.imread(image_path)
   if image is not None:
    image = cv2.resize(image, size)
    image = image.astype('float32') / 255.0
    image = image.flatten()
    return image

def load_dataset(image_folder):
    images = []
    labels = []
    
    for label_folder in os.listdir(image_folder):
        label_path = os.path.join(image_folder, label_folder)
        label = 1 if label_folder == 'diseased' else 0 
        
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            image = preprocess_image(image_path)
            if image is not None:
                images.append(image)
                labels.append(label)
    
    return np.array(images), np.array(labels)

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(X_train, y_train)
    
    return scaler, svm_model

def predict_image(image_path, scaler, svm_model):
    image = preprocess_image(image_path)
    if image is not None:
        image = scaler.transform([image]) 
        prediction = svm_model.predict(image)
        return "Diseased" if prediction[0] == 1 else "Healthy"
    else:
        return "Invalid image"

def main():
    st.title("Plant Disease Classification")
    image_folder = '/Users/tharundv/Projects/test/ML/Leaf_disease_detection/train/'  # Replace with your dataset folder patIh
    X, y = load_dataset(image_folder)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler, svm_model = train_model(X_train, y_train)
    y_pred = svm_model.predict(scaler.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.write("Classification Report:\n", classification_report(y_test, y_pred))
    new_image_path = st.text_input("Enter the path to the new image")
    if new_image_path:
        result = predict_image(new_image_path, scaler, svm_model)
        st.write(f"The given plant is: {result}")

if __name__ == "__main__":
    main()

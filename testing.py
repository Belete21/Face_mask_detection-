import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model(r'C:\face_mask\mask_detection_model.keras')

# Define the class labels manually
class_labels = ['WithMask', 'WithoutMask']  # Adjust this according to your training classes


# Function to classify an image
def classify_image(image):
    # Resize the image to the target size (224x224)
    image_resized = cv2.resize(image, (224, 224))

    # Preprocess the image: Convert to array and scale
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Get the class name from the predicted class index
    class_name = class_labels[predicted_class[0]]

    return class_name, predictions


# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Classify the captured frame
    class_name, probabilities = classify_image(frame)

    # Display the resulting frame with prediction
    cv2.putText(frame, f'Predicted: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Face Mask Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
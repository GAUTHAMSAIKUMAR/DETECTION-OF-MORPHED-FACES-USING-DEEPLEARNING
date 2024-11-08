import cv2
import numpy as np
from tensorflow.keras.models import load_model  # Use tf.keras

# Load the trained model
model = load_model('G:/Main/Main/Pretrainedmodel.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))  # Resize to match model input size
    image = image.astype('float32') / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the image
    preprocessed_image = preprocess_image(frame)

    # Make predictions
    prediction = model.predict(preprocessed_image)

    # Convert prediction to binary result
    # Assuming binary classification with a single output neuron
    label = "Morphed" if prediction[0][0] > 0.5 else "Not Morphed"  # Use prediction[0][0]

    # Display the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the resulting frame
    cv2.imshow("Live Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
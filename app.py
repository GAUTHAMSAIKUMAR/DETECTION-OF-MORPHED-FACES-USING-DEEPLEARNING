import streamlit as st
import numpy as np
from PIL import Image
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import img_to_array
from PIL import Image, ImageChops, ImageEnhance

image_size = (128, 128)


# Load the pretrained model
model = load_model('G:\Main\Main\Pretrainedmodel.h5')

# Define class names
class_names = ['Fake', 'Real']

# Define a function to make predictions
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

def predict(image):
    # Preprocess the image
    image_array = prepare_image(image)
    image_array = image_array.reshape(-1, 128, 128, 3)
    
    # Make prediction
    y_pred = model.predict(image_array)
    y_pred_class = np.argmax(y_pred, axis=1)[0]
    confidence = np.amax(y_pred) * 100
    
    return class_names[y_pred_class], confidence

# Define the Streamlit app
def main():
    st.title("Morphed Image Detection App")
    st.write("Upload an image to detect whether it's morphed or real.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        button_clicked = st.button("Detect")
        # Make prediction
        if button_clicked:
            result, confidence = predict(uploaded_file)
            
            # Display the result
            st.write(f"Prediction: {result}")
            st.write(f"Confidence: {confidence:.2f}%")    
         

# Run the app
if __name__ == "__main__":
    main()

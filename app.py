# Importing necessary libraries
import streamlit as st  # For the web app interface
from PIL import Image  # To handle and process images
import tensorflow as tf  # To load the trained model and make predictions
import numpy as np  # For numerical computations
from keras.utils import img_to_array  # To preprocess image data
from keras.layers import DepthwiseConv2D  # For custom depthwise convolution
import distutils.core
import os  # To handle file operations

# Step 1: Define a Custom DepthwiseConv2D Class
# This ensures compatibility with models that have custom layers or arguments
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')  # Remove unsupported arguments
        super().__init__(**kwargs)

# Step 2: Load the Pre-trained Model
def load_trained_model(model_path):
    """
    Load the pre-trained Keras model with support for custom objects.
    """
    custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

# Step 3: Predict the Condition of Bread
def predict_bread_condition(model, image_path):
    """
    Predict the condition of the bread from the uploaded image.
    """
    # Load and preprocess the image
    img = Image.open(image_path).resize((224, 224))  # Resize to model's input size
    img_array = img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get the class with the highest probability

    # Define labels for prediction
    labels = {
        0: "It's a fresh bread, use it before a week.",
        1: "It's an expired bread, throw it!",
        2: "It has mold on the right side, use the left side.",
        3: "It has mold on the left side, use the right side.",
        4: "It has mold on the top side, use the bottom side.",
        5: "It has mold on the bottom side, use the top side."
    }

    return labels[predicted_class]

# Step 4: Main Streamlit Application
def main():
    """
    Streamlit app to classify bread conditions.
    """
    st.title("Bread Condition Classifier")
    st.write("Upload an image of bread to classify its condition.")

    # File uploader for image upload
    uploaded_file = st.file_uploader("Upload an image of bread", type=['jpg', 'jpeg', 'png'])

    # Load the trained model
    model_path = 'keras_model.h5'  
    model = load_trained_model(model_path)

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Save the uploaded image temporarily
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        # Predict bread condition
        prediction = predict_bread_condition(model, temp_image_path)
        st.write("### Prediction:", prediction)

        # Remove the temporary image after processing
        os.remove(temp_image_path)
                


# Step 5: Run the Application
if __name__ == '__main__':
    main()
        
    # Footer
    st.write("---")
    st.markdown('<center><a href="https://www.instagram.com/suraj_nate/" target="_blank" style="color:white;text-decoration:none">&copy; 2025 @suraj_nate All rights reserved.</a></center>', unsafe_allow_html=True)


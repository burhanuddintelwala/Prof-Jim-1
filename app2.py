import streamlit as st
import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.densenet import DenseNet201
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences                          
from joblib import load

# Load the tokenizer
tokenizer = load(r"tokenizer.pkl")

# Now you can use the tokenizer as needed

# Load the caption model
caption_model = load_model(r"caption_model.keras")                      


# Load the tokenizer
tokenizer = load(r"tokenizer.pkl")

# Now you can use the tokenizer as needed

# # Load the tokenizer
# with open(r"tokenizer.pkl", 'rb') as f:
#     tokenizer = pickle.load(f)

max_length = 34  # Define your max_length here

# Function to extract image features using DenseNet
def extract_image_features(image_path, model):
    img = load_img(image_path, target_size=(224, 224))  # Resize image to match model input
    img = img_to_array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Expand dimensions to fit model input
    feature = model.predict(img, verbose=0)  # Get features from the model
    return feature

# Function to convert integer back to word using the tokenizer
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to predict caption for a given image
def predict_caption(image_path):
    # Load the DenseNet model for feature extraction
    densenet_model = DenseNet201()
    feature_extraction_model = Model(inputs=densenet_model.input, outputs=densenet_model.layers[-2].output)
    
    # Extract features from the image
    feature = extract_image_features(image_path, feature_extraction_model)

    # Initialize the caption generation
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)  # Note: changed to 'maxlen'

        # Predict the next word
        y_pred = caption_model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)

        # Convert the predicted integer to a word
        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break
            
        in_text += " " + word
        
        if word == 'endseq':
            break
            
    return in_text.replace("startseq", "").replace("endseq", "").strip()  # Clean the output caption

# Streamlit app layout
st.title("Image Captioning App")
st.write("Upload an image to generate a caption.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Save the uploaded image to a temporary location
    temp_image_path = os.path.join("temp_dir", uploaded_file.name)
    os.makedirs("temp_dir", exist_ok=True)
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Generate caption
    if st.button("Generate Caption"):
        caption = predict_caption(temp_image_path)
        st.write("Predicted Caption:", caption)

import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, AutoConfig
from PIL import Image
from torchvision import transforms
import streamlit as st


# Load the model and feature extractor
model_name = "chriamue/bird-species-classifier"
model = AutoModelForImageClassification.from_pretrained(model_name)
# extractor = AutoFeatureExtractor.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# Define the image transformations (preprocessing)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the expected size
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet means
])

st.title("Bird Species Classifier")
uploaded_file = st.file_uploader("Choose a bird image...", type="png")


if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image = image.convert('RGB')  # Ensure image has 3 channels

    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image = preprocess(image).unsqueeze(0)


# Inference
    with torch.no_grad():
        outputs = model(image)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    # Get the label
    labels = config.id2label
    predicted_class_label = labels[predicted_class_idx]

    st.write(f"Predicted Bird Species: **{predicted_class_label}**")


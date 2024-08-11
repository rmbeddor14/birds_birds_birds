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
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image = image.convert('RGB')  # Ensure image has 3 channels
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



### the following was from my init test before I added the streamlit gui
# Load and preprocess the image
# image = Image.open("bird_pics/andrea_pic_1.png")  # Replace with the path to your image
## Ensure the image is RGB (3 channels)
## this is because without this I got a mismatch tensor size error
## the error you'll get is 
## RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0
## but andrea has a fancy camera so my sample is a 4 channel RGBA (has an alpha channel)
# if image.mode != 'RGB':
#     image = image.convert('RGB')


# image = preprocess(image).unsqueeze(0)  # Add batch dimension


# # Perform inference
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     outputs = model(image)
#     logits = outputs.logits


# # Get the predicted class
# predicted_class_idx = logits.argmax(-1).item()

# # Map the predicted class index to the class label
# labels = config.id2label
# predicted_class_label = labels[predicted_class_idx]

# print(f"Predicted class index: {predicted_class_idx}")
# print(f"Predicted class label: {predicted_class_label}")

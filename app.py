# Import all of the dependencies
import streamlit as st
import os
import cv2
import numpy as np
import imageio
import imageio_ffmpeg as ffmpeg
import torch
from torchvision import transforms
import mediapipe as mp
from src.models import LipNet
import src.utils
import src.constants as constants

# Initialize Mediapipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Load the LipNet model
@st.cache_resource
def load_model(checkpoint_path):
    model = LipNet(vocab_size=len(constants.vocab), hidden_size=256, input_channels=3)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

# Process video frames to extract lip regions
def process_frames(frames, face_mesh, padding, transform=transforms.ToTensor()):
    processed_frames = []
    for frame in frames:
        lip_region = src.utils.extract_lip_region_from_image(frame, face_mesh, padding)
        if lip_region is not None:
            processed_frames.append(transform(lip_region))
    if not processed_frames:
        return None, None  # No valid frames detected
    frames_tensor = torch.stack(processed_frames)
    frames_tensor = (frames_tensor - frames_tensor.mean()) / frames_tensor.std()
    return frames_tensor, frames_tensor.permute(0, 2, 3, 1).detach().numpy().astype(np.uint8)


# Create a GIF from frames
def create_gif(frames, output_path):
    imageio.mimsave(output_path, frames, fps=10)
    

# Predict from video file
def predict_from_video(file_path, model, device, idx2word):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return "No frames extracted from video.", None
    frames_tensor, processed_frames = process_frames(frames, face_mesh, padding=30)
    if frames_tensor is None:
        return "No lip region detected.", None
    frames_tensor = frames_tensor.unsqueeze(0).to(device)
    output = model(frames_tensor.permute(0, 2, 1, 3, 4))
    prediction = src.utils.ctc_greedy_decode(output, idx2word)
    return prediction, processed_frames

# Streamlit layout
st.set_page_config(layout='wide')

# Sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is inspired by the LipNet deep learning model.')

# Title
st.title('LipNet Full Stack App')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and vocabulary
model = load_model("checkpoints/checkpoint_best.pt").to(device)
_, idx2word = src.utils.get_word2idx_idx2word(constants.vocab)

# Video selection
options = os.listdir(os.path.join( 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

if options:
    col1, col2 = st.columns(2)

    # Column 1: Processed frames as GIF
    with col1:
        st.info('Processed frames as seen by the model')
        file_path = os.path.join('data', 's1', selected_video)
        prediction, processed_frames = predict_from_video(file_path, model, device, idx2word)
        if processed_frames.any():
            processed_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in processed_frames]
            gif_path = 'processed_frames.gif'
            create_gif(processed_frames, gif_path)
            st.image(gif_path, width=500)

    # Column 2: Predictions
    with col2:
        st.info('Predicted tokens from the model')
        st.text(prediction[0])

        # Decode prediction into text
        st.info('Decoded prediction')
        decoded_text = "".join(src.utils.num_to_char(prediction[0], idx2word))
        st.text(decoded_text)
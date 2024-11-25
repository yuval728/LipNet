import torch
from torchvision import transforms
import mediapipe as mp
from ts.torch_handler.base_handler import BaseHandler
import numpy as np
import cv2

from constants import vocab
from utils import (
    ctc_greedy_decode,
    num_to_char,
    extract_lip_region_from_image,
    get_word2idx_idx2word,
    character_error_rate,
    word_error_rate,
)


class VerbalVisionHandler(BaseHandler):
    def initialize(self, context):
        properties = context.system_properties
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.jit.load(f"{properties.get('model_dir')}/model.pt", map_location=self._device)
        self._model.eval()
        self._transform = transforms.Compose(
            [transforms.Resize((50, 100)), transforms.ToTensor()]
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        )
        _, self._idx2word = get_word2idx_idx2word(vocab=vocab)

    def preprocess(self, data):
        # print(data)
        video_data = data[0].get("video")
        if video_data is None:
            raise ValueError("Invalid input. 'video' key not found")
        
        video_np = np.frombuffer(video_data, dtype=np.uint8)
        print(video_np)
        frame = cv2.imdecode(video_np, cv2.IMREAD_UNCHANGED)
        
        print(frame)
        print(f"Frame dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}, shape: {frame.shape}")
        if frame is None:
            raise ValueError("Failed to decode video data")
        
        # Extract lip region from video
        # frames = []
        # while video_stream.isOpened():
        #     ret, frame = video_stream.read()
        #     if not ret:
        #         break
        #     print(f"Frame dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}, shape: {frame.shape}")
        #     if frame.dtype != np.uint8:
        #         frame = (frame * 255).astype(np.uint8) 
        #     lip_region = extract_lip_region_from_image(frame, self._face_mesh, padding=30)
        #     if lip_region is not None:
        #         frames.append(self._transform(lip_region))

        frames = torch.stack(frame)
        mean = frames.mean()
        std = frames.std()
        frames = (frames - mean) / std
        frames = frames.unsqueeze(0)

        return frames.permute(0, 2, 1, 3, 4)

    def postprocess(self, data, ref_text):
        prediction = ctc_greedy_decode(data, self._idx2word, print_output=False)
        sentence = num_to_char(prediction, self._idx2word)

        if ref_text:
            cer = character_error_rate(ref_text, sentence)
            wer = word_error_rate(ref_text, sentence)
            sentence = f"{sentence}\nCharacter Error Rate: {cer:.2f}\nWord Error Rate: {wer:.2f}"
        
        return [sentence]

    def handle(self, data, context):
        data = self.preprocess(data)

        ref_text = None
        if "ref_text" in data[0]:
            ref_text = data[0]["ref_text"]

        output = self._model(data)
        return self.postprocess(output, ref_text)

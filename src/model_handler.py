import torch
from torchvision import transforms
import mediapipe as mp
from ts.torch_handler.base_handler import BaseHandler
import numpy as np
import cv2
import av
import io

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
            [transforms.ToTensor(), transforms.Resize((50, 100))]
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
        
        container = av.open(io.BytesIO(video_data))
    
        frames = []
        frame_count = 0
        
        for frame in container.decode(video=0):
            img = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
            lip_region = extract_lip_region_from_image(img, self._face_mesh, padding=30)
            if lip_region is not None:
                frames.append(self._transform(lip_region))
                frame_count += 1
                
        print(f"Processed {frame_count} frames")
        print(len(frames))
        print(f"Frames shape: {frames[0].shape}")
        

        frames = torch.stack(frames)
        mean = frames.mean()
        std = frames.std()
        frames = (frames - mean) / std
        frames = frames.unsqueeze(0)

        return frames.permute(0, 2, 1, 3, 4)

    def postprocess(self, data, ref_text):
        prediction = ctc_greedy_decode(data, self._idx2word, print_output=False)
        sentence = num_to_char(prediction[0], self._idx2word)

        response = {"Sentence": "".join(sentence)}
        print(ref_text)
        if ref_text:
            cer = character_error_rate(ref_text, sentence)
            wer = word_error_rate(ref_text, sentence)
            response["Character Error Rate"] = cer
            response["Word Error Rate"] = wer
        
        return [response]

    def handle(self, data, context):
        frames = self.preprocess(data)

        ref_text = None
        if "ref_text" in data[0]:
            ref_text = data[0]["ref_text"].decode("utf-8")

        output = self._model(frames)
        return self.postprocess(output, ref_text)

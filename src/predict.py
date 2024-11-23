import torch
from torchvision import transforms
import numpy
import mediapipe as mp
import cv2
import constants
import utils
import argparse
import imageio

imageio.warnings.simplefilter("ignore")

word2idx, idx2word = utils.get_word2idx_idx2word(constants.vocab)


def normalize_frames(frames_tensor):
    return (frames_tensor - frames_tensor.mean()) / frames_tensor.std()


def predict_from_video(
    file_path, model, face_mesh, padding, device, transform=transforms.ToTensor()
):
        frames = utils.extract_lip_region_from_video(
            file_path, face_mesh, padding=padding
        )

        if not frames:
            raise Exception("No frames extracted from video.")

        transformed_frames = [transform(frame) for frame in frames]

        frames_tensor = torch.stack(transformed_frames)
        frames_tensor = normalize_frames(frames_tensor)
        frames_tensor = frames_tensor.unsqueeze(0).to(device)

        output = model(frames_tensor.permute(0, 2, 1, 3, 4))
        prediction = utils.ctc_greedy_decode(output, idx2word, print_output=False)

        return prediction,  frames_tensor.squeeze(0).permute(0, 2, 3, 1).detach().numpy().astype(numpy.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the lip reading from a video")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model file"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the video file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--padding", type=int, default=30, help="Padding around the lip region"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.gif",
        help="Path to save the output GIF",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = torch.jit.load(args.model_path)

    model.eval()
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    )
    prediction, frames = predict_from_video(
        args.video_path, model, face_mesh, args.padding, args.device
    )
    print(prediction)
    for i in prediction:
        text = "".join(utils.num_to_char(i, idx2word))
        print(text)
    if frames is not None:
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
        imageio.mimsave(args.output_path, frames, fps=10)
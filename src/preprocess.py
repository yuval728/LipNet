import os
import numpy as np
import mediapipe as mp
from . import utils
import argparse
from tqdm.auto import tqdm 

def save_lip_region_from_video_to_npy(videos_path, face_mesh, padding=20, output_path=None):
    for video in tqdm(os.listdir(videos_path)):
        video_path = os.path.join(videos_path, video)
        save_path = os.path.join(output_path, video)
        frames = utils.extract_lip_region_from_video(video_path, face_mesh, padding)
        if save_path is None:
            save_path = os.path.splitext(video_path)[0] + '.npy'
            
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    np.save(save_path, frames)
    
    

def main():

    args = argparse.ArgumentParser()
    args.add_argument('--videos_path', type=str, default='data/s1', help='Path to the videos')
    args.add_argument('--save_path', type=str, default='data/lip_region', help='Path to save the lip region')
    args.add_argument('--padding', type=int, default=30, help='Padding around the lip region')
    args = args.parse_args()
    
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    
    save_lip_region_from_video_to_npy(args.videos_path, face_mesh, output_path=args.save_path, padding=args.padding)
        
    print(f'Lip region saved to {args.save_path}')
    face_mesh.close()
    
if __name__ == '__main__':
    main()
    
    

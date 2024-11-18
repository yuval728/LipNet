### Data Preprocessing:
    python src/preprocess.py --videos_path data/s1 --save_path data/lip_region/s1 --padding 30


### Training:
    python src/train.py --data_dir data/s1 --label_dir data/alignments/s1 --checkpoint_dir checkpoints --batch_size 2 --num_epochs 102 --prev_checkpoint checkpoints/check.pt

### Testing:
    python src/eval.py --checkpoint checkpoints/check_best3.pt --data_dir data/s1 --label_dir data/alignments/s1 

### Data Preprocessing:
    python src/preprocess.py --videos_path data/s1 --save_path data/lip_region/s1 --padding 30


### Training:
    python src/train.py --data_dir data/lip_region/s1 --label_dir data/alignments/s1 --checkpoint_path checkpoints --batch_size 2 --num_epochs 2 --prev_checkpoint checkpoints/check1.pt --hidden_size 256 --new_lr 2e-5

### Testing:
    python src/eval.py --checkpoint checkpoints/check_best1.pt --data_dir data/lip_region/s1 --label_dir data/alignments/s1  --run_id 0ca421225389489ea6de0dd41050d02b

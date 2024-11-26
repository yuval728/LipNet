### Data Preprocessing:
    python src/preprocess.py --videos_path data/s1 --save_path data/lip_region/s1 --padding 30


### Training:
    python src/train.py --data_dir data/lip_region/s1 --label_dir data/alignments/s1 --checkpoint_path checkpoints --batch_size 2 --num_epochs 2 --prev_checkpoint checkpoints/check1.pt --hidden_size 256 --new_lr 2e-5

### Testing:
    python src/eval.py --checkpoint checkpoints/check_best1.pt --data_dir data/lip_region/s1 --label_dir data/alignments/s1  --run_id 0ca421225389489ea6de0dd41050d02b

### Trace model:
    python src/trace_model.py --checkpoint_path checkpoints\checkpoint_best.pt --output_path model_store/model.pt

### Prediction:
    python src/prediction.py --model_path model_store/model.pt --video_path data/s1/bbaf2n.mpg --padding 30

### Archive model:
    torch-model-archiver --model-name lipnet  --version 1.0  --serialized-file model_store/model.pt  --handler src/model_handler.py  --extra-files "src/models.py,src/constants.py,src/utils.py" --export-path model_store -f

### Build docker image:
    docker build -t verbalvision .

### Run docker container:
    docker run -p 8080:8080 8081:8081 8082:8082 verbalvision

### API:
    curl http://localhost:8080/ping
    curl http://localhost:8081/models
    curl http://localhost:8081/models/lipnet
    curl -X POST http://localhost:8080/predictions/lipnet \
     -F "video=@data/s1/bbaf2n.mpg" \
     -F "ref_text=bin blue at two now" 
    
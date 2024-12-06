# VerbalVision

VerbalVision is a deep learning-based lip reading application inspired by the LipNet model. It processes video frames to extract lip regions and predicts the spoken words.

## Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Docker
- DVC
- MLflow

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yuval728/verbalvision.git
    cd verbalvision
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up DVC:
    ```sh
    dvc pull
    ```

## Explanation and Approach

### Overview

VerbalVision leverages a deep learning model to perform lip reading from video inputs. The approach is inspired by the LipNet model, which uses a combination of convolutional neural networks (CNNs) and recurrent neural networks (RNNs) to process sequences of video frames and predict the corresponding text.

### Data Preparation

The preprocessing step involves extracting the lip region from each frame of the video. This is done using the MediaPipe Face Mesh solution, which detects facial landmarks and isolates the lip region. The extracted lip regions are then saved as numpy arrays for further processing.

### Model Architecture

The core of VerbalVision is the LipNet model, which consists of the following components:

1. **Convolutional Layers**: These layers extract spatial features from the lip region in each frame.
2. **Recurrent Layers**: Bidirectional GRU layers capture temporal dependencies across frames.
3. **Fully Connected Layer**: This layer maps the output of the recurrent layers to the vocabulary space.

The model is trained using the Connectionist Temporal Classification (CTC) loss function, which is suitable for sequence-to-sequence tasks where the alignment between input and output sequences is unknown.

### Training

The training process involves feeding batches of preprocessed video frames and their corresponding labels to the model. The model's parameters are optimized using the Adam optimizer, and the training progress is monitored using validation loss.

### Evaluation

The trained model is evaluated on a separate validation set to measure its performance. Metrics such as Word Error Rate (WER) and Character Error Rate (CER) are used to quantify the model's accuracy.

### Prediction

For prediction, the model takes a video file as input, preprocesses the frames to extract lip regions, and then feeds the frames through the trained model to generate the predicted text.

## Pipeline

For detailed information about the data processing and model training pipeline, please refer to the [pipe.md](pipe.md) file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Doodle Classifier

A neural network-based doodle classifier that recognizes hand-drawn sketches from a set of categories. The project includes a PyGame-based drawing interface, a custom neural network implementation with data augmentation approach.

![Demo][media/demo.gif]
## Features

- Draw doodles on a 28x28 grid and get real-time predictions.
- Neural network with 5 dense layers, ReLU and Softmax activations.
- Data augmentation with affine transforms and noise.
- Model training and saving/loading of parameters.
- Supports 8 categories (see `categories.py`).

## Project Structure

- [`game.py`](game.py): PyGame GUI for drawing and classifying doodles.
- [`nerual.py`](nerual.py): Neural network implementation and training logic.
- [`processing.py`](processing.py): Data augmentation utilities.
- [`train.py`](train.py): Script to train the model.
- [`categories.py`](categories.py): Category label utilities.
- [`parameters/`](parameters/): Saved model weights and biases.
- [`X.npy`](X.npy), [`y.npy`](y.npy): Training data (images and labels).
- [`quickdraw_data/`](quickdraw_data/): Raw NDJSON data for each category.

## Getting Started

### Requirements

- Python 3.8+
- `numpy`
- `pandas`
- `pygame`
- `scipy`
- `scikit-learn`
- `matplotlib`

Install dependencies:

```sh
pip install numpy pandas pygame scipy scikit-learn matplotlib
```

### Training the Model

1. Specify the categories in `categories.py` if needed.
2. Run ```python load.py``` to download the data and prepare for the training.
2. Run the training script:

```
python train.py
```

Model parameters will be saved in the `parameters/` directory.

### Running the Classifier GUI

After training, launch the drawing interface:

```sh
python game.py
```

- Draw in the left grid.
- The sidebar shows the top predictions.
- Click "Clear" to reset the canvas.

## Data Augmentation

See [`processing.random_augment_image`](processing.py) for details on how images are randomly transformed during training.

The augmentation includes:
- Random scaling (zooming in/out)
- Random rotation
- Random translation (shifting)
- Adding Gaussian noise

## Model Architecture

- Dense(784, 300) → ReLU
- Dense(300, 60) → ReLU
- Dense(60, 50) → ReLU
- Dense(50, 40) → ReLU
- Dense(40, 8) → Softmax

This network uses momentum to accelerate learning and the training is done by mini-batches with the default size of 64. It trains up to 90% accuracy of test images within 200 epochs.

See [`Model`](nerual.py) for implementation details.

# MNIST Image Reconstruction Using a Custom MLP Autoencoder

A from-scratch implementation of an **Artificial Neural Network (ANN) based Autoencoder** for image reconstruction on the **MNIST handwritten digit dataset**.

This project implements a multi-layer perceptron (MLP) autoencoder using only **NumPy-based matrix operations**, without relying on deep learning frameworks for model training. The network learns a compressed representation of handwritten digits and reconstructs the original images from this latent representation.

The project was developed as part of a machine learning coursework project and demonstrates the implementation of neural network fundamentals, including forward propagation, backpropagation, weight initialization, mini-batch training, momentum-based optimization, and validation.

---

## Project Overview

Autoencoders are neural networks designed to learn efficient representations of input data by compressing the input into a lower-dimensional feature space and reconstructing it.

In this project:

* The input images are MNIST grayscale digits.
* Each image is flattened into a vector of **784 pixels**.
* A custom MLP encoder-decoder architecture learns the mapping:

```
784 → 256 → 128 → 128 → 256 → 784
```

* The network is trained to minimize the reconstruction error between the original and reconstructed images.

The objective is to verify that a manually implemented neural network can learn meaningful image representations and perform reconstruction without using pre-built neural network libraries.

---

## Dataset

The project uses the **MNIST handwritten digit dataset**, which contains:

* 60,000 training images
* 10,000 testing images
* Image resolution: 28 × 28 pixels
* Grayscale intensity values: 0–255

The dataset is automatically downloaded using TensorFlow's built-in MNIST loader:

```python
from tensorflow.keras.datasets import mnist
```

Only a subset of the dataset is used during training:

* 20,000 images selected from the training set
* Split into:

  * 60% training data
  * 20% validation data
  * 20% testing data

Images are normalized:

```
pixel_value / 255.0
```

and reshaped:

```
(28, 28) → (784,)
```

---

## Model Architecture

The autoencoder is implemented from scratch using NumPy.

### Network Structure

```
Input Layer
    |
    | 784 neurons
    |
Hidden Layer
    |
    | 256 neurons
    |
Hidden Layer
    |
    | 128 neurons
    |
Bottleneck Layer
    |
    | 128 neurons
    |
Hidden Layer
    |
    | 256 neurons
    |
Output Layer
    |
    | 784 neurons
    |
Reconstructed Image
```

### Activation Function

The network supports:

* ReLU
* Leaky ReLU
* Sigmoid

The default activation function used:

```
ReLU
```

---

## Training Details

The neural network training process is manually implemented, including:

### Weight Initialization

Weights are initialized using variance-scaled random initialization:

```
std = sqrt(1 / number_of_inputs)
```

### Optimization

The model uses:

| Parameter     | Value                    |
| ------------- | ------------------------ |
| Epochs        | 5                        |
| Learning Rate | 0.01                     |
| Momentum      | 0.9                      |
| Batch Size    | 128                      |
| Loss Function | Mean Squared Error (MSE) |

### Training Process

For every batch:

1. Forward propagation
2. Reconstruction error calculation
3. Backpropagation
4. Gradient accumulation
5. Momentum-based weight update

The best model weights are saved according to the lowest validation MSE.

---

## Results

After training, the model reconstructs handwritten digits from the compressed representations.

The notebook provides:

* Training MSE progression
* Validation MSE evaluation
* Original vs reconstructed image comparisons

Example output:

```
Original Image        Reconstructed Image

   7                       7
   3                       3
   9                       9
```

The reconstructed images demonstrate that the manually implemented autoencoder successfully learns the important visual features of handwritten digits.

---

## Project Structure

```
./
├── notebooks/
│   └── main.ipynb          # Complete implementation and experiments
│
├── requirements.txt        # Required Python packages
├── README.md               # Project documentation
├── LICENSE                 # MIT License
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook notebooks/main.ipynb
```

Alternatively, the notebook can be executed directly using Google Colab.

---

## Requirements

Main dependencies:

```
numpy==2.0.2
matplotlib==3.10.0
tensorflow==2.20.0
scikit-learn==1.6.1
tqdm==4.67.3
```

---

## Future Improvements

Possible extensions:

* Implement fully vectorized training for faster execution
* Add additional regularization techniques
* Compare against TensorFlow/PyTorch autoencoders
* Experiment with deeper architectures
* Visualize the learned latent space using dimensionality reduction methods

---

## License

This project is released under the MIT License.

See the `LICENSE` file for more information.

---


# 🧠 Sign Language Recognition

This project implements a deep learning model to recognize American Sign Language (ASL) hand gestures. The model is trained to classify alphabet signs (A–Z), numbers(0-9) from image data using a Convolutional Neural Network (CNN). It includes training, evaluation, and testing components, along with necessary preprocessing steps.

The application aims to enhance communication accessibility for the hearing- and speech-impaired by converting hand gestures into readable text or voice output using computer vision techniques.

---

## 📁 Project Structure

- `Sign_Language.ipynb`: Notebook for data preprocessing, model building, training, and saving the CNN model.
- `Test_model.ipynb`: Notebook for loading the trained model and testing it on new input images.
- `requirements.txt`: Lists all required Python packages to run the notebooks.

---

## 🚀 Features

- Classifies ASL alphabet signs (A–Z)
- CNN-based architecture with convolution, pooling, and dense layers
- Evaluation using accuracy metrics
- Easy testing with custom images
- Clean and modular notebooks

---

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/RafiAhamed07/Sign_Language_Classcification.git
cd sign-language-recognition
```

2. ### ⚙️ Environment Setup (with CUDA support)

> ⚠️ **Note:** This project is best run in a `conda` environment with proper CUDA support for GPU acceleration.

 **Create and activate a Conda environment:**
```bash
conda create -n signlang python=3.9
conda activate signlang
```

 **Install TensorFlow with GPU support (CUDA):**
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.10.1
```

Ensure that your system has a compatible NVIDIA GPU and the correct drivers installed. You can verify GPU access in Python using:

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

---

## 🧪 Usage

1. **Training the model:**
   - Open `Sign_Language.ipynb`
   - Run cells step-by-step: data loading → preprocessing → model training → save model

2. **Testing the model:**
   - Open `Test_model.ipynb`
   - Load the saved model and test it on new images

---

## 🖼️ Dataset

You will need a dataset of ASL hand gesture images. For example, the [Sign Language Gesture Images Dataset]([https://www.kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/ahmedkhanak1995/sign-language-gesture-images-dataset/data)) from Kaggle is a great option.

Make sure the dataset is structured as expected in the training notebook.

---

## 📊 Model Architecture

- **Input Layer**: Preprocessed image data
- **Convolutional Layers**: Feature extraction using filters
- **MaxPooling Layers**: Downsampling to reduce complexity
- **Fully Connected Layers**: Classification into 26 alphabet classes
- **Softmax Output**: Multi-class probability prediction

---

## ✅ Requirements

All dependencies are listed in `requirements.txt`. Key libraries include:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Gradio (optional for UI)

Install them using:

```bash
pip install -r requirements.txt
```

---

## ✨ Acknowledgments

- Dataset from Kaggle
- Built with TensorFlow and Keras
- Inspired by inclusive tech and accessibility solutions






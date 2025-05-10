# Handwritten Digit Recognition using MNIST and TensorFlow

This project demonstrates how to build a simple neural network that can recognize handwritten digits using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The model is trained using TensorFlow and Keras in a Jupyter Notebook.

## 🧠 Project Overview

The goal of this project is to train a machine learning model that can correctly identify digits (0–9) from grayscale images. The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits, each sized 28x28 pixels.

### ✨ Key Features

* Loads and normalizes MNIST data.
* Builds a simple neural network with:

  * `Flatten` layer to convert images to 1D arrays.
  * `Dense` layer with ReLU activation.
  * `Dense` output layer with softmax activation.
* Trains the model for 20 epochs.
* Evaluates the model on test data.
* Optionally tests with a custom image using OpenCV.

---

## 📁 Files Included

* `handwritten_digit_recognition.ipynb`: The Jupyter notebook with all code.
* (Optional) Sample test image like `img_7.png` for testing the model on real input.
* `README.md`: This file.

---

## ⚙️ Requirements

You need Python and the following libraries:

```bash
pip install tensorflow opencv-python matplotlib
```

---

## 🚀 How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/yourusername/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

2. Open the Jupyter Notebook:

```bash
jupyter notebook handwritten_digit_recognition.ipynb
```

3. Run all cells in order to:

   * Load the data
   * Train the model
   * Evaluate performance
   * Test with a custom image (if available)

---

## 📷 Custom Image Testing (Optional)

You can test your own handwritten digit image:

1. Create an image (like `img_7.png`) that's 28x28 pixels in grayscale.
2. Place the image in the same folder as the notebook.
3. The notebook contains code to load and predict the digit using:

```python
cv2.imread('img_7.png', cv2.IMREAD_GRAYSCALE)
```

---

## 📊 Output

After training, the model typically achieves around **98% accuracy** on the test set.

---

## 📚 Concepts Used

* Neural Networks
* Activation Functions (ReLU, Softmax)
* Image Preprocessing
* Model Evaluation

---

## 🧑‍💻 Author

Made with ❤️ by Rohan Kumar Singh

---

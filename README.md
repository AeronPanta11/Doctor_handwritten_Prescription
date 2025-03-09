<body>
  <h1>Doctor Prescription Identifier</h1>
  <p>This project implements a deep learning model to identify handwritten prescriptions using TensorFlow and Keras.</p>

  <h2>Table of Contents</h2>
  <ul>
      <li><a href="#introduction">Introduction</a></li>
      <li><a href="#dataset">Dataset</a></li>
      <li><a href="#model-architecture">Model Architecture</a></li>
      <li><a href="#training">Training</a></li>
      <li><a href="#results">Results</a></li>
      <li><a href="#conclusion">Conclusion</a></li>
      <li><a href="#installation">Installation</a></li>
      <li><a href="#usage">Usage</a></li>
      <li><a href="#license">License</a></li>
  </ul>

  <h2 id="introduction">Introduction</h2>
  <p>The Doctor Prescription Identifier is a convolutional neural network (CNN) model designed to classify images of handwritten prescriptions. The model is trained on a dataset of labeled images and can predict the class of new images.</p>

  <h2 id="dataset">Dataset</h2>
  <p>The dataset used in this project consists of:</p>
  <ul>
      <li>Training images: 3120 images</li>
      <li>Validation images: 780 images</li>
      <li>Testing images: 50 images (not labeled)</li>
  </ul>
  <p>Images are resized to 224x224 pixels and normalized to have pixel values between 0 and 1.</p>

  <h2 id="model-architecture">Model Architecture</h2>
  <p>The model architecture consists of:</p>
  <ul>
      <li>Base model: VGG16 (pre-trained on ImageNet)</li>
      <li>Flatten layer to convert the 2D matrix to a 1D vector</li>
      <li>Dense layers for classification</li>
      <li>Dropout layers to prevent overfitting</li>
      <li>Output layer with softmax activation function for multi-class classification</li>
  </ul>

  <h2 id="training">Training</h2>
  <p>The model is trained using:</p>
  <ul>
      <li>Optimizer: Adam</li>
      <li>Loss function: Categorical Crossentropy</li>
      <li>Metrics: Accuracy</li>
      <li>Batch size: 32</li>
      <li>Epochs: 5</li>
  </ul>
  <p>Data augmentation techniques such as rotation, zooming, and shifting are applied to the training images to improve model generalization.</p>

  <h2 id="results">Results</h2>
  <p>The model achieved an accuracy of 100% on the training set, but validation accuracy was not reported correctly. The model needs further tuning and validation to ensure it generalizes well to unseen data.</p>

  <h2 id="conclusion">Conclusion</h2>
  <p>The Doctor Prescription Identifier demonstrates the effectiveness of convolutional neural networks in image classification tasks. Future work may include using transfer learning with pre-trained models to enhance accuracy and improve validation performance.</p>



  <h2 id="usage">Usage</h2>
  <p>To use the model, load the trained weights and pass an image through the model to get predictions. The model can classify images of handwritten prescriptions into different classes.</p>

  <h3>Example Code</h3>
  <pre><code>
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model
model = tf.keras.models.load_model('path_to_your_model.h5')

# Load and preprocess the image
img_path = 'path_to_your_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)
print(f"Predicted Class: {predicted_class}")
  </code></pre>

  <h2 id="license">License</h2>
  <p>This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details.</p>
</body>

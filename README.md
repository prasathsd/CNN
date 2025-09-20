Cat vs. Dog Image Classification Project
Overview
This project explores and compares three different machine learning models for the binary classification of cat and dog images. My goal was to understand the strengths and weaknesses of different approaches, from a simple Convolutional Neural Network (CNN) to more advanced hybrid models, and to achieve the highest possible accuracy.

Model 1: CNN Trained from Scratch
This model was a foundational deep learning approach where I built and trained a CNN from scratch to learn features directly from the images.

Key Steps:

Data Preprocessing: I cleaned the dataset to remove corrupted images, resized all images to 128x128 pixels, and normalized the pixel values.

Data Augmentation: To improve the model's ability to generalize, I used data augmentation techniques like random horizontal flips, rotations, and zooms.

Architecture: The model consisted of three convolutional blocks (Conv2D, BatchNormalization, MaxPooling2D), followed by a Flatten layer, a Dense layer with Dropout, and a final output layer for classification.

Training: I trained the model for 15 epochs using the Adam optimizer and binary_crossentropy loss.

Result:

Final Validation Accuracy: 84.68%

This model successfully demonstrated the basic principles of CNNs for image classification.

Model 2: Hybrid VGG16 + XGBoost Classifier
This approach was inspired by the innovative research in neuroimaging, which often uses hybrid models. I wanted to apply a similar concept to this simpler problem to better understand how it works. The idea was to leverage the feature-learning power of a pre-trained CNN and combine it with a robust traditional classifier.

Key Steps:

Feature Extraction: I used the pre-trained VGG16 model (without its top classification layers) as a feature extractor. This model has already learned a rich set of features from the large ImageNet dataset.

Feature Vector Creation: I processed 1000 cat and 1000 dog images through the VGG16 model, taking the output of the last convolutional layer and flattening it into a single feature vector for each image.

Classification: These feature vectors were then used to train an XGBoost Classifier, an ensemble model known for its high performance on structured data.

Result:

Final Test Accuracy: 94.75%

This hybrid model was the most successful, proving that combining a pre-trained feature extractor with a powerful classifier can lead to significant improvements in accuracy.

Model 3: Hybrid VGG16 + PCA + XGBoost Classifier
As a final experiment, I explored the effect of dimensionality reduction on the hybrid model's performance.

Key Steps:

Feature Extraction: I again used VGG16 to extract features.

Dimensionality Reduction: I applied Principal Component Analysis (PCA) to the feature vectors, reducing their dimensionality to 256 components. The goal was to test if a smaller feature set could maintain high accuracy.

Classification: I trained an XGBoost Classifier on these PCA-transformed features.

Result:

Final Test Accuracy: 84.34%

The lower accuracy in this experiment suggests that while PCA can reduce model complexity, it can also remove crucial information needed for optimal classification performance.

<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/3cd5b206-a20b-4c13-a38b-5d657d90ef4c" />


Conclusion
This project provided a comprehensive exploration of image classification techniques. The most successful model was the hybrid VGG16 + XGBoost classifier, which achieved an accuracy of 94.75%. This project was a valuable exercise in applying advanced machine learning concepts, inspired by neuroimaging research, to a practical computer vision problem.

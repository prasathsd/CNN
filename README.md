# CNN
Summary :
This document outlines the entire process I followed to build and evaluate several machine learning models for classifying images of cats and dogs. My project progressed from training a basic Convolutional Neural Network (CNN) from scratch to using more advanced hybrid models, with the goal of achieving the highest possible accuracy.

# Model 1: A Convolutional Neural Network (CNN) from Scratch
My first approach was to build a standard CNN model from scratch. The main goal here was to create a network that could learn to extract features from the raw image data on its own.
My Process:
Data Preparation: The first and most crucial step was to clean the dataset. I used a function called strict_clean_folder to remove any corrupted or unreadable images, which ensured my training process would run smoothly.
Data Preprocessing: Next, I loaded the images and resized all of them to a uniform size of 128×128 pixels. I also normalized the pixel values to a range between 0 and 1, a standard practice for training neural networks.
Data Augmentation: To prevent my model from overfitting and to help it generalize better to new images, I applied data augmentation. I used random horizontal flips, rotations, and zooms on the training images.
Model Architecture: I designed a sequential Keras model with three main convolutional blocks. Each block consisted of a Conv2D layer, a BatchNormalization layer, and a MaxPooling2D layer. This structure helps the model learn increasingly complex features. I then flattened the output and added a Dense hidden layer with a Dropout layer (with a 0.5 rate) for regularization, and a final Dense layer with a sigmoid activation for binary classification.
Training: I compiled the model using the Adam optimizer and binary_crossentropy loss. I then trained the model for 15 epochs, monitoring its performance on a separate validation set.
Result:
Final Validation Accuracy: 84.68%
This model provided a good starting point and proved that a CNN could learn to distinguish between the two classes with reasonable accuracy.


# Model 2: My Hybrid VGG16 + XGBoost Classifier
Inspired by the research in neuroimaging, I wanted to explore a hybrid approach to make my project more relevant to the field. I was fascinated by the work on neuroimaging and decided to apply a similar methodology to a simpler problem like cat vs. dog classification to better understand the process. I believe the patch-based feature extraction and the use of an ensemble classifier is a powerful technique with wide-ranging applications.
My Process:
Feature Extraction: I used the pre-trained VGG16 model, which is already an expert at recognizing general image features because it was trained on the massive ImageNet dataset. I removed its final classification layers, turning it into a powerful feature extractor.
Feature Vector Creation: For a subset of 1000 cat and 1000 dog images, I passed each image through the VGG16 model. The output of the last convolutional block, a high-dimensional feature map, was then flattened into a single feature vector. This method of breaking down images into patches and extracting features is a simplified version of techniques used in neuroimaging analysis.
Classification: I then used these extracted feature vectors as input to an XGBoost Classifier. XGBoost is an ensemble machine learning model that is highly effective on structured, tabular data like my feature vectors. I configured the model with 100 estimators and a maximum depth of 5.
Result:
Final Test Accuracy: 94.75%
This hybrid model was my most successful approach. It demonstrated the power of transfer learning, as the VGG16 model provided a rich set of features that the XGBoost classifier could use to make highly accurate predictions.

# Model 3: My Hybrid VGG16 + PCA + XGBoost Classifier
As a final experiment, I wanted to see if I could improve the performance or efficiency of the hybrid model by reducing the dimensionality of the features.
My Process:
Feature Extraction: I used the same VGG16 model to extract the initial feature vectors.
Dimensionality Reduction: I applied Principal Component Analysis (PCA) to the feature vectors, reducing the number of features to 256 components. My goal was to make the model lighter and potentially faster to train.
Classification: I trained an XGBoost Classifier on these new, reduced-dimension features.
Result:
Final Test Accuracy: 84.34%
The accuracy was lower with PCA than without it. This suggests that the dimensionality reduction, while simplifying the data, removed some of the key feature information that the XGBoost classifier needed to make accurate predictions.



# Conclusion

This project provided a comprehensive exploration of various machine learning techniques for image classification. While the custom Convolutional Neural Network (CNN) model demonstrated satisfactory performance, the hybrid methodology employing VGG16 for feature extraction and XGBoost for classification proved to be the most efficacious, achieving the highest accuracy. This endeavor underscored the significant advantages of transfer learning and the synergistic potential of combining distinct models to attain superior outcomes. Furthermore, the experimentation with Principal Component Analysis (PCA) highlighted that while dimensionality reduction can offer benefits, it may, at times, inadvertently compromise valuable feature information crucial for accurate predictions.

<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/9253cfff-c32d-46b1-877f-ed549d447873" />




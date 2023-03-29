# Lemon Classification Model

This repository is developed for the classification of lemons' quality. There 2 classes to represent classes: 
  - Good quality
  - Bad quality
  
Dataset called [Lemon Quality Dataset](https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset) is retrieved from Kaggle. Some sample input images are reprensented below:

***Bad Quality Image Samples:***

<a href="url">
  <img src="https://github.com/ekaraali/Lemon_Classification_Model/blob/main/images/bad_quality_14.jpg?raw=true" height="250" width="250">
  <img src="https://github.com/ekaraali/Lemon_Classification_Model/blob/main/images/bad_quality_5.jpg?raw=true" height="250" width="250">
  <img src="https://github.com/ekaraali/Lemon_Classification_Model/blob/main/images/bad_quality_6.jpg?raw=true" height="250" width="250">
</a>
<br/>
<br/>

***Good Quality Image Samples:***

<a href="url">
  <img src="https://github.com/ekaraali/Lemon_Classification_Model/blob/main/images/good_quality_3.jpg?raw=true" height="250" width="250">
  <img src="https://github.com/ekaraali/Lemon_Classification_Model/blob/main/images/good_quality_6.jpg?raw=true" height="250" width="250">
  <img src="https://github.com/ekaraali/Lemon_Classification_Model/blob/main/images/good_quality_9.jpg?raw=true" height="250" width="250">
</a>
<br/>
<br/>

# Introduction

This repo has an original classifier which includes mainly Conv2D, BatchNormalization and Linear layers. This classifier utilizes **1222** and **1047** good and bad quality lemon images respectively.

  - To prevent overfitting problem and assess the skill of model, cross-validation with 5 folds are applied.
  - Train-test set splitted with a ratio of 0.8 <-> 0.2.
  - This classifier is trained with the following parameters:
    - Loss function -> BCE Loss
    - Optimizer -> SGD
    - Learning rate -> 0.001
    - Batch size -> 64
    - Epoch number -> 25
  
Parameters can be updated by editing related lines in [**main.py**]().

# Results

Results of the classifier are assessed by a few evaluation metrics: 
- Accuracy: 98.2%
  - Precision: 99.5%
  - Recall: 96.62%
  - F1 Score: 98.04%

  Training Loss Curve can be seen in below:

  <a href="url">
  <img src="https://github.com/ekaraali/Lemon_Classification_Model/blob/main/images/train_loss_curve.png?raw=true">
  </a>





**Name:** Ramoju Gnana Deepika

**Company:** CODETECH IT solutions

**Id:** CT6WSFC

**Domain:** Data Science

**Duration:** February 25th to April 10th, 2025

**Mentor:** N.Santhosh

## Overview of the project

### Project :Image Classification using Deep Learning (PyTorch)

## Objective:
This project aims to implement a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset. The model learns to recognize objects across 10 categories, such as airplanes, cars, birds, and cats.

## Tools & Technologies:

-**Programming Language:** Python

-**Deep Learning Framework:** PyTorch

-**Dataset:** CIFAR-10 (Preloaded in torchvision.datasets)

-**Visualization:** Matplotlib for plotting training performance and sample predictions

##  Project Workflow:

-**Dataset Preparation:**
-Load CIFAR-10 dataset using torchvision.datasets.

-Normalize pixel values for better training efficiency.

-Use PyTorch DataLoader for batch processing.

-**Model Architecture (CNN):**
-2 Convolutional Layers to extract features.

-MaxPooling Layers to reduce spatial dimensions

-Fully Connected Layers to classify images into 10 categories.

-**Training Process:**
-Use CrossEntropyLoss as the loss function.

-Optimize with Adam optimizer.

-Train for 10 epochs, adjusting model weights based on loss.

-**Evaluation & Accuracy Calculation:**
-Model predicts test images.

-Compare predictions with true labels.

-Compute classification accuracy.

-**Visualization:**
-Training Loss & Accuracy Plot (to track performance).

-Predictions on Sample Images (to verify model output).


## Expected Results:
-Training loss decreases over epochs.

-Accuracy improves, reaching around 70-80% on CIFAR-10.

-Visualization of model predictions on test images.

## Possible Improvements:
-Add data augmentation to improve generalization.
-Use more convolutional layers to increase accuracy.
-Fine-tune learning rate and hyperparameters.
-Experiment with pre-trained models (ResNet, VGG) for better results.

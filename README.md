# Deep Learning Uni Projects

## Project 1: Image Classification with Convolutional Neural Networks

This project focuses on multi-class classification of blood cell images using deep learning techniques. A convolutional neural network (CNN) was designed to process a dataset of labeled blood cell images and classify them into eight distinct categories.

### Problem Overview

The dataset contained 13,759 RGB images (96x96 pixels), categorized into eight classes. After removing artifacts and inconsistencies, the usable dataset consisted of 11,738 images. To address class imbalance, class weight balancing was applied during training.

### Methodology

#### Data Preparation

- **Dataset Splits**:
  - Training: 7,981 images
  - Validation: 1,409 images
  - Test: 2,348 images
- **Data Augmentation**: 
  - Random transformations (e.g., flip, rotation, zoom, noise) were applied to expand the dataset.

#### Model Architecture

- **Base Model**: MobileNetV3Large pretrained on ImageNet.
- **Custom Classifier**:
  - Squeeze-and-Excitation block for feature recalibration.
  - Global Average Pooling (GAP) for dimensionality reduction.
  - Dense layers with Swish activation, L2 regularization, Batch Normalization, and Dropout.
- **Fine-Tuning**:
  - Made 140 layers of MobileNetV3Large trainable for task-specific adaptation.

#### Optimization

- Techniques like EarlyStopping, ReduceLROnPlateau, and balanced class weights were implemented to prevent overfitting and improve performance.

### Results

- **Accuracy**:
  - Internal Test Set: 97.59%
  - Benchmark Platform: 0.67
- The confusion matrix highlighted overfitting to dominant classes, suggesting further refinements in class balancing and preprocessing could enhance performance.

| Notebook | Report |
| --- | --- |
| [Project 1 Notebook](./Image%20Classification/Notebook%20Homework%201.ipynb) | [Project 1 Report](./Image%20Classification/Report%20Homework%201.pdf) |

### Conclusion

While the model achieved high local accuracy, its generalization performance on the benchmark was limited. Future improvements may include better data preprocessing, alternative custom classifiers, and refined class balancing strategies.

---

## Project 2 - Mars Terrain Segmentation Project

This project tackles a multi-class semantic segmentation challenge using U-Net-based deep learning models. The objective was to classify Martian terrain images into five categories: background, soil, bedrock, sand, and big rock.

### Problem Overview

The dataset comprised grayscale images (64x128 pixels) with manually annotated masks. Artifacts were removed, reducing the training set to 2,505 samples. Class imbalance posed a significant challenge, particularly for the "big rock" class, which required targeted interventions.

### Methodology

#### Data Preparation

- **Data Augmentation**: Geometric transformations (e.g., rotation, flipping) were applied consistently to both images and masks to expand the dataset.
- **Data Balancing**: Strategies included:
  - Removing single-class masks.
  - Generating targeted samples for underrepresented classes.

#### Model Architecture

- **Base Model**: Dual U-Net architectures:
  - **Macro U-Net**: Captured global features using dilated convolutions and attention mechanisms.
  - **Micro U-Net**: Preserved finer details and textures with fewer, larger filters and a Global Context Module.
- **Custom Modules**:
  - **Squeeze-and-Excitation Block**: Enhanced channel-wise feature importance.
  - **Dilated Inception Block**: Extracted features at multiple scales.
  - **Cellular Automata Module**: Refined details through iterative local interactions.
- **Ensemble Approach**: Integrated a simpler model targeting the "big rock" class with the advanced architecture.

#### Optimization

- Custom loss functions, including Focal Loss with class-specific weights, were explored to address class imbalance.

### Results

- **Benchmark Score**: 0.52828
- The model performed well for most classes but struggled with "big rock," as reflected in the confusion matrix and low IoU score.

| Notebook | Report |
| --- | --- |
| [Project 2 Notebook](./Image%20Segmentation/anndl-homework-2.ipynb) | [Project 2 Report](./Image%20Segmentation/Report%20Homework%202.pdf) |

### Conclusion

The project demonstrated effective segmentation for most classes but highlighted challenges in handling severe class imbalance. Future work could focus on improved data augmentation, alternative loss functions, or novel architectures to enhance performance, particularly for underrepresented classes.
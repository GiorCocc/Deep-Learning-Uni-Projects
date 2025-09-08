# 🩸 Blood Cell Image Classification Project

A comprehensive deep learning project for multi-class classification of blood cell images using advanced convolutional neural networks and transfer learning techniques.

## 📋 Table of Contents
- [🎯 Project Overview](#-project-overview)
- [📊 Dataset Description](#-dataset-description)
- [🏗️ Model Architecture](#️-model-architecture)
- [🔧 Implementation Details](#-implementation-details)
- [📈 Results & Performance](#-results--performance)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [📖 Key Insights](#-key-insights)
- [🔮 Future Improvements](#-future-improvements)

## 🎯 Project Overview

This project implements a state-of-the-art image classification system for analyzing blood cell images, a critical task in medical diagnostics and hematology research. The system can automatically identify and classify blood cells into 8 distinct categories, potentially assisting medical professionals in diagnosis and research.

### 🏆 Key Achievements
- ✅ **97.59%** accuracy on internal test set
- 📊 **0.67** benchmark platform score
- 🎯 8-class blood cell classification
- 🧠 Advanced transfer learning with MobileNetV3Large
- ⚡ Optimized for both accuracy and efficiency

## 📊 Dataset Description

### 📈 Dataset Statistics
- **Total Images**: 13,759 RGB images (original)
- **Usable Images**: 11,738 images (after cleaning)
- **Image Size**: 96×96 pixels
- **Classes**: 8 distinct blood cell types
- **Color Space**: RGB

### 🔄 Data Splits
| Split | Images | Percentage |
|-------|--------|------------|
| 🏋️ Training | 7,981 | 68.0% |
| ✅ Validation | 1,409 | 12.0% |
| 🧪 Test | 2,348 | 20.0% |

### 🎨 Data Preprocessing & Augmentation
- **Normalization**: Pixel values scaled to [0, 1]
- **Artifact Removal**: Cleaned inconsistent and corrupted images
- **Class Balancing**: Applied class weights to handle imbalanced dataset
- **Data Augmentation**:
  - 🔄 Random horizontal/vertical flips
  - 🔀 Random rotations
  - 🔍 Random zoom
  - ⚡ Gaussian noise injection
  - 🌈 Brightness/contrast adjustments

## 🏗️ Model Architecture

### 🔥 Base Architecture: MobileNetV3Large
- **Pretrained Weights**: ImageNet
- **Input Shape**: (96, 96, 3)
- **Trainable Layers**: 140 layers (fine-tuned)
- **Frozen Layers**: Initial feature extraction layers

### 🧩 Custom Classifier Components

#### 1. 🎯 Squeeze-and-Excitation Block
```
SE Block:
├── Global Average Pooling
├── Dense(filters//16, activation='relu')
├── Dense(filters, activation='sigmoid')
└── Element-wise multiplication
```

#### 2. 🌊 Global Feature Processing
- **Global Average Pooling**: Reduces spatial dimensions
- **Dropout**: 0.3 for regularization
- **Batch Normalization**: Stable training

#### 3. 🧠 Classification Head
```
Classifier:
├── Dense(512, activation='swish')
├── L2 Regularization (0.01)
├── Batch Normalization
├── Dropout(0.4)
├── Dense(256, activation='swish')  
├── Batch Normalization
├── Dropout(0.3)
└── Dense(8, activation='softmax')
```

### ⚙️ Key Architecture Features
- **Activation Function**: Swish (SiLU) for better gradients
- **Regularization**: L2 regularization + Dropout
- **Normalization**: Batch Normalization for stability
- **Attention Mechanism**: Squeeze-and-Excitation for feature recalibration

## 🔧 Implementation Details

### 🎛️ Training Configuration
- **Optimizer**: Adam with adaptive learning rate
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Batch Size**: 32
- **Initial Learning Rate**: 0.001

### 📉 Training Strategies
- **Early Stopping**: Patience of 10 epochs
- **Learning Rate Reduction**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Class Weights**: Balanced to handle class imbalance
- **Validation Monitoring**: Based on validation accuracy

### 🔄 Two-Stage Training Process
1. **Stage 1**: Train only custom classifier (frozen backbone)
2. **Stage 2**: Fine-tune entire network with lower learning rate

## 📈 Results & Performance

### 🏆 Overall Performance
| Metric | Value |
|--------|-------|
| **Internal Test Accuracy** | 97.59% |
| **Benchmark Score** | 0.67 |
| **Training Time** | ~2-3 hours |
| **Model Size** | ~15 MB |

### 📊 Detailed Metrics
- **Precision**: High for majority classes
- **Recall**: Varied across classes due to imbalance
- **F1-Score**: Balanced performance metric
- **Confusion Matrix**: Available in notebook for detailed analysis

### 🎯 Class-wise Performance
- ✅ **Strong Performance**: Well-represented classes
- ⚠️ **Challenges**: Minority classes due to data imbalance
- 🔍 **Overfitting**: Noted on dominant classes

## 📁 Project Structure

```
Image Classification/
├── 📓 Notebook Homework 1.ipynb       # Main implementation notebook
│   ├── 🔍 Data exploration & visualization
│   ├── 🏗️ Model architecture definition
│   ├── 🎯 Training & validation loops
│   ├── 📊 Performance evaluation
│   └── 🔮 Results analysis
├── 📄 AN2DL_Homeworks_Report.pdf      # Comprehensive project report
├── 📊 training_set.npz                # Blood cell dataset
└── 📖 README.md                       # This documentation
```

## 🚀 Getting Started

### 📋 Prerequisites
```python
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
matplotlib>=3.5.0
pandas>=1.4.0
scikit-learn>=1.0.0
seaborn>=0.11.0
PIL>=8.0.0
```

### 🏃‍♂️ Quick Start
1. **Open the Notebook**: Launch `Notebook Homework 1.ipynb`
2. **Install Dependencies**: Run the first cells to install required packages
3. **Load Data**: The notebook automatically loads the dataset
4. **Run Training**: Execute all cells to train the model
5. **Evaluate Results**: Review performance metrics and visualizations

### 💾 Model Loading
```python
# Load pre-trained model
model = tf.keras.models.load_model('best_model.h5')

# Make predictions
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
```

## 📖 Key Insights

### ✨ Technical Innovations
- **Transfer Learning**: Leveraged ImageNet pretrained features
- **Custom Architecture**: Designed domain-specific classifier
- **Attention Mechanism**: SE blocks for feature importance
- **Advanced Regularization**: Multiple techniques to prevent overfitting

### 🧠 Lessons Learned
- **Data Quality**: Critical for model performance
- **Class Balance**: Major impact on generalization
- **Transfer Learning**: Effective for medical imaging
- **Hyperparameter Tuning**: Essential for optimization

### ⚠️ Challenges Addressed
- **Class Imbalance**: Handled with weighted loss and data augmentation
- **Overfitting**: Mitigated with dropout, regularization, and early stopping
- **Limited Data**: Addressed with transfer learning and augmentation
- **Generalization**: Improved with ensemble techniques consideration

## 🔮 Future Improvements

### 🎯 Model Enhancements
- **Ensemble Methods**: Combine multiple model predictions
- **Advanced Augmentation**: More sophisticated data augmentation
- **Architecture Search**: Automated neural architecture search
- **Knowledge Distillation**: Teacher-student training paradigm

### 📊 Data Improvements
- **Data Collection**: Acquire more balanced dataset
- **Quality Control**: Better artifact removal and cleaning
- **Synthetic Data**: Generate synthetic blood cell images
- **Cross-validation**: More robust evaluation methodology

### 🚀 Deployment Considerations
- **Model Optimization**: Quantization and pruning for efficiency
- **Edge Deployment**: Optimize for mobile/edge devices
- **Real-time Inference**: Optimize for clinical workflow integration
- **Interpretability**: Add explainable AI features for medical use

---

## 📞 Contact & Support

For questions, suggestions, or collaborations related to this blood cell classification project, please refer to the main repository or contact the development team.

**🔬 Medical AI • 🩸 Hematology • 🤖 Computer Vision • 📊 Deep Learning**

---

*This project demonstrates the application of advanced deep learning techniques to medical imaging, showcasing the potential of AI in healthcare diagnostics and research.*
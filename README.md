# 🧠 Deep Learning University Projects

This repository contains two comprehensive deep learning projects developed for the Artificial Neural Networks and Deep Learning course, showcasing advanced techniques in computer vision and neural network architectures.

## 🩸 Project 1: Blood Cell Image Classification
**Objective**: Multi-class classification of blood cell images into 8 categories  
**Architecture**: MobileNetV3Large with custom classifier layers  
**Key Results**: 
- ✅ **97.59%** accuracy on internal test set
- 📊 **0.67** benchmark score
- 🔬 Processed 11,738 blood cell images (96x96 pixels)

[📂 View Project Details](./Image%20Classification/)

---

## 🔴 Project 2: Mars Terrain Segmentation  
**Objective**: Semantic segmentation of Martian terrain into 5 classes (background, soil, bedrock, sand, big rock)  
**Architecture**: Dual U-Net with attention mechanisms and ensemble approach  
**Key Results**:
- 🎯 **0.52828** benchmark score
- 🚀 Advanced U-Net architectures with custom modules
- 🌍 Processed 2,505 Mars terrain images (64x128 pixels)

[📂 View Project Details](./Image%20Segmentation/)

---

## 📋 Repository Structure

```
Deep-Learning-Uni-Projects/
├── 📁 Image Classification/           # Blood cell classification project
│   ├── 📓 Notebook Homework 1.ipynb   # Main implementation notebook
│   ├── 📄 AN2DL_Homeworks_Report.pdf  # Detailed project report
│   └── 📊 training_set.npz            # Blood cell dataset
├── 📁 Image Segmentation/             # Mars terrain segmentation project  
│   ├── 📓 anndl-homework-2.ipynb      # Main implementation notebook
│   ├── 📓 big-rock-specialized-model.ipynb # Specialized model for big rocks
│   ├── 📓 ensemble-experiment.ipynb   # Ensemble approach experiments
│   ├── 📄 AN2DL_Homework_2_Report.pdf # Detailed project report
│   └── 📊 mars_for_students.npz       # Mars terrain dataset
└── 📖 README.md                       # This file
```

## 🎯 Key Highlights

- **State-of-the-art Architectures**: Implementation of MobileNetV3Large and advanced U-Net models
- **Custom Modules**: Squeeze-and-Excitation blocks, attention mechanisms, and cellular automata
- **Data Handling**: Comprehensive preprocessing, augmentation, and class balancing strategies  
- **Performance Optimization**: Advanced techniques including focal loss, ensemble methods, and fine-tuning
- **Real-world Applications**: Medical imaging and space exploration computer vision tasks

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV, PIL
- **Data Science**: NumPy, Pandas, Matplotlib
- **Development**: Jupyter Notebooks, Google Colab

---

*Developed as part of the Artificial Neural Networks and Deep Learning course - Showcasing practical applications of advanced deep learning techniques in computer vision.*
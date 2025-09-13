# Deep Learning for Alzheimer's Stage Classification from MRI Scans

A robust deep learning model designed for the analysis and classification of medical images, specifically for identifying the stages of Alzheimer's disease from brain MRI scans. This project leverages **transfer learning with a ResNet50 architecture** to achieve high accuracy, with a special focus on handling severe class imbalance.

---

## 📌 Project Overview
Early diagnosis of Alzheimer's disease is critical for patient care and treatment planning.  
This project develops an **automated system** for stage classification using state-of-the-art computer vision techniques.  
The model classifies an MRI scan into one of four categories:

- **Non-Demented**  
- **Very Mild Demented**  
- **Mild Demented**  
- **Moderate Demented**

The dataset presents a **severe class imbalance**, with later-stage cases being rare. This project implements techniques to overcome this, ensuring the model is a **reliable diagnostic aid**.

---

## ✨ Key Features & Results
- **High-Accuracy Classification**: Achieved **95.42% validation accuracy** on 6,400 MRI images.  
- **ResNet50 Architecture**: Leveraged transfer learning from **ImageNet** for robust feature extraction.  
- **Advanced Imbalance Handling**: Applied **class weighting**, achieving **1.0 precision** for the rarest class (*Moderate Demented*).  
- **Efficient Model Design**: Custom classification head with Global Average Pooling to reduce overfitting.  

---

## 📊 Dataset
- **Source**: [Alzheimer’s MRI Dataset - Kaggle](https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset)  
- **Total Images**: 6,400  
- **Classes**: 4 (Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented)  

**Class Distribution**:
- Non-Demented: 3,200 (50.0%)  
- Very Mild Demented: 2,240 (35.0%)  
- Mild Demented: 896 (14.0%)  
- Moderate Demented: 64 (1.0%)  

---

## ⚙️ Methodology
1. **Data Preprocessing**  
   - Stratified train/val/test split to preserve class ratios.  

2. **Data Augmentation**  
   - Realistic transformations: rotations, shifts, flips.  

3. **Model Architecture**  
   - Pre-trained **ResNet50** as convolutional base.  
   - Final block unfrozen for fine-tuning.  
   - Custom head:  
     - Global Average Pooling  
     - Dense + ReLU  
     - Softmax output layer  

4. **Training**  
   - Optimizer: **Adam**  
   - Loss: **categorical_crossentropy**  
   - Class weights applied to address imbalance.  

5. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-score  
   - Special focus on minority class performance.  

---

## 🏆 Results
| Metric            | Score |
|-------------------|-------|
| **Accuracy**      | 95.42% |
| **Precision (Moderate)** | 1.00 |
| **Recall (Moderate)**    | 1.00 |
| **F1-Score (Moderate)**  | 0.96 |

📌 *The confusion matrix shows strong performance in detecting rare “Moderate Demented” cases without false positives.*

---

## 🚀 How to Run This Project
### Prerequisites
- Python 3.9+  
- TensorFlow 2.10+  
- Scikit-learn  
- Pandas  
- Matplotlib  

### Installation
```bash
# Clone repository
git clone https://github.com/jatinsahu1708/Deep-Learning-for-Alzheimer-s-Stage-Classification.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

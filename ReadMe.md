# Gallstone Prediction from Clinical and Bioimpedance Data

**Predicting gallstone presence using non-invasive clinical, laboratory, and bioimpedance features** with a combination of classical machine learning and deep learning methods.  

This project demonstrates a full data science workflow, including exploratory data analysis, preprocessing, model training, evaluation, and interpretability, using **Python, scikit-learn and TensorFlow**.

---

## 📝 Problem Statement

Gallstone disease is a common digestive disorder. Early prediction using non-invasive metrics such as bioimpedance, laboratory values, and patient demographics can improve patient outcomes and reduce the need for invasive diagnostics.  

**Goal:** Build models that predict gallstone presence from 38 clinical and bioimpedance features.

---

## 📂 Dataset

- Source: [UCI Gallstone Dataset](https://www.kaggle.com/datasets/xixama/gallstone-dataset-uci)  
- Samples: 319 patients  
- Features:  
  - Clinical demographics: Age, Sex, BMI, Comorbidities  
  - Bioimpedance: TBW, ECW, ICW, Lean Mass, Fat Content, Visceral Fat, Bone Mass, etc.  
  - Laboratory: Glucose, Cholesterol, HDL, LDL, Triglycerides, AST, ALT, CRP, Vitamin D, etc.  
- Target: Gallstone presence (binary: 0 = present, 1 = absent)  

---

## 💡 Project Workflow

1. **Exploratory Data Analysis (EDA)**  
   - Visualize distributions, correlations  
   - Identify patterns between features and target  

2. **Data Preprocessing**  
   - Handle missing values (none in this dataset)  
   - Standardize numerical features  
   - Encode categorical variables  
   - **Stratified train/test split** to preserve class balance  
   - **Stratified K-Fold cross-validation** for robust evaluation  

3. **Baseline Model: ElasticNet**  
   - Train linear model with L1/L2 regularization  
   - Hyperparameter tuning via cross-validation  
   - Evaluate metrics: Accuracy, ROC-AUC, Confusion Matrix  
   - Feature coefficients for interpretability  

4. **Neural Network Classifier (MLP)**  
   - 2–3 hidden layers with ReLU activation  
   - Dropout and BatchNorm for regularization  
   - Binary cross-entropy loss, Adam optimizer  
   - Compare performance with ElasticNet baseline  

5. **Model Interpretability**  
   - SHAP values to identify top predictors  
   - Compare insights from ElasticNet and Neural Network  
   - Discuss clinical relevance (e.g., Vitamin D, CRP, Lean Mass)

---

## 📈 Results

| Model          | Accuracy | ROC-AUC |
|----------------|---------|---------|
| ElasticNet     | xx.x%   | 0.xx    |
| Neural Network | xx.x%   | 0.xx    |

**Top predictors (from SHAP & ElasticNet):**


---

## 🛠️ Technologies & Tools

- Python 3.x  
- scikit-learn (ElasticNet, preprocessing, K-Fold CV)  
- TensorFlow  
- pandas, numpy, matplotlib, seaborn (data handling & visualization)  
- SHAP (model interpretability)  
- Jupyter Notebooks for workflow organization  

---

## 🗂️ Repository Structure

gallstone-prediction

├── 01_exploratory_data_analysis.ipynb

├── 02_data_preprocessing.ipynb

├── 03_baseline_elasticnet.ipynb

├── 04_neural_network_model.ipynb

├── 05_model_interpretability.ipynb

│── data/ # dataset placeholder

│── src/ # optional helper scripts


│── README.md
# Gallstone Prediction – Data Science Workflow Demonstration

**Predicting gallstone presence using non-invasive clinical, laboratory, and bioimpedance features** with a combination of classical machine learning and deep learning methods.  

This project explores a medical dataset on gallstones with the goal of **demonstrating a complete data science workflow**, rather than establishing a state-of-the-art predictive model.

The focus is on showing how to:

- Perform exploratory data analysis (EDA), and identify outliers.

- Apply stratification strategies (by gallstone status, gender) and discuss their effect on model performance.

- Implement cross-validation (CV), nested cross-validation, and compare them to baseline models.

- Use feature selection methods (LASSO) to explore model robustness.

- Evaluate models with multiple metrics (accuracy, F1, ROC AUC) and interpret variability across folds.

- Compare traditional ML (logistic regression with Elastic Net regularization) to a simple neural network (NN) to highlight differences in workflow.



## 📂 Dataset

- Source: [UCI Gallstone Dataset](https://www.kaggle.com/datasets/xixama/gallstone-dataset-uci)
- Description and analysis:[ Medicine 103(8):p e37258, February 23, 2024.](https://journals.lww.com/md-journal/toc/2024/02230) | DOI: 10.1097/MD.0000000000037258  
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
   - **Stratified train/test split** to preserve class balance  
   - **Stratified K-Fold cross-validation** for robust evaluation  

3. **Modelling with ElasticNet: Baseline and tuning**  
   - Train linear model with L1/L2 regularization  
   - Hyperparameter tuning via cross-validation
   - **Nested cross-validation**
   - Feature Slection with Lasso
   - Evaluate metrics: Accuracy, ROC-AUC

4. **Neural Network Classifier (MLP)**  
   - 2–3 hidden layers with ReLU activation  
   - Dropout and BatchNorm for regularization  
   - Binary cross-entropy loss, Adam optimizer  
   - Compare performance with ElasticNet baseline  
   - SHAP values to identify top predictors

---

## 📈 Results

| Model          | Accuracy | ROC-AUC |
|----------------|---------|---------|
| ElasticNet     | 77.9%   | 0.843    |
| Neural Network | 78.9%   | 0.831    |

**Top predictors (from SHAP & ElasticNet):**
 
Stratified by gallstone status: CRP, Bone Mass, Total Body Fat Ratio, Intracellular Water, Gender
Stratified by gallstone status + gender: Intracellular Water, Obesity, Vitamin D, Total Body Fat Ratio, Hemaglobin

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

├── 03_regression_model.ipynb

├── 04_neural_network_model.ipynb

├──  data/ saved data splits

├── scripts/ some utiliies for pipeline evaluation

├── results/ for NN part some routines (especially, the nested CV) take a considerable time, so the results are saved in filed

│── README.md
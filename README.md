# Current Progress

## 1. Project Initialization
The project structure was created and organized into separate folders for:
- raw and processed data
- notebooks
- source code
- trained models
- application files

This structure helps keep the project clean, scalable, and easier to maintain.

---

## 2. Dataset Collection
The dataset used in this project is the **Heart Disease Dataset** obtained from **Kaggle**.

The dataset contains clinical and physiological patient information commonly used for heart disease prediction tasks, including:
- age
- sex
- chest pain type
- blood pressure
- cholesterol
- ECG results
- maximum heart rate
- exercise-induced angina
- and other medical attributes

The target variable indicates whether the patient is likely to have heart disease.

---

## 3. Importing Libraries and Loading the Dataset
Essential Python libraries for:
- data manipulation
- numerical computation
- visualization

were imported using:
- Pandas
- NumPy
- Matplotlib
- Seaborn

The dataset was loaded using Pandas, while `pathlib` was used to create a portable and clean file path structure instead of relying on hardcoded paths.

---

## 4. Initial Data Exploration
An initial inspection of the dataset was performed to understand:
- dataset dimensions
- column names
- data types
- general structure of the dataset

This step is important before moving into preprocessing or model training.

---

## 5. Duplicate Record Detection

The dataset was checked for duplicated rows using:

```python
df.duplicated().sum()
```

A large number of duplicate records were detected during the initial data inspection process.

At first, the dataset was split into training and testing sets before removing duplicate samples. However, after further analysis, it became clear that duplicated rows could appear in both the training and testing datasets.

This issue may introduce several problems, including:

- Data leakage
- Unrealistic model evaluation
- Biased learning behavior

This happens because the model may encounter identical samples during both training and testing phases, which can artificially improve evaluation metrics without reflecting true generalization performance.

To avoid this problem, duplicate records were removed before performing the final train-test split.

---

## 6. Duplicate Removal

To avoid potential data leakage and improve data quality, duplicated rows were removed before performing the final train-test split.

This ensures that:

- training and testing datasets remain independent
- evaluation results become more reliable
- the model generalizes better to unseen data

---

## 7. Train-Test Split

After cleaning the dataset, the data was divided into:

- training set
- testing set

using:

```python
train_test_split()
```

A stratified split was applied using:

```python
stratify=y
```

The purpose of stratified sampling is to preserve the original target class distribution in both the training and testing datasets.

This is especially important in medical machine learning problems because an imbalanced split may lead to:

- biased model learning
- misleading evaluation metrics
- weak generalization performance

By maintaining a similar class distribution across both datasets, the model evaluation becomes more reliable and representative of real-world performance.

---

## 8. Distribution Verification After Splitting

After splitting the dataset, several distribution checks were performed to verify that the split was statistically reasonable and did not introduce noticeable sampling bias.

### Target Distribution Check

The target class distribution was compared between:

- training set
- testing set

to ensure that stratified sampling successfully preserved the original balance of the classes.

Maintaining a similar target distribution across both datasets helps improve the reliability of model evaluation and reduces the risk of biased learning behavior.

---

### Age Distribution Check

The age distribution was analyzed using KDE plots and histograms to ensure that:

- both datasets represent similar age groups
- the split did not unintentionally introduce age-related bias

This step is important because a model trained mostly on one age group may fail to generalize properly to patients from other age ranges.

---

### Gender Distribution Check

The gender distribution was also verified in both datasets.

The purpose of this check was not to force perfectly equal male/female ratios, but rather to ensure that:

- the split did not create abnormal demographic imbalance
- both datasets still represent similar patient populations

Performing these verification steps helps ensure that the training and testing datasets remain representative and suitable for reliable machine learning evaluation.

---

## 9. Current Status

At this stage:

- the dataset has been successfully loaded and inspected
- duplicate-related data leakage issues have been identified and resolved
- duplicated records have been removed
- the dataset has been properly split into training and testing sets
- stratified sampling has been applied to preserve target class balance
- target, age, and gender distributions have been verified after splitting

These steps help ensure that the dataset is clean, statistically reasonable, and suitable for reliable machine learning development.

The project is now ready to move into the next phase:

### Discovering and Visualizing the Data (EDA)

In the next stage, deeper exploratory data analysis will be performed to better understand feature distributions, relationships between variables, correlations, potential patterns, and medically relevant insights within the dataset.

---

# Discovering and Visualizing the Data (EDA)

## Initial Statistical Exploration

Basic statistical analysis was performed using:

```python
train_set.describe()
train_set.info()
```

This step helped identify:
- data types
- feature ranges
- potential outliers
- distribution characteristics
- possible anomalies within the dataset

Initial observations indicated that some numerical features, such as cholesterol and oldpeak, may contain extreme values that require further visual investigation during the exploratory data analysis phase.

---

# Discover and Visualize the Data

In this stage of the project, an extensive Exploratory Data Analysis (EDA) process was performed to better understand the structure, quality, and behavior of the dataset before moving into machine learning modeling.

The primary goal of this phase was not only to visualize the data, but also to investigate hidden patterns, detect anomalies, understand feature distributions, and identify relationships between clinical variables and heart disease outcomes.

The analysis was conducted entirely on the training set in order to avoid data leakage and preserve the integrity of the test data for later evaluation.

---

## Numerical Feature Analysis

Several numerical clinical features were analyzed individually using histograms, KDE plots, and boxplots, including:

- Age (`age`)
- Resting Blood Pressure (`trestbps`)
- Cholesterol (`chol`)
- Maximum Heart Rate (`thalach`)
- Oldpeak (`oldpeak`)

The analysis focused on:

- Distribution shape
- Skewness
- Symmetry
- Spread and variability
- Potential outliers
- Clinical plausibility of extreme values

### Key Findings

- `age` and `thalach` appeared relatively close to normal distributions with limited skewness.
- `chol` and `oldpeak` showed noticeable right-skewness and contained several significant outliers.
- Most detected outliers appeared medically plausible rather than obvious measurement errors.
- Extreme cholesterol and oldpeak values may represent high-risk cardiovascular patients instead of invalid records.
- Different numerical features exhibited different statistical behaviors, demonstrating the importance of feature-specific analysis.

---

## Numerical Features vs Target Analysis

Numerical variables were then analyzed in relation to the target variable to investigate whether certain measurements differed between patients with and without heart disease.

Boxplots were used to compare distributions across target groups.

### Main Observations

- `oldpeak` demonstrated one of the clearest separations between target groups and appeared highly informative.
- `thalach` also showed meaningful differences between groups, suggesting useful predictive potential.
- `age` exhibited substantial overlap between groups, indicating limited standalone discriminative power.
- `chol` showed very weak separation despite its clinical importance, suggesting that cholesterol alone may not strongly predict disease status.

These findings highlighted that some features become more useful when combined with others rather than acting as independent predictors.

---

## Categorical Features vs Target Analysis

Categorical clinical variables were analyzed using countplots to investigate how category distributions differed between target classes.

The following features were explored:

- Chest Pain Type (`cp`)
- Exercise Induced Angina (`exang`)
- Thal (`thal`)
- Slope (`slope`)
- Sex (`sex`)

### Main Observations

- `cp`, `thal`, and `slope` showed strong class-related patterns and appeared highly informative for classification.
- `exang` demonstrated noticeable target imbalance across categories, suggesting meaningful predictive information.
- `sex` revealed both target-related differences and dataset imbalance, with male patients representing a significantly larger proportion of the dataset.
- Some categorical features provided clearer class separation than several numerical variables.

---

## Correlation Analysis

A correlation heatmap was generated to investigate linear relationships between features and the target variable.

### Important Correlation Findings

The strongest relationships with the target variable were observed for:

- `cp`
- `exang`
- `thalach`
- `oldpeak`
- `ca`
- `thal`

Meanwhile:

- `chol`
- `fbs`
- `trestbps`

showed relatively weak linear correlations with the target.

The heatmap also revealed several notable feature-to-feature relationships, including:

- Negative correlation between `oldpeak` and `slope`
- Negative correlation between `age` and `thalach`

Many of these relationships were consistent with earlier visual observations made during the EDA process.

---

## Outlier Interpretation and Preprocessing Considerations

During the analysis, several features exhibited outliers and skewed distributions.

However, since this is a medical dataset, these extreme values were not immediately removed because they may represent clinically meaningful high-risk cases rather than invalid measurements.

The EDA findings suggested that:

- The dataset contains a mixture of relatively normal and skewed distributions.
- Some features contain medically plausible extreme observations.
- Robust preprocessing techniques may be more appropriate than aggressive outlier removal.

These conclusions will guide the preprocessing and scaling decisions in the next stage of the project.

---

## Summary of the EDA Stage

This stage provided a comprehensive understanding of:

- Feature distributions
- Statistical behavior
- Outlier patterns
- Feature importance
- Class separation
- Potential dataset imbalance
- Relationships between clinical variables and heart disease outcomes

The insights obtained during this stage will serve as the foundation for the preprocessing, feature preparation, and machine learning modeling phases of the project.


---

# Prepare the Data for Machine Learning

In this stage, the dataset was prepared for machine learning by organizing features, applying preprocessing techniques, and building a structured preprocessing workflow suitable for model training and future deployment.

The preprocessing process was designed carefully to prevent data leakage and preserve clinically meaningful information identified during the exploratory data analysis stage.

---

## Separating Features and Target Variable

The dataset was separated into:

- input features (`X`)
- target variable (`y`)

for both the training and testing datasets.

This separation is required before applying preprocessing techniques and machine learning models.

---

## Feature Type Definition

The features were divided into two main groups:

### Numerical Features
- `age`
- `trestbps`
- `chol`
- `thalach`
- `oldpeak`

### Categorical Features
- `sex`
- `cp`
- `fbs`
- `restecg`
- `exang`
- `slope`
- `ca`
- `thal`

This separation allowed different preprocessing techniques to be applied depending on the feature type.

---

## Scaling Numerical Features

Numerical variables were scaled using:

```text
RobustScaler
```

This scaler was selected based on findings from the exploratory data analysis stage, where several numerical features exhibited:

- skewed distributions
- medically plausible outliers
- non-symmetric behavior

Unlike scaling methods that rely heavily on the mean and standard deviation, `RobustScaler` uses the median and interquartile range (IQR), making it more resistant to extreme values while preserving important clinical information.

Importantly, the scaling process does not remove outliers. Instead, it reduces their influence on feature scaling while retaining potentially meaningful medical observations.

---

## Encoding Categorical Features

Categorical variables were encoded using:

```text
OneHotEncoder
```

Although several categorical features were represented numerically within the dataset, these values represented categories rather than true numerical quantities.

One-Hot Encoding was applied to prevent the machine learning model from incorrectly assuming artificial ordinal relationships between categorical values.

The encoder was configured with:

```python
handle_unknown="ignore"
```

to improve robustness during testing and deployment by safely handling previously unseen categories.

---

## Building the Preprocessing Pipeline

A preprocessing workflow was constructed using:

- `ColumnTransformer`
- `RobustScaler`
- `OneHotEncoder`

This preprocessing structure allowed:

- numerical and categorical features to be processed separately
- preprocessing steps to remain organized and reusable
- data leakage risks to be minimized
- preprocessing consistency between training and testing datasets

The pipeline was fitted only on the training dataset and then applied to the testing dataset using the same learned transformations.

---

## Prepared Dataset Inspection

After preprocessing:

- numerical features were scaled
- categorical variables were expanded through One-Hot Encoding
- transformed features were combined into machine-learning-ready matrices

The resulting prepared datasets were verified to ensure that preprocessing had been applied successfully and that the data was ready for model training.

---

## Summary of the Preprocessing Stage

This stage transformed the raw clinical dataset into a structured machine learning-ready format while preserving important statistical and medical characteristics discovered during the EDA phase.

The preprocessing workflow was designed to:

- maintain data integrity
- reduce the impact of outliers
- avoid data leakage
- standardize preprocessing behavior
- improve future scalability and deployment readiness

The project is now ready to move into the next phase:

# Selecting and Training Machine Learning Models

---

# Select and Train Machine Learning Models

In this stage, multiple machine learning classification models were trained and evaluated to predict heart disease risk using the prepared clinical dataset.

The primary objective of this phase was to:
- establish baseline model performance
- compare generalization behavior across different algorithms
- identify overfitting patterns
- determine which model architecture is most suitable for the dataset

All models were trained using the same preprocessing pipeline and evaluated using consistent classification metrics to ensure fair comparison.

---

## Evaluation Metrics

The following evaluation metrics were used throughout the experiments:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

Special attention was given to the recall metric because false negative predictions are particularly important in medical diagnosis problems.

---

# Logistic Regression

Logistic Regression was selected as the initial baseline model because it is:
- simple
- interpretable
- computationally efficient
- widely used in healthcare classification tasks

### Performance Summary

#### Training Performance
- Accuracy: ~88%
- Recall: ~93%
- F1-score: ~90%

#### Testing Performance
- Accuracy: ~84%
- Recall: ~85%
- F1-score: ~85%

### Observations

- The model demonstrated strong and balanced generalization performance.
- The performance gap between training and testing datasets remained relatively small.
- No severe overfitting was observed.
- The model achieved strong recall performance while maintaining balanced classification behavior.

Among all evaluated baseline models, Logistic Regression produced the strongest overall generalization performance on unseen data.

---

# Decision Tree Classifier

A Decision Tree classifier was evaluated to explore nonlinear decision boundaries and hierarchical feature interactions.

### Performance Summary

#### Training Performance
- Accuracy: ~93%
- Recall: ~97%
- F1-score: ~94%

#### Testing Performance
- Accuracy: ~79%
- Recall: ~79%
- F1-score: ~80%

### Observations

- The model achieved very high training performance.
- A noticeable gap between training and testing performance suggested moderate overfitting.
- The model appeared to memorize portions of the training data rather than generalizing effectively.
- Generalization performance was weaker than Logistic Regression.

---

# Random Forest Classifier

Random Forest was evaluated as an ensemble-based extension of Decision Trees to reduce overfitting and improve stability.

### Performance Summary

#### Training Performance
- Accuracy: ~93%
- Recall: ~98%
- F1-score: ~94%

#### Testing Performance
- Accuracy: ~79%
- Recall: ~82%
- F1-score: ~81%

### Observations

- Random Forest improved stability compared to a single Decision Tree but still exhibited noticeable overfitting.
- Generalization performance remained weaker than Logistic Regression.
- The relatively small dataset size may have limited the advantages of ensemble methods.

---

# Support Vector Machine (SVM)

An SVM classifier was evaluated to test a more flexible nonlinear classification approach.

### Performance Summary

#### Training Performance
- Accuracy: ~93%
- Recall: ~97%
- F1-score: ~93%

#### Testing Performance
- Accuracy: ~79%
- Recall: ~79%
- F1-score: ~80%

### Observations

- SVM also demonstrated moderate overfitting behavior.
- Despite its ability to model complex decision boundaries, it did not outperform Logistic Regression.
- The dataset may contain relatively simple or near-linear feature relationships.

---

# Gradient Boosting Classifier

Gradient Boosting was evaluated as a sequential ensemble learning method designed to iteratively reduce prediction errors.

### Performance Summary

#### Training Performance
- Accuracy: ~99.6%
- Recall: 100%
- F1-score: ~99.6%

#### Testing Performance
- Accuracy: ~77%
- Recall: ~79%
- F1-score: ~79%

### Observations

- The model exhibited severe overfitting.
- Training performance became nearly perfect while testing performance remained relatively weak.
- The model appeared to memorize the training data rather than generalize effectively.

---

# XGBoost Classifier

XGBoost was evaluated as an advanced optimized boosting algorithm designed for structured tabular data.

### Performance Summary

#### Training Performance
- Accuracy: ~99%
- Recall: 100%
- F1-score: ~99%

#### Testing Performance
- Accuracy: ~77%
- Recall: ~73%
- F1-score: ~77%

### Observations

- XGBoost also demonstrated strong overfitting behavior.
- Although highly powerful, the model generalized poorly on unseen data.
- The lower recall performance is particularly important in this medical classification problem because more diseased patients were missed.

---

# Overall Model Comparison

The experiments revealed several important findings:

- More complex models did not necessarily outperform simpler models.
- Ensemble and boosting methods frequently overfitted the relatively small dataset.
- Logistic Regression consistently achieved the best balance between:
  - stability
  - generalization
  - recall performance
  - overall classification quality

These results suggest that:
- the dataset size may limit the effectiveness of highly complex models
- the underlying feature relationships may be relatively simple or near-linear
- simpler models may generalize more effectively for this problem

---

# Conclusion

Among all evaluated baseline models, Logistic Regression emerged as the strongest candidate for further optimization and deployment consideration.

The next stage of the project will focus on:
- hyperparameter tuning
- cross-validation
- threshold optimization
- ROC/AUC analysis
- final model selection

to further improve predictive performance and model reliability.

---

# Fine-Tuning the Machine Learning Models

After comparing multiple baseline machine learning models, the strongest candidates were selected for hyperparameter optimization using cross-validation and GridSearchCV.

The primary objective of this stage was to:
- improve model generalization
- reduce overfitting
- optimize recall performance
- identify the most reliable model configuration for heart disease prediction

Because this project focuses on medical classification, recall was selected as the primary optimization metric in order to minimize false negative predictions.

---

# Cross-Validation Strategy

A 5-fold cross-validation strategy was used during hyperparameter tuning.

This approach:
- divides the training dataset into multiple folds
- trains the model on different subsets
- validates performance across several data splits

Using cross-validation provides more reliable model evaluation compared to relying on a single train-validation split.

---

# Logistic Regression Fine-Tuning

Logistic Regression was selected for optimization because it demonstrated the strongest generalization performance during baseline model comparison.

GridSearchCV was used to search for the best hyperparameter configuration.

### Hyperparameters Evaluated

- Regularization strength (`C`)
- Penalty type
- Solver algorithm

### Best Configuration

```python
{
    'model__C': 0.01,
    'model__penalty': 'l2',
    'model__solver': 'lbfgs'
}
```

### Best Cross-Validation Recall

```text
0.916
```

### Observations

- A relatively small `C` value was selected, indicating that stronger regularization improved generalization performance.
- The tuning process confirmed that simpler and more regularized models perform better on this dataset.
- Logistic Regression maintained strong stability and balanced recall performance across validation folds.

---

# Random Forest Fine-Tuning

Random Forest was selected as the second candidate for optimization because it showed promising recall performance during baseline evaluation.

GridSearchCV was applied to optimize several tree-related hyperparameters.

### Hyperparameters Evaluated

- Number of trees (`n_estimators`)
- Tree depth (`max_depth`)
- Minimum samples required for splitting
- Minimum samples per leaf

### Best Configuration

```python
{
    'model__max_depth': 3,
    'model__min_samples_leaf': 1,
    'model__min_samples_split': 2,
    'model__n_estimators': 50
}
```

### Best Cross-Validation Recall

```text
0.924
```

### Observations

- The tuning process selected shallower trees, suggesting that reducing model complexity improved generalization behavior.
- Tree depth appeared to be the most influential factor affecting overfitting.
- Random Forest achieved strong validation recall performance while reducing excessive model flexibility.

---

# Key Findings

The fine-tuning experiments revealed several important insights:

- Strong regularization improved Logistic Regression performance.
- Reducing tree depth helped mitigate overfitting in Random Forest.
- Cross-validation provided more stable and reliable evaluation compared to relying only on a single train-test split.
- Dataset characteristics continued to favor simpler and more regularized models.

---

# Conclusion

The fine-tuning stage successfully optimized the strongest candidate models using cross-validation and hyperparameter search techniques.

The next stage of the project will focus on:
- final evaluation on the testing dataset
- comparison of tuned models
- confusion matrix analysis
- ROC/AUC analysis
- final model selection

to identify the most reliable model for heart disease prediction.

---

# Evaluate the System on the Test Set

After completing baseline model comparison and hyperparameter tuning, the optimized models were evaluated on the unseen testing dataset to measure real-world generalization performance.

The primary objectives of this stage were to:
- evaluate tuned model performance on unseen data
- compare optimized candidate models
- analyze classification behavior and error patterns
- assess medical prediction reliability
- select the final production-ready model

---

# Final Candidate Models

Two tuned candidate models were selected for final evaluation:

- Tuned Logistic Regression
- Tuned Random Forest

These models were chosen because they demonstrated the strongest validation performance during the fine-tuning stage.

---

# Evaluation Metrics

The final evaluation used multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix
- ROC Curve
- AUC Score

Special emphasis was placed on recall performance because minimizing false negative predictions is critically important in medical diagnosis applications.

---

# Tuned Logistic Regression Evaluation

The tuned Logistic Regression model achieved the strongest overall performance on the testing dataset.

### Final Testing Performance

- Accuracy: ~84%
- Precision: ~83%
- Recall: ~88%
- F1-score: ~85%
- AUC Score: ~0.91

### Observations

- The model maintained strong and stable generalization performance.
- Hyperparameter tuning improved recall performance while preserving overall stability.
- The model demonstrated excellent class separation capability according to ROC/AUC analysis.
- False negative predictions were relatively limited, which is highly important for healthcare-related classification tasks.

The ROC curve remained significantly above the random classification baseline, and the AUC score indicated excellent discrimination ability between diseased and non-diseased patients.

---

# Tuned Random Forest Evaluation

The tuned Random Forest model demonstrated improved performance compared to the baseline Random Forest configuration but remained weaker than Logistic Regression.

### Final Testing Performance

- Accuracy: ~80%
- Precision: ~80%
- Recall: ~85%
- F1-score: ~82%

### Observations

- Hyperparameter tuning reduced overfitting and improved testing stability.
- Shallower trees improved generalization behavior.
- Despite these improvements, Logistic Regression still achieved stronger overall performance.

---

# Final Model Comparison

| Model | Accuracy | Recall | F1-score |
|---|---|---|---|
| Tuned Logistic Regression | ~84% | ~88% | ~85% |
| Tuned Random Forest | ~80% | ~85% | ~82% |

---

# Final Model Selection

Based on:
- generalization performance
- recall performance
- F1-score
- ROC/AUC analysis
- model stability
- overfitting behavior

the tuned Logistic Regression model was selected as the final model for the Heart Disease Prediction System.

---

# Key Findings

The final evaluation phase revealed several important insights:

- Simpler and more regularized models generalized better on this dataset.
- More complex ensemble and boosting models frequently overfitted the relatively small dataset.
- Logistic Regression consistently demonstrated the best balance between:
  - predictive performance
  - recall
  - stability
  - interpretability

The experiments also suggested that the dataset likely contains relatively simple or near-linear relationships that favor less complex models.

---

# Conclusion

The tuned Logistic Regression model achieved strong and reliable predictive performance and demonstrated excellent classification capability according to ROC/AUC analysis.

This model was selected as the final production candidate for the heart disease prediction system due to its:
- strong generalization
- balanced classification behavior
- reduced overfitting
- strong recall performance
- clinical suitability

The next stage of the project will focus on:
- saving and deploying the final model
- building a prediction workflow
- monitoring and maintaining the system
- future improvements and scalability

---

# Application and Deployment Preparation

After completing the machine learning workflow in the notebook, the final model was converted into a reusable application structure using Python files.

## Python Project Structure

The project was organized into reusable modules:

- `src/data_preprocessing.py`  
  Contains the preprocessing pipeline, including numerical scaling and categorical encoding.

- `src/train_model.py`  
  Loads the cleaned dataset, builds the final Logistic Regression pipeline, trains the model, and saves it.

- `src/predict.py`  
  Loads the saved model and provides a prediction function for new patient data.

- `app/app.py`  
  Contains the Streamlit web application interface.

---

## Final Model Saving

The final tuned Logistic Regression pipeline was saved as a `.pkl` file.

The saved pipeline includes:

- preprocessing steps
- RobustScaler
- OneHotEncoder
- tuned Logistic Regression model

This allows the application to make predictions without manually repeating preprocessing steps.

---

## Streamlit Web Application

A Streamlit application was created to provide an interactive user interface.

The application allows users to enter patient clinical information such as:

- age
- sex
- chest pain type
- resting blood pressure
- cholesterol
- maximum heart rate
- exercise-induced angina
- oldpeak
- slope
- ca
- thal

After submitting the inputs, the application returns:

- predicted class
- risk probability
- risk percentage
- risk level: Low, Medium, or High Risk

---

## How the Application Works

The Streamlit app collects patient input values from the user interface and sends them to the prediction function in `src/predict.py`.

The prediction function:

1. Loads the saved final model pipeline.
2. Converts patient input into a Pandas DataFrame.
3. Applies the same preprocessing steps used during training.
4. Generates the prediction and probability score.
5. Converts the probability into a risk category.

This makes the system usable as an end-to-end patient risk prediction tool.

---

## Running the Application Locally

To run the application locally:

```bash
streamlit run app/app.py
```

If Streamlit is not recognized directly, use:

```bash
python -m streamlit run app/app.py
```

---

## Deployment Plan

The project is prepared for deployment using Streamlit Community Cloud.

After uploading the project to GitHub, the application can be deployed by selecting:

```text
app/app.py
```

as the main application file.

Once deployed, the system can be accessed through a public web link and shared on GitHub, LinkedIn, or a professional portfolio.

## Live Demo

https://heart-disease-predictor-ml-hassan.streamlit.app



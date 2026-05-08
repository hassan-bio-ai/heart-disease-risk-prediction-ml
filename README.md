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



# Credit-risk

## Data Exploration

I started project with a comprehensive data exploration. 

Dataset Shape: Explored the shape of the dataset to understand the number of rows and columns.

Summary Statistics: Calculated summary statistics, including minimum, maximum, median, mean, and standard deviation for numeric features.

Missing Values: Checked for missing values in the dataset and decided to explore their impact on the target variable.

Feature Distribution: Utilized Plotly and Matplotlib to visualize the distribution of features, helping to identify potential outliers and data patterns.


## Handling Missing Data

While dealing with missing data, I took a different approach. Instead of filling missing values with the median or mean, I opted for a regression approach. It was clear that the 'loan_int_rate' feature had a substantial impact on the target variable. To predict missing values, I employed a RandomForestRegressor, considering other features as predictors(without target variable loan_status).

The reasoning behind this was to maintain the data's integrity and capture the potential correlation between the target variable and the feature. Using a regression model helped me ensure that the filled values were influenced by the relationships present in the dataset.


## Data Preprocessing

To ensure that the data was prepared for machine learning, I performed feature scaling. Using StandardScaler, I scaled the features, giving more weight to those with higher standard deviations.

During this phase, I also conducted an analysis of multicollinearity among features. This analysis led to the removal of the 'person_age' feature due to multicollinearity and low correlation with target variable.


## Model Selection

For the task of credit risk prediction, I emploeed Automated Machine Learning (AutoML) to identify the most suitable machine learning model. Through this automated process, XGBoost Classifier emerged as the top-performing model, showcasing its prowess in handling the complex task of credit risk prediction. Additionally, I utilized feature importance scores obtained from the XGBoost Classifier to further optimize the model. By removing the least impactful features.


## Hyperparameter Tuning and oversampling

The final step in the model development process was hyperparameter tuning. I leveraged grid search to identify the best hyperparameters for the XGBoost Classifier. Fine-tuning the hyperparameters was essential to enhance the model's performance.

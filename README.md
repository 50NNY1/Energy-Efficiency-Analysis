# IS53051A: Machine Learning Assignment (BSc and MSc):

# Marking criteria:
- Data inspection and visualisation: 5 marks 
- Data pre-processing: 20 marks 
- Models training and optimisation: 48 marks
- Model selection: 5 marks 
- Evaluating the 2 best models on test set: 5 marks 
- Comments and clarity of explanations provided in code: 17 marks 

# Tasks
- energy efficiency prediction problem (regression problem)
- required to perform an analysis using different building shapes, with different characteristics, and predict the heating load of the building.
- The buildings differ with respect to the glazing area, the glazing area distribution, the orientation, and other aspects comprised in the dataset.
- The dataset (below) for this task includes 9 features, denoted by X0, X1, ..., X8, and an outcome variable Y which needs to be predicted.
- The dataset contains missing values.

# You are required to perform this analysis comprising:

1. data inspection and visualisation
    - looking at the data
    - investigating missing values
    - creating histograms and visual inspections
    - examining categorical data
    - examining correlations
    - conclusions of how to handle data in transformations(?) and treatment of missing values
2. data preprocessing including data splitting in 70% training data, and 30% test data
    - [PUT STEPS HERE]
3. data transformations you consider useful for this task, 
    - normalising values if needed
    - manipulating data if needed (eg. changing column names, normalising values [WHAT ELSE HERE?])
    - IS THIS NOT THE SAME AS TREATMENT OF MISSING VALUES?
4. treatment of missing values,
    - replacing with mean, median or dropping
    - replacing categorical data etc
5. feature selection if you consider it useful for helping you achieve a better performance
    - [I DON'T KNOW]

The analysis should include developing the predictive models based on the following algorithms already studied in this module, or that are going to be studied such as neural networks: simple Linear Regression, Ridge Regression, Lasso Regression, Elastic Net Regression, Polynomial Regression with regularisation, and Neural Network. These models (except simple Linear Regression which needs only to be simply trained) will be tuned using the training set. The training set will be used to select the best 2 models. Only these 2 best models will be evaluated on the test set. You can use any Python library.

# The meaning of the 10 variables:
- X0: Category of the building
- X1: Relative Compactness
- X2: Surface Area
- X3: Wall Area
- X4: Roof Area
- X5: Overall Height
- X6: Orientation
- X7: Glazing Area
- X8: Glazing Area Distribution
- Y: Heating Load (outcome variable Y which needs to be predicted)


------------------------------------------------------------------------------
# online references:
https://www.freecodecamp.org/news/end-to-end-machine-learning-project-turorial/



------------------------------------------------------------------------------
# chatgpt and general notes:

### handling missing values:
Replacing missing values with the mean or dropping them from the dataset are two common strategies for handling missing data. However, which strategy to choose depends on the nature of the missing data and the specific problem you are trying to solve.

If the amount of missing data is small and randomly distributed, it may be reasonable to replace missing values with the mean of the non-missing values. This strategy can help preserve the overall distribution of the data and prevent the loss of valuable information. However, if the amount of missing data is large or missing values are non-random, replacing them with the mean may not be appropriate.

If the missing data is large or non-random, it may be better to drop the missing values altogether. This strategy can help prevent the introduction of bias into the analysis and reduce the risk of making incorrect predictions. However, dropping missing values can also result in a loss of information and reduce the sample size, potentially reducing the accuracy of the model.

In terms of when to perform these operations, it is generally recommended to perform data preprocessing steps such as handling missing values before making train-test splits. This is because the train-test splits should reflect the distribution of the original data as closely as possible, and preprocessing steps such as handling missing values can affect the overall distribution of the data. Therefore, it is best to perform these operations on the entire dataset before splitting it into training and testing sets.

------------------------------------------------------------------------------

### handling categorical data:
Non-numerical data, also known as categorical data, cannot be used directly in machine learning algorithms. Therefore, before using categorical data in a machine learning model, it is necessary to convert it into numerical form.

* Label encoding: In this technique, each category is assigned a unique numerical value. For example, if we have a categorical feature "color" with values "red", "green", and "blue", we can assign them numerical values 1, 2, and 3, respectively. Label encoding is simple to implement but can be problematic if the numerical values imply an ordering or magnitude that does not exist in the categorical feature.

* One-hot encoding: In this technique, a new binary column is created for each category in the categorical feature. Each row in the dataset is assigned a 1 in the corresponding category column and 0 in all other category columns. For example, if we have a categorical feature "color" with values "red", "green", and "blue", we can create three new columns "color_red" "color_green", and "color_blue" and assign binary values based on the presence or absence of each category in each row.

* Binary encoding: This technique is similar to one-hot encoding, but it reduces the number of columns required to represent categorical data. In binary encoding, the categories are first encoded as integers and then converted to binary code. For example, if we have a categorical feature "color" with values "red", "green", and "blue", we can first assign them numerical values 1, 2, and 3, respectively, and then convert them to binary code 001, 010, and 100, respectively.

* Count encoding: In this technique, each category is replaced with the count of the number of times it appears in the dataset. This technique can be useful if the frequency of occurrence of each category is important for the analysis.

* Target encoding: In this technique, each category is replaced with the mean of the target variable for that category. This technique can be useful for classification problems where the target variable is a categorical variable.

Overall, the appropriate technique to use for handling categorical data depends on the specifics of your problem, the number of categories, and the type of machine learning algorithm being used. It is important to experiment with different techniques and evaluate their performance on the problem at hand.

------------------------------------------------------------------------------

### conducting sensitivity analysis:
Sensitivity analysis is a technique used to understand how sensitive the output of a machine learning model is to changes in the input variables or parameters. The main goal of sensitivity analysis is to identify the inputs or parameters that have the most significant impact on the model's output and to understand how the model's output would change if these inputs or parameters were changed.

In machine learning, sensitivity analysis can be performed at various stages of the modeling process, depending on the specific problem and the goals of the analysis. Some common stages where sensitivity analysis is performed include:

* During feature selection: Sensitivity analysis can be used to identify the most important features for a given task and to determine how the model's output would change if some of these features were removed or added.

* During hyperparameter tuning: Sensitivity analysis can be used to determine how the model's performance is affected by changes in hyperparameters such as learning rate, regularization strength, or number of hidden layers.

* After model deployment: Sensitivity analysis can be used to understand how the model's predictions are affected by changes in the input data distribution or by changes in the model architecture.

Sensitivity analysis is particularly useful for regression problems because it can help identify which input variables have the greatest influence on the model's predicted output. This information can be used to improve the model's accuracy and interpretability. For example, if the sensitivity analysis indicates that a particular input variable has a large impact on the model's predictions, then the model's performance may be improved by collecting more data on that variable or by incorporating additional features that capture its nuances.

------------------------------------------------------------------------------







------------------------------------------------------------------------------
# chatgpt conversations


### question
this is a housing energy efficiency analysis where Y is representing of the heating load and X3 is representing the wall area
In the context of a housing energy efficiency analysis, the wall area (X3) is an important feature as it can significantly impact the heating load (Y). Therefore, it's important to handle the missing values in X3 appropriately.

### answer
As the percentage of missing values in X3 is relatively low (5.2%), and X3 has a moderate positive correlation with Y (0.31), replacing the missing values with the mean value may be a reasonable approach.

However, it's important to note that the impact of the missing values on the analysis results should be carefully evaluated. If the missing values are not missing at random, and there is a systematic difference between the missing and non-missing values, then replacing the missing values with the mean may introduce bias into the analysis results.

Therefore, it's important to consider other imputation methods and conduct sensitivity analyses to evaluate the impact of missing values on the analysis results.


### getting percentage of missing values and how you can replace the missing values in column X3 with the mean value in Python using Pandas:

```python
import pandas as pd

# Load the dataset into a Pandas dataframe
dataset = pd.read_csv('your_dataset.csv')
# Replace missing values in column X3 with the mean value
mean_X3 = dataset['X3'].mean()
dataset['X3'].fillna(mean_X3, inplace=True)
# Check the updated percentage of missing values in X3
missing_X3 = dataset['X3'].isna().mean()
print(f"Percentage of missing values in X3: {missing_X3:.2%}")

# In this example, we first load the dataset into a Pandas dataframe using the read_csv function. We then calculate the mean value of column X3 using the mean method and replace the missing values in X3 using the fillna method. The inplace=True parameter is used to modify the original dataframe.

# Finally, we check the updated percentage of missing values in X3 using the isna method to identify the missing values and the mean method to calculate the percentage.
```


### our correlation analysis results:
We then create a copy of the dataset called random_drop, where we randomly drop values in X3 to simulate missing values. The probability of dropping a value is equal to the percentage of missing values in X3. We then replace the missing values in random_drop with the mean value, as we did before.

Finally, we calculate the correlation between X3 and Y in the original dataset with mean imputation and in the random_drop dataset with random drop and mean imputation, and print the results.

By comparing the two correlation results, we can evaluate the impact of missing values in X3 on the correlation between X3 and Y. If the two correlation results are similar, it suggests that the missing values in X3 do not have a significant impact on the correlation. If the two correlation results are different, it suggests that the missing values in X3 may have a significant impact on the correlation, and other imputation methods or analysis strategies may be needed.


# machine learning model workflow:

### would it be correct in assume that when we are inspecting data in machine learning we do the following:

- Data inspection and visualisation
- Data pre-processing
- Models training and optimisation
- Model selection 
- Evaluating the 2 best models on test set
- Comments and clarity of explanations provided in code

### Yes, your assumption is generally correct. Here's a brief explanation of each step:

Data inspection and visualization: In this step, you explore the data to understand its structure, quality, and relationships between different variables. This can involve looking at summary statistics, creating visualizations such as histograms and scatter plots, and identifying any missing or inconsistent data.

Data pre-processing: In this step, you prepare the data for modeling by cleaning, transforming, and normalizing it. This can involve tasks such as handling missing values, removing outliers, scaling features, and encoding categorical variables.

Model training and optimization: In this step, you use the pre-processed data to train machine learning models and optimize their hyperparameters. This can involve using various algorithms and techniques, such as decision trees, neural networks, and gradient descent, to find the best possible model for the data.

Model selection: In this step, you compare the performance of different models and select the best one based on some evaluation metric, such as accuracy or AUC.

Evaluating the 2 best models on test set: In this step, you evaluate the performance of the best two models on a separate test set, which the models have not seen before. This helps to ensure that the models generalize well to new data and are not overfitting to the training data.

Comments and clarity of explanations provided in code: In this step, you document your code and provide clear explanations of your methodology, assumptions, and results. This can help others understand and reproduce your work, as well as facilitate collaboration and future development.

### with this in mind, in what steps do we need to check correlations with the target variable?

Checking correlations with the target variable is an important step in machine learning, as it helps to identify the variables that are most predictive of the target and can improve the accuracy of the model. Here are the steps in which you can check correlations with the target variable:

Data inspection and visualization: In this step, you can create scatter plots or heatmaps to visualize the correlation between each independent variable and the target variable.

Data pre-processing: In this step, you can calculate the correlation coefficients between each independent variable and the target variable, and use this information to identify highly correlated variables that may need to be removed or combined.

Model training and optimization: In this step, you can use the correlation coefficients to select the most important variables and optimize the model performance.

Model selection: In this step, you can compare the performance of different models with and without the highly correlated variables, and select the best model based on the evaluation metric.

Evaluating the 2 best models on test set: In this step, you can evaluate the performance of the best two models on the test set, and compare the results with and without the highly correlated variables to ensure that the model is not overfitting.

Comments and clarity of explanations provided in code: In this step, you can document the correlations between the independent variables and the target variable, and provide clear explanations of how these correlations were used to select and optimize the model.


### is it appropriate to perform sensitivity analysis when deciding what to do with missing values from the dataset?

Yes, performing sensitivity analysis can be a useful technique for deciding what to do with missing values from a dataset. Sensitivity analysis involves testing different scenarios or assumptions to see how sensitive the results are to changes in the input data or model parameters. In the case of missing values, sensitivity analysis can help you understand how different approaches to handling missing data might affect the accuracy and reliability of your model.

For example, you could perform sensitivity analysis by comparing the results of different methods for handling missing data, such as imputation or deletion, and examining how sensitive the results are to changes in the amount or pattern of missing data. You could also test the robustness of your model by introducing synthetic missing values or simulating different levels of data quality, and evaluating how well the model performs under these conditions.

By performing sensitivity analysis, you can gain a deeper understanding of the potential impact of missing data on your model, and make more informed decisions about how to handle missing data in a way that minimizes bias and maximizes accuracy.


### with all this in mind, could you rewrite the ML model workflow you originally provided and include these extra steps?

Sure, here is an updated ML model workflow that includes steps for checking correlations with the target variable and performing sensitivity analysis for missing data:

Data inspection and visualization: Explore the data to understand its structure, quality, and relationships between different variables. Create scatter plots or heatmaps to visualize the correlation between each independent variable and the target variable.

Data pre-processing: Prepare the data for modeling by cleaning, transforming, and normalizing it. Calculate the correlation coefficients between each independent variable and the target variable, and use this information to identify highly correlated variables that may need to be removed or combined. Decide how to handle missing values by performing sensitivity analysis and testing different approaches, such as imputation or deletion.

Model training and optimization: Use the pre-processed data to train machine learning models and optimize their hyperparameters. Use the correlation coefficients to select the most important variables and optimize the model performance.

Model selection: Compare the performance of different models with and without the highly correlated variables and different approaches for handling missing values, and select the best model based on the evaluation metric.

Evaluating the 2 best models on test set: Evaluate the performance of the best two models on a separate test set, which the models have not seen before. Compare the results with and without the highly correlated variables and different approaches for handling missing values to ensure that the model is not overfitting.

Comments and clarity of explanations provided in code: Document the correlations between the independent variables and the target variable, and provide clear explanations of how these correlations were used to select and optimize the model. Also, document the approach taken for handling missing values and any sensitivity analysis conducted.


### in this workflow model, where would you say handling missing or non-numerical values in our dataset would be? would this fall under preprocessing or other places?

Handling missing or non-numerical values in the dataset would fall under the preprocessing step. Preprocessing is the stage where you clean and transform your data so that it can be used for machine learning algorithms. This includes handling missing values, converting non-numerical data into numerical data, and normalizing or standardizing the data.

There are several techniques for handling missing values, such as imputation (filling in missing values with a calculated value), deletion (removing rows or columns with missing values), or using algorithms that can handle missing values directly. The specific approach used will depend on the nature and extent of the missing data, as well as the specific machine learning algorithm being used.

Similarly, converting non-numerical data into numerical data is an important step in preprocessing. This can involve techniques such as one-hot encoding (representing categorical variables as binary variables) or label encoding (representing categorical variables as integer values).

Overall, handling missing or non-numerical values is an important part of the preprocessing step, as it ensures that the data is in a suitable format for machine learning algorithms to be applied.
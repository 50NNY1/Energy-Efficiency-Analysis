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
2. data preprocessing including data splitting in 70% training data, and 30% test data
3. data transformations you consider useful for this task, 
4. treatment of missing values,
5. feature selection if you consider it useful for helping you achieve a better performance

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





### conducting sensitivity analysis:
Sensitivity analysis is a technique used to understand how sensitive the output of a machine learning model is to changes in the input variables or parameters. The main goal of sensitivity analysis is to identify the inputs or parameters that have the most significant impact on the model's output and to understand how the model's output would change if these inputs or parameters were changed.

In machine learning, sensitivity analysis can be performed at various stages of the modeling process, depending on the specific problem and the goals of the analysis. Some common stages where sensitivity analysis is performed include:

* During feature selection: Sensitivity analysis can be used to identify the most important features for a given task and to determine how the model's output would change if some of these features were removed or added.

* During hyperparameter tuning: Sensitivity analysis can be used to determine how the model's performance is affected by changes in hyperparameters such as learning rate, regularization strength, or number of hidden layers.

* After model deployment: Sensitivity analysis can be used to understand how the model's predictions are affected by changes in the input data distribution or by changes in the model architecture.

Sensitivity analysis is particularly useful for regression problems because it can help identify which input variables have the greatest influence on the model's predicted output. This information can be used to improve the model's accuracy and interpretability. For example, if the sensitivity analysis indicates that a particular input variable has a large impact on the model's predictions, then the model's performance may be improved by collecting more data on that variable or by incorporating additional features that capture its nuances.


### is it appropriate to perform sensitivity analysis when deciding what to do with missing values from the dataset?

Yes, performing sensitivity analysis can be a useful technique for deciding what to do with missing values from a dataset. Sensitivity analysis involves testing different scenarios or assumptions to see how sensitive the results are to changes in the input data or model parameters. In the case of missing values, sensitivity analysis can help you understand how different approaches to handling missing data might affect the accuracy and reliability of your model.

For example, you could perform sensitivity analysis by comparing the results of different methods for handling missing data, such as imputation or deletion, and examining how sensitive the results are to changes in the amount or pattern of missing data. You could also test the robustness of your model by introducing synthetic missing values or simulating different levels of data quality, and evaluating how well the model performs under these conditions.

By performing sensitivity analysis, you can gain a deeper understanding of the potential impact of missing data on your model, and make more informed decisions about how to handle missing data in a way that minimizes bias and maximizes accuracy.
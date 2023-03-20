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
4. treatment of missinggit st values,
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
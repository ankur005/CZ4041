Library Dependencies
:
1. pandas

2. sklearn

3. scipy

4. numpy

5. xgboost (Install from https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en)


Source Code Files
:
1. generateDataSets.py: Generates OutData/train_set_merged.csv and OutData/test_set_merged.csv containing consolidated train set and test set respectively.

2. trainLasso.py: Trains a model using Lasso Regression. This can be run with or without cross-validation for different values of alpha.

3. trainRFR.py: Trains a model using Random Forest regression. This can be run for different percentage of features from the train set.

4. trainXGBoost.py: Trains a model using Extreme Gradient Boosting. This can be run for different values of hyper parameters.

5. constants.py: Contains constants that store the location of different directories.


How to run
:
1. Install the required library dependencies.

2. Create the following directories -
	a. Data
	b. OutData (On running a model, the merged train and test data set is stored in this directory)
	c. RFR
	d. Regression
	e. XGBoost

3. Download the dataset from kaggle and copy all csv files to Data folder.


4. If running Extreme Gradient Boosting, remember to delete XGBoost/bags directory if it exists.
5. Run either trainLasso.py, trainRFR.py or trainXGBoost.py to train using Lasso Regression, Random Forest Regression or Extreme Gradient Boosting respectively.



How to check Results

:
1. Lasso Regression: Train and Test predictions are generated in Regression/train_predictions_lasso.csv and Regression/test_predictions_lasso.csv respectively.


2. Random Forest Regression: Train and Test predictions are generated in RFR/train_predictions_RFR.csv and RFR/test_predictions_RFR.csv respectively.


3. Extreme Gradient Boosting: Train and test predictions for different bags are generated in XGBoost/bags/[bag_number]/train_predictions_xgboost.csv and XGBoost/bags/[bag_number]/test_predictions_xgboost.csv respectively where [bag_number] refers to the bag number. The mean test predictions are generated in XGBoost/mean_test_predictions_xgboost.csv
import numpy as np
import pandas as pd
import pickle

#The Machine learning alogorithm
from sklearn.ensemble import RandomForestClassifier
# Test train split
from sklearn.model_selection import train_test_split

	
# Just to switch off pandas warning
pd.options.mode.chained_assignment = None

	
# Used to write our model to a file
from sklearn.externals import joblib

#______________________GATHER DATA_______________________________
data = pd.read_csv("titanic_train.csv")
data.head()
# print data.columns
data_inputs = data[["pclass", "age", "sex"]]
print data_inputs.head()
#__________________________________________________________________
median_age = int(data['age'].median())
print "median age is = " + str(median_age)
# mean_age = data['age'].mean()
# print "average age is = " + str(mean_age)
# mode_age = data['age'].mode()
# print "common age is = " + str(mode_age)

#______________________CLEANING DATA___________________________________
data['age'].fillna(median_age, inplace=True)
# print data['age'].head()



data_inputs = data[["pclass", "age", "sex"]]
print data_inputs.head()

expected_output = data[["survived"]]
print expected_output.head()

median_age = data_inputs["age"].median()
data_inputs["age"].fillna(median_age, inplace=True)
data_inputs["pclass"].replace("3rd", 3, inplace=True)
data_inputs["pclass"].replace("2nd", 2, inplace=True)
data_inputs["pclass"].replace("1st", 1, inplace=True)
data_inputs["sex"] = np.where(data_inputs["sex"] == "female", 0, 1)
print data_inputs.head()
#________________________________________________________________________
print "_________TRAINING__________"
	
inputs_train, inputs_test, expected_output_train, expected_output_test   = \
train_test_split (data_inputs, expected_output, test_size = 0.33, random_state = 42)
print inputs_train.head()
print expected_output_train.head()

	
rf = RandomForestClassifier (n_estimators=100)	
rf.fit(inputs_train, expected_output_train)

accuracy = rf.score(inputs_test, expected_output_test)
print("Accuracy = "+str(accuracy*100)+"%")

joblib.dump(rf,"titanic_model1", compress=9)
with open('titanic_pickle_model','wb') as f:
	pickle.dump(rf,f)

import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# Just to switch off pandas warning
pd.options.mode.chained_assignment = None

data = pd.read_csv("titanic_train.csv")
print data.columns

inputs = data[["pclass", "sex"]]
print inputs.head()

inputs["pclass"].replace("1st", 1, inplace = True)
inputs["pclass"].replace("2nd", 2, inplace = True)
inputs["pclass"].replace("3rd", 3, inplace = True)
inputs["sex"] = np.where(inputs["sex"] == "male", 1,0)
print inputs.head()

result = data[["survived"]]
print result.head()

input_train, input_test, output_train, output_test = \
train_test_split(inputs, result, test_size = 0.33, random_state = 42)
print input_train.head()
print output_train.head()
	
clf = RandomForestClassifier (n_estimators=100)	
clf.fit(input_train, output_train)
accuracy = clf.score(input_test, output_test)
print "accuracy = "+str(accuracy*100)+"%"

joblib.dump(clf,"titanic_model2", compress=9)
with open('titanic_pickle_model2','wb') as f:
	pickle.dump(clf,f)
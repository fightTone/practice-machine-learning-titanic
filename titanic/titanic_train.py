import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None
from sklearn.externals import joblib

data = pd.read_csv("titanic_test.csv")
print len(data)


data_inputs = data[["pclass", "sex"]]

# median_age = data_inputs["age"].median()
# data_inputs["age"].fillna(median_age, inplace=True)
data_inputs["pclass"].replace("1st", 1, inplace=True)
data_inputs["pclass"].replace("2nd", 2, inplace=True)
data_inputs["pclass"].replace("3rd", 3, inplace=True)
data_inputs["sex"] = np.where(data_inputs["sex"] == "female", 0, 1)
result = data[["survived"]]
print data_inputs.head()


rf = joblib.load("titanic_model2")
pred = rf.predict(data_inputs)

# print pred
# print result

np.savetxt(r'titanic_result.txt', result.values, fmt='%d')
f = open('titanic_result.txt','rb')
new_res = result.as_matrix()
# for i in f:
# 	new_res.append(i)

x =0
match = 0
for i in pred:
	print str(i)+"="+str(new_res[x])
	if float(i) == float(new_res[x]):
		print "Match"
		match+=1
	else:
		print"mismatch"
	x+=1

print "match = "+str(match)
print "length of pred = "+str(len(pred))
match = float(match)
l = float(len(pred))
percent_diff =  (match/l)  * 100
print "Titanic: percentage match is: "+str(percent_diff)

# def find_err(pred):
# 	titanic_data = np.loadtxt("titanic_result.txt", dtype="int32")
# 	diff_arr = np.equal(titanic_data, pred)
# 	correct_answers = np.sum(diff_arr)
# 	percent_diff = correct_answers / len(pred) * 100
# 	print "Titanic: percentage match is: "+str(percent_diff)
# find_err(pred)
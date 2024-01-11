#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below

# Plot all points
for point in data:
	salary = point[0]
	bonus = point[1]
	plt.scatter(salary, bonus)

plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.show()

## Find the outlier and index
# Reshape the 2D array to 1D array
data_1D = data.reshape((1, len(data)*len(data[0])))[0]
# Find the maximum value
max_point = sorted(data_1D, reverse=True)[0]
# Find the index
for item in data_dict:
	if data_dict[item]['bonus'] == max_point:
		print(item)
# Drop the "querk"
data_dict.pop("TOTAL", 0)

# Check again
data = featureFormat(data_dict, features)

for point in data:
	salary = point[0]
	bonus = point[1]
	plt.scatter(salary, bonus)

plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.show()

# Two more outliers - Find them
for item in data_dict:
	if (data_dict[item]['bonus'] != 'NaN') & (data_dict[item]['salary'] != 'NaN'):
		if (data_dict[item]['bonus'] > 5e6) & (data_dict[item]['salary'] > 1e6):
			print(item)
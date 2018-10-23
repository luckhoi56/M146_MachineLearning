* 10/10/2018

Information gain = entropy(parent) - [m/(m+n)] entropy(child 1) +  [n/(m+n)* entropy(child2)]

* Hyperparameters for decision trees
- In order to create decision tree that will generalize to new problems well, we can tune aspects of the tree. We call these aspects hyperparameters. 

* Maximum depth: The largest length between the root to a leaf. A tree of maximum length k can have at most 2^k leaves.

* Minimum number of samples per leaf:
- Sometimes, we can split unevenly (one has 99 samples where the other node has only one sample). We need to set this in order to avoid this problem

* Minimum number of samples per split:
    This is the same as the minimum number of samples per leaf, but applied on any split of a node.
* Maximum number of features:
    Sometimes we will have too many features to build a tree. If this is the case, in every split, we have to check the entire dataset on each of the features. THis can be very expensive.

* Decision Trees in sklearn:
- In this section, you'll use decision tree fit a given sample dataset.
* Decision tree classifier:

>> from sklearn.tree import DecisionTreeClassifier
>> model = DecisionTreeClassifier()
>> model.fit(x_values, y_values)
>>> print(model.predict([ [0.2, 0.8], [0.5, 0.4] ]))
[[ 0., 1.]]
>>> model = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 10)

The model variable is a decision tree model that has been fitted to the data x_values and y_values. Fitting the model means finding the best line that fits the training data. Let's make two predictions using the model's predict function

# Decision Tree with Titanic Exploration
# import libraries necessary for this project.

import numpy as np
import pandas as pd
from IPython.display import display #allow the use of display() for dataframes

#pretty display for notebooks
%matplotlib inline

# set a random seed
import random
random.seed(42)

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)


# Print the first few entries of th RMS titanic data
display(full_data.head())

# store the 'survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
features_raw = full_data.drop('Survived', axis =1)

# show the new dataset with 'Survived' removed
display(features_raw.head())


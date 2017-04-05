#
# titanic.py
# This file will use the public Titanic data, and using knn, panda and machine learning algorithms will predict whether a person survived the titanic
#
#

import numpy as np
from sklearn import datasets
from sklearn import cross_validation
import pandas as pd

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('titanic.csv', header=0)
df.head()
df.info()

# let's drop columns with too few values or that won't be meaningful
# Here's an example of dropping the 'body' column:
#if doesnt work .values
df = df.drop('body', axis=1)  # axis = 1 means column
df = df.drop('name', axis=1)  # everything except the 'survival' column
df = df.drop('ticket', axis=1)
df = df.drop('embarked', axis=1)
df = df.drop('boat', axis=1)
df = df.drop('home.dest', axis=1)
df = df.drop('age', axis=1)
df = df.drop('sibsp', axis=1)
df = df.drop('parch', axis=1)
df = df.drop('cabin', axis=1)



# let's drop all of the rows with missing data:
df = df.dropna()

# let's see our dataframe again...
# I ended up with 1001 rows (anything over 500-600 seems reasonable)
df.head()
df.info()
    

# You'll need conversion to numeric datatypes for all input columns
#   Here's one example
#
def tr_mf(s):
    """ from string to number
    """
    d = { 'male':0, 'female':1 }
    return d[s]

df['sex'] = df['sex'].map(tr_mf)  # apply the function to the column

# let's see our dataframe again...
df.head()
df.info()


# you will need others!
print("+++ end of pandas +++\n")

# import sys
# sys.exit(0)

print("+++ start of numpy/scikit-learn +++")

# We'll stick with numpy - here's the conversion to a numpy array

# extract the underlying data with the values attribute:
y_data = df[ 'survived' ].values

df = df.drop('survived', axis=1)
df.info()

X_data = df.values

#y_data = df[ 'survived' ].values # also addressable by column name(s)

#
# you can take away the top 42 passengers (with unknown survival/perish data) here:
#
from matplotlib import pyplot as plt

X_test = X_data[0:42]              # the final testing data
X_train = X_data[42:]              # the training data

y_test = y_data[0:42]                  # the final testing outputs/labels (unknown)
y_train = y_data[42:] 


# feature engineering...
X_data[:,1] *= 2   # maybe the first column is worth much more!
X_data[:,2] *= 0.1   # maybe the fourth column is worth much more!


#
# the rest of this model-building, cross-validation, and prediction will come here:
#     build from the experience and code in the other two examples...
#

from sklearn.neighbors import KNeighborsClassifier
max_num = []
#for x in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]:
for x in []:
    knn = KNeighborsClassifier(n_neighbors= 10)
    
#
# cross-validate (use part of the training data for training - and part for testing)
#   first, create cross-validation data (here 3/4 train and 1/4 test)

    knn_score_dataTrain = 0
    knn_score_dataTest = 0

    for i in range(10):
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
        cross_validation.train_test_split(X_train, y_train, test_size=0.25) # random_state=0 

    # fit the model using the cross-validation data
    #   typically cross-validation is used to get a sense of how well it works
    #   and tune any parameters, such as the k in kNN (3? 5? 7? 41?, etc.)
        knn.fit(cv_data_train, cv_target_train)

        knn_score_dataTrain += knn.score(cv_data_train,cv_target_train)
        knn_score_dataTest += knn.score(cv_data_test,cv_target_test)

    knn_dataTrain = knn_score_dataTrain/10
    knn_dataTest = knn_score_dataTest/10

    max_num.append([knn_dataTest, x])

    print("KNN cv training-data average score:", knn_dataTrain)
    print("KNN cv testing-data average score:", knn_dataTest)

#print(max_num) 
#print(max(max_num))


knn = KNeighborsClassifier(n_neighbors= 10)
knn.fit(X_train, y_train) 
print("\nCreated and trained a knn classifier")  #, knn

# here are some examples, printed out:
print("This test's predicted outputs are")
print(knn.predict(X_test))



"""
Comments and results:

Briefly mention how this went:
  
  + what value of k did you decide on for your kNN?
  For my Knn, I decided to go with 10, because it had the highest cross validation testing score.
  
  + how high were you able to get the average cross-validation (testing) score?
  I was able to get my cross validation testing score to 0.791167



Then, include the predicted labels of the 12 digits with full data but no label:
Past those labels (just labels) here:
You'll have 12 lines:

[0 0 0 0 0 0 1 1 0 1 1 1 0 1 0 0 0 0 0 0 0 1 0 1 1 1 1 1 1 0 0 0 1 1 0 0 1
 0 0 1 0 1]

The overall process for this was easier than the others, because I got the hang of everything by now. The hardest part
was finding out what parts of the data to take, for x_test and y_test. Other than that, it was easy to find the cross-reference and figure out what
the best predictions were for the unknowns.

"""

#extra credit- visualizing the data into a HTML file with a table

import csv
from collections import *

def readcsv( csv_file_name = "titanic1.csv" ):
    """
    This takes in a csv file, and converts it into a useable list
    """
    try:
        csvfile = open( csv_file_name, newline='' )  # open for reading
        csvrows = csv.reader( csvfile )              # creates a csvrows object

        all_rows = []                               # we need to read the csv file
        for row in csvrows:
            all_rows.append( [row [2]] + [row [1]] + [row [0]] + [row [3]])

        del csvrows                                  # acknowledge csvrows is gone!
        csvfile.close()                            
        return all_rows                              # return the list of lists

    except FileNotFoundError as e:
        print("File not found: ", e)
        return []

def csv_to_html():
      """
      This converts the titanic csv file to a html table, for a visualization of the passengers- shows their names, if they survived, their passenger class and gender
      """
      data = readcsv()
      len_data = int(len(data))
      
      new_html_string = '<table>'
      
      # new_html_string += '<tr>'
      # new_html_string += '<th>' + data[0][0] + '</th>'
      # new_html_string += '<th>' + data[0][1] + '</th>'
      # new_html_string += '<th>' + data[0][2] + '</th>'
      # new_html_string += '<th>' + data[0][3] + '</th>'
      # new_html_string += '<tr>'
      
      for x in range(len_data):
            new_html_string += '<tr>'
            for i in range(4):
                  printer = data[x][i]
                  new_html_string += '<td>' + printer + '</td>'
            new_html_string += '</tr>'
      
      new_html_string += '</table>'
      return new_html_string
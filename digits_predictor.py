#
#
# digits.py
# This file will predict the type of digit an unknown data point is based on machine learning algorithms, and where the digits pixels lie
#

import numpy as np
from sklearn import cross_validation
import pandas as pd

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('digits.csv', header=0)
df.head()
df.info()

# Convert feature columns as needed...
# You may to define a function, to help out:
def transform(s):
    """ 
    from number to string
    """
    return 'digit ' + str(s)
    
df['label'] = df['64'].map(transform)  # apply the function to the column
print("+++ End of pandas +++\n")

# import sys
# sys.exit(0)

print("+++ Start of numpy/scikit-learn +++")

# We'll stick with numpy - here's the conversion to a numpy array
X_data_full = df.iloc[:,0:64].values        # iloc == "integer locations" of rows/cols
y_data_full = df[ '64' ].values      # also addressable by column name(s)

#
# feature display - use %matplotlib to make this work smoothly
#
from matplotlib import pyplot as plt

X_data_full = X_data_full[0:,]   # 2d array
y_data_full = y_data_full[0:]     # 1d column


X_test = X_data_full[0:23,0:64]              # the final testing data
X_train = X_data_full[23:,0:64]              # the training data

y_test = y_data_full[0:23]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[23:]                  # the training outputs/labels (known)


X_data_full[:,0:4] *= .01   # making the first few pixels worth the least


#
from sklearn.neighbors import KNeighborsClassifier
max_num = []
#for x in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]:
for x in []:
    knn = KNeighborsClassifier(n_neighbors= 5)
    
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


knn = KNeighborsClassifier(n_neighbors= 5)
knn.fit(X_train, y_train) 
print("\nCreated and trained a knn classifier")  #, knn

# here are some examples, printed out:
print("This test's predicted outputs are")
print(knn.predict(X_test))



#def show_digit( Pixels ):
    #""" input Pixels should be an np.array of 64 integers (from 0 to 15) 
        #there's no return value, but this should show an image of that 
        #digit in an 8x8 pixel square
    #"""
    #print(Pixels.shape)
    #Patch = Pixels.reshape((8,8))
    #plt.figure(1, figsize=(4,4))
    #plt.imshow(Patch, cmap=plt.cm.gray_r, interpolation='nearest')  # cm.gray_r   # cm.hot
   # plt.show()
    
# try it!
#row = 15
#Pixels = X_data_full[row:row+1,:]
#show_digit(Pixels)
#print("That image has the label:", y_data_full[row])




"""
Comments and results:

Briefly mention how this went:
  + what value of k did you decide on for your kNN?
  I decided to use 5 for my KNN value for the full-data unknowns

  + how smoothly were you able to adapt from the iris dataset to here?
  It was quite smooth- all we had to do was change the numbers to match the unkowns

  + how high were you able to get the average cross-validation (testing) score?
  0.98657407407407405


Then, include the predicted labels of the 12 digits with full data but no label:
Past those labels (just labels) here:
You'll have 12 lines:

[1 2 3 4 9 6 7 8 9 0 1 2 3 4 5 6 7 8 9 1 2 3 4 5 6 7 8 7 7 3 5 1 0 0 2 2 8
 2 0 1 2 6 3 3 7 3 3 4 6 6]


And, include the predicted labels of the 10 digits that are "partially erased" and have no label:
Mention briefly how you handled this situation!?

In order to handle this situation, I changed:

X_test = X_data_full[0:23,0:64]              
X_train = X_data_full[23:,0:64]  

and 

y_test = y_data_full[0:23]                 
y_train = y_data_full[23:] 

Past those labels (just labels) here:
You'll have 10 lines:

[0 0 0 1 7 7 7 4 0 9 9 9 5 5 6 5 0 9 8 9 8 4 0]

Overall, I enjoyed this problem. Although I had to keep bothering Dodds, I definately learnt a lot, about the nearest neighbors and how to deal with
scraping data to predict future choices.

"""
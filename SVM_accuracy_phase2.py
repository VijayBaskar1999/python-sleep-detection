import pandas as pd
import matplotlib.pyplot as  plt
import numpy as np
df = pd.read_csv("dataset_phase2.csv")
ds=df.sample(frac=1)
ds.to_csv('data2.csv',index=False)
print ("Total number of rows in the dataset : ",ds.shape[0])
X=ds[['minimum_thresh','frames(no of frames eye closed)','yawn count','headframe(no of frames head down)']]
Y = ds['Ground Truth']

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')

average_list=[]

a=ds.shape[0]
m=round(a/10)

X_train = X.iloc[int(m):int(a)]
X_test = X.iloc[0:int(m)]
Y_train = Y.iloc[int(m):int(a)]
Y_test = Y.iloc[0:int(m)]

svclassifier.fit(X_train, Y_train)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10)
avg=svclassifier.score(X_test,Y_test)
print("\nAccuracy for Test[ 0 -",m-1,"] and Train[",m,"-",a-1,"]\t\t\t\t\t:",avg)
average_list.append(avg)

n=m
for i in range(8):

    X_train = X.iloc[np.r_[0:int(n), int(n+m):int(a)]]
    X_test = X.iloc[int(n):int(n+m)]
    Y_train = Y.iloc[np.r_[0:int(n), int(n+m):int(a)]]
    Y_test = Y.iloc[int(n):int(n+m)]

    svclassifier.fit(X_train, Y_train)
    avg=svclassifier.score(X_test,Y_test)
    print("Accuracy for Test[ ",n,"-", n+m - 1, "] and Train[ 0 -", n-1,"]&[",n+m, "-", a - 1, "]\t:", avg)
    average_list.append(avg)
    n=n+m

X_train = X.iloc[0:int(a-m)]
X_test = X.iloc[int(a-m):int(a)]
Y_train = Y.iloc[0:int(a-m)]
Y_test = Y.iloc[int(a-m):int(a)]

svclassifier.fit(X_train, Y_train)
avg=svclassifier.score(X_test,Y_test)
print("Accuracy for Test[",a-m ,"-",a-1,"] and Train[ 0 -",a-m-1,"]\t\t\t\t:",avg)
average_list.append(avg)


print("\n\nAll Accuracies:\n",average_list)
print("\n\nAverage accuracy:",sum(average_list)/len(average_list))


"""
y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
"""
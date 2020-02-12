import pandas as pd
import matplotlib.pyplot as  plt
import numpy as np
df = pd.read_csv("dataset.csv")
df.head()

plt.scatter(df['result'],df['frames'])

x=df[['minimum_thresh','frames']]
y=df['result']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.1)
print(len(x_train))
print(len(x_test))
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train,y_train)
print(y_test)
print(clf.predict(x_test))

print("Accuracy:",clf.score(x_test,y_test)*100)

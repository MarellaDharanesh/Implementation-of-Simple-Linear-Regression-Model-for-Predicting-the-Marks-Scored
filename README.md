# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MARELLA DHARANESH
RegisterNumber: 212222240062 
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv('dataset/student_scores.csv')
df.head()

#segregating data to variables
x = df.iloc[:, :-1].values
x

#splitting train and test data
y = df.iloc[:, -1].values
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
## Output:
![image](https://github.com/MarellaDharanesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707669/e676086e-fbf2-4fa8-b375-9a6cba57cc65)



![image](https://github.com/MarellaDharanesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707669/d6a23098-482a-4063-ab60-5c1ed2c9a108)




![image](https://github.com/MarellaDharanesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707669/01304742-0f72-4f13-aa72-14e1acd8ee2c)




![image](https://github.com/MarellaDharanesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707669/12c9b8aa-69fb-464a-8b3b-4aea7f68b604)



![image](https://github.com/MarellaDharanesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707669/982cb043-de13-4740-9986-08507193ca09)

![image](https://github.com/MarellaDharanesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707669/04a69139-0d55-42c4-a185-9eaa79e956fd)
![image](https://github.com/MarellaDharanesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707669/06f821ea-8018-4168-b1a2-509b2c7ff043)



![image](https://github.com/MarellaDharanesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707669/a16ff482-cc93-4c0b-8b71-b5c12dacb153)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

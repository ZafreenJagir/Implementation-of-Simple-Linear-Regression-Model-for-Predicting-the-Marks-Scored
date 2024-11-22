# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

```
1.Import the standard Libraries.
 2.Set variables for assigning dataset values.
 3.Import linear regression from sklearn.
 4.Assign the points for representing in the graph.
 5.Predict the regression for marks by using the representation of the graph.
 6.Compare the graphs and hence we obtained the linear regression for the given datas.
```


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Zafreen J
RegisterNumber:  212223040252
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
```

```
df.head()
```

```
df.tail()
```


```
x = df.iloc[:,:-1].values
x
```

```
y = df.iloc[:,1].values
y
```

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred

```


```
y_test
```


```
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

```
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```


```
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
```


```
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
```


```
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```



![Screenshot 2024-08-30 134804](https://github.com/user-attachments/assets/8687906e-7cb4-4c7b-bf89-ec8f302a7434)



![Screenshot 2024-08-30 134916](https://github.com/user-attachments/assets/9e62611a-2064-40f6-9d04-5035fa220b68)



![Screenshot 2024-08-30 135013](https://github.com/user-attachments/assets/c60d110b-5cb8-4c78-8f0b-a2a8692a389f)




![Screenshot 2024-08-30 135101](https://github.com/user-attachments/assets/55e82b9f-ed6a-4182-b111-98fb4e167a8d)



![Screenshot 2024-08-30 135244](https://github.com/user-attachments/assets/3297248a-0faf-42a3-98b0-91b55125bfd7)



![Screenshot 2024-08-30 141801](https://github.com/user-attachments/assets/cc77424d-d159-427d-ae74-8c8c1aa78664)



![Screenshot 2024-08-30 141023](https://github.com/user-attachments/assets/e5f847da-8e80-41d7-8f44-6a288615e527)



![Screenshot 2024-08-30 141233](https://github.com/user-attachments/assets/2093a1e1-d93f-4067-9261-7e74f80b884a)



![Screenshot 2024-08-30 141408](https://github.com/user-attachments/assets/8cc1003c-3be6-4350-9390-3f58fd2c9501)



![Screenshot 2024-08-30 141457](https://github.com/user-attachments/assets/39d3208b-229c-4160-b1d2-275c1f44cda7)




![Screenshot 2024-08-30 141625](https://github.com/user-attachments/assets/0aa4e4d3-6fdd-424a-bb62-aecfec93707f)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

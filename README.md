# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.

## Program:

/*
Program to implement the linear regression using gradient descent.
Developed by:Ranjan Kumar G
RegisterNumber:  212223240138
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt  

```
## Dataset:
```
a=pd.read_csv('student_scores.csv')
a
```
## output:
![Screenshot 2024-09-08 201139](https://github.com/user-attachments/assets/430bf434-6d55-4651-ab8f-b8da810f5b40)
## Head and Tail:
```
print(a.head())
print(a.tail())
```
## output:
![image](https://github.com/user-attachments/assets/1761b9f4-8a89-41ad-b3b2-3e3e02f8af97)
## Information of Dataset:
```
a.info()
```
## Output:
![image](https://github.com/user-attachments/assets/4125fe2d-90c7-4318-a94b-7f4a895b3ed9)
## x and y value:
```
x=a.iloc[:,:-1].values
print(x)
y=a.iloc[:,-1].values
print(y)
```
## output:
![image](https://github.com/user-attachments/assets/db93de5f-c9f6-44b9-bf08-a5d17cd62948)
## Program:
```
m=0
c=0
l=0.0001
epochs=5000
n=float(len(x))
error=[]
for i in range(epochs):
  y_pred=m*x + c
  dm=(-2/n) * sum(x*(y-y_pred))
  dc=(-2/n) * sum(y-y_pred)
  m=m - l *dm
  c=c - l *dc
  error.append(sum(y-y_pred)**2)
```
## Display the Output and Error:
```
print(m, c)
type(error)
print(len(error))
```
## output:
![image](https://github.com/user-attachments/assets/8d2c605c-f308-483d-a26e-3684dbaf8387)
## Graph Plotting:
```
plt.plot(range(0,epochs),error)
```
## output:
![Screenshot 2024-09-08 202059](https://github.com/user-attachments/assets/8fe18134-ece5-4c44-b820-07620e3e8aed)
![image](https://github.com/user-attachments/assets/ce21486c-84c5-4626-a8f2-8f5d206319ba)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming

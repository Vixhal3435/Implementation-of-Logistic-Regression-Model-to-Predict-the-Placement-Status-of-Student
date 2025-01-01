# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Problem Definition

Identify the independent variables (X, e.g., grades, internships, and attendance) and the dependent variable (Y, placement status: 1 for placed, 0 for not placed).

2.Load and Preprocess Data

Import the dataset using pandas and inspect it. Handle missing values and encode categorical variables, if any.

3.Split the Data

Divide the dataset into training and testing subsets.

4.Train Logistic Regression Model

Initialize the logistic regression model from sklearn and train it using the training data.

5.Prediction

Use the trained model to predict placement status on the test dataset.

6.Model Evaluation

Evaluate the model using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

7.Visualization (Optional)

Visualize the decision boundary, if applicable, or use an ROC curve to assess model performance.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: vishal.v
RegisterNumber: 24900179 
*/
 import pandas as pd
 data=pd.read_csv(r"C:\Users\admin\Downloads\Placement_Data.csv")
 print(data.head())
 data1=data.copy()
 data1=data1.drop(["sl_no","salary"],axis=1)
 print(data1.head())
  data1.isnull().sum()
 data1.duplicated().sum()
 from sklearn.preprocessing import LabelEncoder
 le=LabelEncoder()
 data1["gender"]=le.fit_transform(data1["gender"])
 data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
 data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
 data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
 data1["degree_t"]=le.fit_transform(data1["degree_t"])
 data1["workex"]=le.fit_transform(data1["workex"])
 data1["specialisation"]=le.fit_transform(data1["specialisation"])
 data1["status"]=le.fit_transform(data1["status"])
 print(data1)
 x=data1.iloc[:,:-1]
 print(x)
 y=data1["status"]
 print(y)
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0
from sklearn.linear_model import LogisticRegression
 lr=LogisticRegression(solver="liblinear")
 lr.fit(x_train,y_train)
 y_pred=lr.predict(x_test)
 print(y_pred)
from sklearn.metrics import accuracy_score
 accuracy=accuracy_score(y_test,y_pred)
 print(accuracy)
from sklearn.metrics import confusion_matrix
 confusion=confusion_matrix(y_test,y_pred)
 print(confusion)
from sklearn.metrics import classification_report
 classification_report1=classification_report(y_test,y_pred)
 print(classification_report1)
 lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:

![image](https://github.com/user-attachments/assets/368cb835-48e9-45ff-9cb9-ee3444d6a949)
![image](https://github.com/user-attachments/assets/36776c8c-23d4-4021-b22d-07e6dd345170)
![image](https://github.com/user-attachments/assets/84bae896-1239-4ab5-8c0f-9d96ac794613)
![image](https://github.com/user-attachments/assets/56d261b7-a85b-4894-80e3-212bf2345a7e)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
result = pd.read_csv("petrol_consumption.csv")
print(result)
print(result.head())
print(result.describe())
print(result.isnull().sum())
print(result.columns)
x=result[["Petrol_tax" ,"Average_income","Paved_Highways","Population_Driver_licence(%)"]]
y=result["Petrol_Consumption"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
r=LinearRegression()
r.fit(xtrain,ytrain)
coefficient=pd.DataFrame(r.coef_,x.columns,columns=["Coefficient"])
print(f"Coefficient is {coefficient}")
yprediction=r.predict(xtest)
df=pd.DataFrame({
    "Actual":ytest,
    "Predicted":yprediction
})
print("Actual vreses prdicted")
print(df)
m=metrics.mean_absolute_error(ytest,yprediction)
nse=metrics.mean_squared_error(ytest,yprediction)
rmse=np.sqrt(nse)
print("Model evaluation")
print(f"mean absolute error is {m}")
print(f"mean squared error is {nse}")
print(f"mean root squared error is {rmse}")
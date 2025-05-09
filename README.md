# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/1fdabb91-3781-4215-9e1b-54665b36d0a2)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/659a6b21-d175-4fe3-b693-9f8517297673)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/db72d631-0bb5-4345-9bd0-fb727f0e7706)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/0db3a6fd-db62-4fe0-bde4-80f2535df4b7)
```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/fad78d9f-f294-41cc-8579-16dabba17b0d)
```
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/37c955d0-b64d-4428-95d8-76ff85807ceb)
```
df3=pd.read_csv("bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/29373d70-e30c-44b8-bcdf-715dab5448ce)
```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
![image](https://github.com/user-attachments/assets/e6ea638e-8b50-4040-a142-45c54ecd9159)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips = sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/84821326-ccbe-45b3-ab19-7fe260e13148)
```
contingency_table = pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/2cfc149c-e5ca-4752-8eff-06d4c5aa7756)
```
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value:{p}")
```
![image](https://github.com/user-attachments/assets/06a699f9-e69d-4f90-9ef0-377605261e92)

# RESULT:
Feature scaling and feature selection process has been successfullyperformed on the data set.

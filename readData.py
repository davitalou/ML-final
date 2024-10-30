import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def readData():
    df = pd.read_csv('Loan_Default.csv')
    x = df.drop('dtir1', axis = 1)
    y = df['dtir1']
    print("x shape: ", x.shape)
    print("data fields:", end=' ')
    oneHotEncoder = LabelEncoder()
    for i in x.columns:
        x[i] = oneHotEncoder.fit_transform(x[i])
        print(i, end=', ')
    print(x)
    return x, y
readData()
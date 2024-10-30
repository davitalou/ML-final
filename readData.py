import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def readData():
    df = pd.read_csv('Customer_support_data.csv')

    x = df.drop('CSAT Score', axis = 1)
    y = df['CSAT Score']
    print("x shape: ", x.shape)
    print("data fields:", end=' ')
    oneHotEncoder = LabelEncoder()
    for i in x.columns:
        x[i] = oneHotEncoder.fit_transform(x[i])
        print(i, end=', ')
    print(x)
    return x, y
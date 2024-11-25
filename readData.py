import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def readData():
    df = pd.read_csv('Loan_Default.csv')

    x = df.drop(['Status', 'ID', 'year'], axis=1)
    x = x[:10000]
    y = df['Status']
    y = y[:10000]
    print("sơ bộ về dữ liệu: ")
    print(x.info())
    oneHotEncoder = LabelEncoder()
    for i in x.columns:
        if not pd.api.types.is_numeric_dtype(x[i]):
            x[i] = oneHotEncoder.fit_transform(x[i])
    y = oneHotEncoder.fit_transform(y)

    mean_values = df.select_dtypes(include=['number']).mean()
    x = x.fillna(mean_values)
    print("dữ liệu sau chuẩn hóa: ")
    print(x.info())
    return x, y


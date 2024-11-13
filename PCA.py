import numpy as np
import pandas as pd
import readData
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

x, y = readData.readData()

# Trực quan hóa thành từng thành phần chính

# phân tích tham số thống kê
# statistics = x.describe()
# print("statistics: ", statistics)

def pca(x):
    #x = x.drop('year', axis=1)
    x = (x - x.mean()) / x.std(ddof = 0)
    x = x.dropna(axis=1, how='all')
    # tính ma trận hiệp phương sai
    x_corr = (1/150) * x.T.dot(x)
    # tính giá trị riêng và vector riêng
    u, s, v = np.linalg.svd(x_corr)
    eig_value, eig_vector = s, u

    # tính lượng thông tin bảo tồn theo phương sai giải thích
    np.linalg.eig(x_corr)
    explained_variance = (eig_value / np.sum(eig_value)) * 100

    sorted_indices = np.argsort(eig_value)[::-1]  # Sắp xếp giảm dần
    sorted_eigenvalues = np.sort(eig_value)[::-1]
    sorted_eigenvectors = np.sort(eig_vector)[::-1]

    # Tính tỷ lệ phương sai tích lũy
    explained_variance = eig_value / np.sum(eig_value)
    cumulative_variance = np.cumsum(explained_variance)
    #tính số chiều tối ưu
    n_component = np.argmax(cumulative_variance >= 0.95) + 1

    selected_vectors = sorted_eigenvectors[:, :n_component]

    return x.dot(selected_vectors)

# Tính các giá trị riêng
print(x)
x_pca = pca(x)
print(x_pca)

pca = PCA(n_components=25)
pca.fit(x)
x_pca_fit = pca.fit_transform(x)
print(x_pca_fit)





import numpy as np
import pandas as pd
import readData
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#x, y = readData.readData()

def pca(x):
    x_scale = (x - x.mean()) / x.std(ddof = 0)
    x_scale = x_scale.dropna(axis=1, how='all')
    # tính ma trận hiệp phương sai
    x_corr = np.cov(x_scale.T)
    # tính giá trị riêng và vector riêng
    u, s, v = np.linalg.svd(x_corr)
    eig_value, eig_vector = s, u

    # Tính tỷ lệ phương sai giải thích và phương sai tích lũy
    explained_variance = eig_value / np.sum(eig_value)
    cumulative_variance = np.cumsum(explained_variance)
    print("Tỷ lệ phương sai giải thích:", explained_variance)
    print("Tỷ lệ phương sai tích lũy:", cumulative_variance)
    # tính số chiều tối ưu
    n_component = np.argmax(cumulative_variance >= 0.9) + 1
    print("Số chiều tối ưu với phương sai tích lũy >=0.9:", n_component)
    pca = PCA(n_components = n_component)
    x_pca_fit = pca.fit_transform(x_scale)
    #hiện hiển thị trực quan từng cặp dữ liệu
    #draw(x)
    #Trực quan hóa mối quan hệ giữa các thành phần chính và đầu ra
    #draw_component_output_connection(x)
    return x_pca_fit
# pca(x)






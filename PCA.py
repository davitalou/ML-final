import numpy as np
import pandas as pd
import readData
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

x, y = readData.readData()
def draw(x):
    n_components = 6  # Hoặc bạn có thể chọn 4, 5 hoặc 6 thành phần chính
    pca = PCA(n_components=n_components)
    pca_6component = pca.fit_transform(x)

    # 3. Tạo DataFrame cho các thành phần chính
    df_pca = pd.DataFrame(pca_6component, columns=[f'PC{i + 1}' for i in range(n_components)])
    fig, axes = plt.subplots(nrows=n_components - 1, ncols=n_components - 1, figsize=(12, 10))

    # Vẽ scatter plot cho từng cặp thành phần chính
    for i in range(n_components):
        for j in range(i + 1, n_components):
            ax = axes[i, j - 1]  # Tạo subplot cho cặp (i, j)
            ax.scatter(df_pca[f'PC{i + 1}'], df_pca[f'PC{j + 1}'], alpha=0.5)
            ax.set_xlabel(f'PC{i + 1}')
            ax.set_ylabel(f'PC{j + 1}')
            ax.set_title(f'Scatter plot: PC{i + 1} vs PC{j + 1}')

    plt.tight_layout()
    plt.show()


def draw_component_output_connection(df_pca):
    df_pca = pd.DataFrame(df_pca)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_pca['dtir1'], df_pca['income'], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel('dtir1')
    plt.ylabel('income')
    plt.title('Scatter plot of dtir1 vs income colored by target variable')
    plt.colorbar(label='Status')
    plt.show()

    # Vẽ thêm các cặp thành phần chính khác với đầu ra (ví dụ: PC1 vs PC3, PC2 vs PC3, ...)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_pca['dtir1'], df_pca['loan_amount'], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel('dtir1')
    plt.ylabel('loan_amount')
    plt.title('Scatter plot of dtir1 vs loan_amount colored by target variable')
    plt.colorbar(label='Status')
    plt.show()
def pca(x):
    # Phân tích các tham số thống kê của dữ liệu
    print(x.head())
    print(x.info())
    print(x.describe())

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
    print("Explained Variance Ratio:", explained_variance)
    print("Cumulative Explained Variance:", cumulative_variance)
    # tính số chiều tối ưu
    n_component = np.argmax(cumulative_variance >= 0.95) + 1

    pca = PCA(n_components = n_component)
    x_pca_fit = pca.fit_transform(x_scale)
    #hiện hiển thị trực quan từng cặp dữ liệu
    draw(x)
    #Trực quan hóa mối quan hệ giữa các thành phần chính và đầu ra
    draw_component_output_connection(x)
    return x_pca_fit
pca(x)






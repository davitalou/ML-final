import numpy as np
import readData
import NaiveBayes, pca
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

def draw_component_output_connection(df_pca, y):
    df_pca = df_pca[:100]
    y = y[:100]
    # for i in range(len(df_pca[0])):
    #     df_pca[0, i] = df_pca[0, i] * 100
    plt.figure(figsize=(10, 6))
    df_pca = pd.DataFrame(df_pca)
    plt.scatter(df_pca[0], y, c=y, cmap='viridis', alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('Status')
    plt.title('biểu đồ phân tán của PC1 va Status')
    plt.colorbar(label='Status')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(df_pca[1], y, c=y, cmap='viridis', alpha=0.7)
    plt.xlabel('PC2')
    plt.ylabel('Status')
    plt.title(f'biểu đồ phân tán của PC2 và Status')
    plt.colorbar(label='Status')
    plt.show()

def draw_couple_main_components(x):
    x = x[:100]
    x = x[:, :6]
    n_components = 6  # Hoặc bạn có thể chọn 4, 5 hoặc 6 thành phần chính
    # pca = PCA(n_components=n_components)
    # pca_6component = pca.fit_transform(x)

    # 3. Tạo DataFrame cho các thành phần chính
    x = pd.DataFrame(x, columns=[f'PC{i + 1}' for i in range(n_components)])

    fig, axes = plt.subplots(nrows=n_components - 1, ncols=n_components - 1, figsize=(12, 10))

    # Vẽ scatter plot cho từng cặp thành phần chính
    for i in range(n_components):
        for j in range(i + 1, n_components):
            ax = axes[i, j - 1]  # Tạo subplot cho cặp (i, j)
            ax.scatter(x[f'PC{i + 1}'], x[f'PC{j + 1}'], alpha=0.5)
            ax.set_xlabel(f'PC{i + 1}')
            ax.set_ylabel(f'PC{j + 1}')
            ax.set_title(f'biểu đồ phân tán: PC{i + 1} vs PC{j + 1}')

    plt.tight_layout()
    plt.show()

x, y = readData.readData()
print("Các phương pháp giảm chiều:")
print("    phân tích tham số thống kê:")
print(x.describe())
print("PCA:")
x_pca = pca.pca(x)
#Trực quan hóa dữ liệu theo từng cặp 2 thành phần chính với 6 thành phần chính
draw_couple_main_components(x_pca)
#Trực quan hóa mối quan hệ của một số chiều dữ liệu chính với đầu ra
draw_component_output_connection(x_pca, y)

print("LDA:")
print("KMeans:")

print("KNN:")
print("Decision Tree:")
print("Naive Bayes:")

print("Perceptron:")
print("Hồi quy KNN:")
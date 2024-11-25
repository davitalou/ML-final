import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import readData
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, classification_report


def draw_correlation_coefficient(y_test, y_pred):
    plt.figure(figsize=(8, 6))

    # Vẽ scatter plot giữa giá trị thực tế và dự đoán
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)

    # Vẽ đường chéo (giới hạn lý tưởng, nơi dự đoán = thực tế)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

    plt.xlabel('Giá trị thực tế (y_true)')
    plt.ylabel('Giá trị dự đoán (y_pred)')
    plt.title('Tương quan giữa Dự đoán và Thực tế')

    plt.show()


def NaiveBayes(x , y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = GaussianNB()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    #draw_correlation_coefficient(y_test, y_pred)
    # đánh giá tương quan
    corr = pearsonr(y_test, y_pred)
    print(f'Hệ số tương quan Pearson: {corr}')
#
# x, y = readData.readData()
# NaiveBayes(x, y)
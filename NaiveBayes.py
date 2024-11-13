import numpy as np
import readData


class MultinomialNaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.likelihoods = {}
        self.classes = []
        self.vocab_size = 0

    def fit(self, X, y):
        n_documents = X.shape[0]
        self.classes = np.unique(y)
        self.vocab_size = X.shape[1]

        # Tính xác suất tiên nghiệm cho từng lớp
        for c in self.classes:
            n_c = np.sum(y == c)  # Số tài liệu thuộc lớp c
            self.class_priors[c] = n_c / n_documents
            # Tính xác suất điều kiện cho từng đặc trưng
            X_c = X[y == c]  # Tài liệu thuộc lớp c
            word_counts = np.sum(X_c, axis=0) + 1  # Thêm 1 để tránh chia cho 0 (Laplace smoothing)
            self.likelihoods[c] = word_counts / (np.sum(word_counts) + self.vocab_size)

    def predict(self, X):
        predictions = []
        X = X.values
        for x in X:
            class_probabilities = np.zeros(len(self.classes))  # Mảng cho xác suất từng lớp
            for idx, c in enumerate(self.classes):
                prior = self.class_priors[c]

                likelihood = np.prod(self.likelihoods[c] ** x)  # Tính xác suất điều kiện cho lớp c
                class_probabilities[idx] = prior * likelihood
                print(x)
            # Dự đoán lớp có xác suất cao nhất
            predicted_class = self.classes[np.argmax(class_probabilities)]
            predictions.append(predicted_class)
        return np.array(predictions)


x, y = readData.readData()
# x = (x - x.mean()) / x.std(ddof = 0)
x_train = x[:1200]
x_test = x[1200:2000]
y_train = y[:1200]
y_test = y[1200:2000]
model = MultinomialNaiveBayes()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
#print("Predictions:", y_pred)
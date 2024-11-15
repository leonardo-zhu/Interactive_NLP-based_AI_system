import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


class IntentClassifier:
    def __init__(self):
        self.count_vect = CountVectorizer(stop_words="english")
        self.tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True)
        self.classifier = LogisticRegression(random_state=0, class_weight="balanced")

    def train(self, texts, labels, test_size=0.25, random_state=42):
        # 构建 Bag-of-Words 和 TF-IDF 特征
        X_counts = self.count_vect.fit_transform(texts)
        X_tfidf = self.tfidf_transformer.fit_transform(X_counts)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, labels, test_size=test_size, random_state=random_state
        )

        # 训练分类器
        self.classifier.fit(X_train, y_train)

        # 模型评估
        y_pred = self.classifier.predict(X_test)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

    def save_model(self, path="intent_classifier.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.classifier, self.count_vect, self.tfidf_transformer), f)

    def load_model(self, path="intent_classifier.pkl"):
        with open(path, "rb") as f:
            self.classifier, self.count_vect, self.tfidf_transformer = pickle.load(f)

    def predict(self, text):
        # 对输入文本进行预测
        input_counts = self.count_vect.transform([text])
        input_tfidf = self.tfidf_transformer.transform(input_counts)
        return self.classifier.predict(input_tfidf)[0]

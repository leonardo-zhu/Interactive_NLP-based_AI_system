import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")


class IntentClassifier:
    def __init__(self, model_path="intent_classifier.pkl"):
        self.model_path = model_path
        self.pipeline = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
            ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
            ('clf', LogisticRegression(random_state=0, class_weight='balanced', max_iter=200))
        ])

    def train(self, texts, labels, test_size=0.25, random_state=42, perform_grid_search=False):
        """
        Train the model on the given texts and labels.
        If perform_grid_search is True, perform hyperparameter tuning using GridSearchCV.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )

        if perform_grid_search:
            # Parameter grid for GridSearchCV
            param_grid = {
                'vect__min_df': [1, 2, 5],
                'vect__max_df': [0.9, 1.0],
                'clf__C': [0.1, 1, 10],
            }

            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                scoring='f1_weighted',
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            print(f"Best Params: {grid_search.best_params_}")
            self.pipeline = grid_search.best_estimator_
        else:
            self.pipeline.fit(X_train, y_train)

        # Evaluation
        y_pred = self.pipeline.predict(X_test)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
        print("Recall:", recall_score(y_test, y_pred, average="weighted", zero_division=0))
        print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    def save_model(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load_model(self):
        with open(self.model_path, "rb") as f:
            self.pipeline = pickle.load(f)

    def predict(self, text):
        if not self.pipeline:
            print("Model not initialized. Please train or load the model first.")
            return None
        if not isinstance(text, str) or text.strip() == "":
            print("Invalid input text.")
            return None

        return self.pipeline.predict([text])[0]
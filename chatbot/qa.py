from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class QuestionAnswering:
    def __init__(self, dataset_path=None):
        self.qa_pairs = {}
        self.vectorizer = TfidfVectorizer()

        if dataset_path:
            self.load_dataset(dataset_path)
        else:
            # 如果没有提供数据集，使用预定义的问答对
            self.qa_pairs = {
                "what is your name": "I am your friendly chatbot.",
                "how are you": "I'm just a bot, but I'm functioning well!",
                "what can you do": "I can chat with you and answer simple questions.",
                "tell me a joke": "Why did the computer go to the doctor? Because it caught a virus!",
            }

        # 使用 TF-IDF 向量化问题
        self.questions = list(self.qa_pairs.keys())
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def load_dataset(self, path):
        """从 CSV 文件加载问答对"""
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            question = row['Question'].strip().lower()
            answer = row['Answer'].strip()
            self.qa_pairs[question] = answer

    def answer(self, user_question):
        """基于语义相似度回答问题"""
        user_vec = self.vectorizer.transform([user_question.lower()])
        similarities = cosine_similarity(user_vec, self.tfidf_matrix)

        max_sim_idx = similarities.argmax()
        max_sim_score = similarities[0, max_sim_idx]

        # 如果相似度高于阈值，返回匹配的答案
        if max_sim_score > 0.5:
            return self.qa_pairs[self.questions[max_sim_idx]]
        else:
            #返回最匹配的答案，同时提示
            uncertain_response = f"I think you might be asking: '{self.questions[max_sim_idx]}'. Here's my best guess:"
            return uncertain_response + f" {self.qa_pairs[self.questions[max_sim_idx]]}"

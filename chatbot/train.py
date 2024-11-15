from chatbot.intents import IntentClassifier

# 数据集：用户输入和对应的意图标签
data = [
    ("hello", "greeting"),
    ("hi there", "greeting"),
    ("hey", "greeting"),
    ("good morning", "greeting"),
    ("goodbye", "farewell"),
    ("see you later", "farewell"),
    ("bye", "farewell"),
    ("what's the weather", "ask_weather"),
    ("is it raining", "ask_weather"),
    ("tell me about the forecast", "ask_weather"),
]

texts, labels = zip(*data)

# 训练并保存模型
classifier = IntentClassifier()
classifier.train(texts, labels)
classifier.save_model("intent_classifier.pkl")
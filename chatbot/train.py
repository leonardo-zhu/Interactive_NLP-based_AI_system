import pandas as pd
from chatbot.intents import IntentClassifier
from collections import Counter

# 加载 CSV 数据
dataset_path = "../data/dataset.csv"
df = pd.read_csv(dataset_path)

# 提取问题和标记意图
csv_data = [(row['Question'].strip().lower(), "dataset_question") for _, row in df.iterrows()]

# 新的训练数据集
data = [
    # 问候语
    ("hello", "greet"),
    ("hi", "greet"),
    ("hey", "greet"),
    ("hey there", "greet"),
    ("good morning", "greet"),
    ("good afternoon", "greet"),
    ("good evening", "greet"),

    # 询问机器人功能
    ("what can you do", "ask_capabilities"),
    ("tell me what you can do", "ask_capabilities"),
    ("what are you capable of", "ask_capabilities"),
    ("can you help me", "ask_capabilities"),
    ("what's your purpose", "ask_capabilities"),

    # 询问名字
    ("what is my name", "ask_name"),
    ("do you know my name", "ask_name"),
    ("tell me my name", "ask_name"),
    ("who am I", "ask_name"),

    # 提供名字
    ("my name is Alice", "tell_name"),
    ("call me Bob", "tell_name"),
    ("i am Charlie", "tell_name"),
    ("please call me David", "tell_name"),
    ("you can call me Emma", "tell_name"),

    # 状态询问
    ("how are you", "ask_how_are_you"),
    ("how are you doing", "ask_how_are_you"),
    ("are you okay", "ask_how_are_you"),
    ("how's it going", "ask_how_are_you"),

    # 告别
    ("bye", "farewell"),
    ("goodbye", "farewell"),
    ("see you", "farewell"),
    ("talk to you later", "farewell"),
]

data.extend([
    ("can you help me with something", "ask_capabilities"),
    ("what can you do for me", "ask_capabilities"),
    ("tell me your capabilities", "ask_capabilities"),
])

data.extend(csv_data)

texts, labels = zip(*data)

label_counts = Counter(labels)
print(label_counts)

# 训练并保存模型
classifier = IntentClassifier()
classifier.train(texts, labels)
classifier.save_model("intent_classifier.pkl")
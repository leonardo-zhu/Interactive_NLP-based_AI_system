import pandas as pd
import nlpaug.augmenter.word as naw
import nltk
nltk.download('wordnet')
from intents import IntentClassifier
from collections import Counter

# load CSV dataset
dataset_path = "../data/dataset.csv"
df = pd.read_csv(dataset_path)

# extract questions and labels
csv_data = [(row['Question'].strip().lower(), "dataset_question") for _, row in df.iterrows()]

# define training dataset
data = []

# greet
data.extend([
    ("hello", "greet"),
    ("hi", "greet"),
    ("hey", "greet"),
    ("hey there", "greet"),
    ("good morning", "greet"),
    ("good afternoon", "greet"),
    ("good evening", "greet"),

    # 增加多样表达
    ("hello there", "greet"),
    ("hi there", "greet"),
    ("hiya", "greet"),
    ("howdy", "greet"),
    ("greetings", "greet"),
    ("hey buddy", "greet"),
    ("hello friend", "greet"),
])

# yes
data.extend([
    ("yes", "yes"),
    ("yep", "yes"),
    ("yeah", "yes"),
    ("sure", "yes"),
    ("of course", "yes"),
    ("absolutely", "yes"),
    ("exactly", "yes"),
    ("definitely", "yes"),
    ("indeed", "yes"),

    # 新增变体
    ("yup", "yes"),
    ("affirmative", "yes"),
    ("certainly", "yes"),
    ("for sure", "yes"),
    ("right", "yes"),
])

# ask name
data.extend([
    ("what is my name", "ask_name"),
    ("do you know my name", "ask_name"),
    ("tell me my name", "ask_name"),
    ("who am I", "ask_name"),
    ("can you tell me my name", "ask_name"),
    ("do you know my name", "ask_name"),  # 与上重复，可换一种说法
    ("what's my name", "ask_name"),
    ("who am i", "ask_name"),
])

# ask capabilities
data.extend([
    ("what can you do", "ask_capabilities"),
    ("tell me what you can do", "ask_capabilities"),
    ("what are you capable of", "ask_capabilities"),
    ("can you help me", "ask_capabilities"),
    ("what's your purpose", "ask_capabilities"),
    ("what can you help with", "ask_capabilities"),
    ("what are your capabilities", "ask_capabilities"),
    ("can you assist me", "ask_capabilities"),
    ("what services do you provide", "ask_capabilities"),
    ("can you help me with something", "ask_capabilities"),
    ("what can you do for me", "ask_capabilities"),
    ("tell me your capabilities", "ask_capabilities"),

    # 增加多样表达
    ("list your abilities", "ask_capabilities"),
    ("in what ways can you assist", "ask_capabilities"),
    ("how can you be useful", "ask_capabilities"),
    ("explain your functions to me", "ask_capabilities"),
    ("what tasks can you handle", "ask_capabilities"),
])

# book flight
data.extend([
    ("booking a flight", "book_flight"),
    ("book a flight", "book_flight"),
    ("i want to book a flight", "book_flight"),
    ("i want to reserve a flight", "book_flight"),
    ("i want to book a flight from london to paris", "book_flight"),
    ("book a round trip from new york to tokyo", "book_flight"),
    ("can i get a ticket to dubai next monday", "book_flight"),

    # 增加多样表达
    ("i need to purchase a flight ticket", "book_flight"),
    ("please help me reserve a flight seat", "book_flight"),
    ("could you arrange a flight for me", "book_flight"),
    ("i want to schedule a flight journey", "book_flight"),
])

# change flight
data.extend([
    ("change my flight", "change_flight"),
    ("reschedule my flight", "change_flight"),
    ("i need to change my flight", "change_flight"),
    ("i want to change my flight", "change_flight"),
    ("can i change my flight", "change_flight"),
    ("i need to reschedule my flight", "change_flight"),

    # 增加多样表达：不仅限于日期，还可涉及其他更改
    ("please shift my flight to another date", "change_flight"),
    ("i want to alter my flight details", "change_flight"),
    ("adjust my flight schedule", "change_flight"),
    ("change my flight itinerary", "change_flight"),
    ("modify my flight reservation", "change_flight"),
    ("could you help me pick a different flight time", "change_flight"),
    ("i would like to select a later flight", "change_flight"),
    ("i want to switch to an earlier flight", "change_flight"),
    ("can i change the departure time of my flight", "change_flight"),
    ("please help me move my flight to another day", "change_flight"),
])

# check my flight
data.extend([
    ("check my flight", "check_flight"),
    ("can you check my flight", "check_flight"),
    ("i want to check my flight", "check_flight"),
    ("i need to check my flight", "check_flight"),
    ("can you verify my flight", "check_flight"),
    ("i want to confirm my flight", "check_flight"),
    ("can you check my flight status", "check_flight"),

    # 增加多样表达
    ("please look up my flight details", "check_flight"),
    ("verify my reservation", "check_flight"),
    ("check the details of my booked flight", "check_flight"),
    ("could you confirm my travel itinerary", "check_flight"),
])

# farewell
data.extend([
    ("bye", "farewell"),
    ("goodbye", "farewell"),
    ("see you", "farewell"),
    ("talk to you later", "farewell"),

    # 增加多样表达
    ("farewell", "farewell"),
    ("see you later", "farewell"),
    ("catch you next time", "farewell"),
    ("so long", "farewell"),
])

# data.extend(csv_data)

print("Total samples before augmentation:", len(data))

# 使用nlpaug进行数据增强
# 创建一个SynonymAug增强器，它使用WordNet来查找同义词
aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=2)

augmented_data = []
for text, label in data:
    # 对每个文本进行数据增强
    aug_texts = aug.augment(text, n=1)  # 增强一条数据
    if isinstance(aug_texts, list):
        aug_text = aug_texts[0]
    else:
        aug_text = aug_texts

    # 将增强后的样本加入数据集
    # 请注意，如果增强结果与原文完全相同(无同义词可替换)，可能需要更多尝试或者使用其它augmenter
    augmented_data.append((aug_text, label))

# 将增强的数据并入原数据集中
data.extend(augmented_data)

print("Total samples after augmentation:", len(data))

def main():
    texts, labels = zip(*data)

    label_counts = Counter(labels)
    print(label_counts)

    # 训练并保存模型
    classifier = IntentClassifier()
    classifier.train(texts, labels)
    classifier.save_model()

    print("Model training complete.")

    return classifier

if __name__ == "__main__":
    main()

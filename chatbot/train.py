import pandas as pd
from chatbot.intents import IntentClassifier
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
])

# ask name
data.extend([
    ("what is my name", "ask_name"),
    ("do you know my name", "ask_name"),
    ("tell me my name", "ask_name"),
    ("who am I", "ask_name"),
    ("can you tell me my name", "ask_name"),
    ("do you know my name", "ask_name"),
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
    ("tell me what you can do", "ask_capabilities"),
    ("what are your capabilities", "ask_capabilities"),
    ("can you assist me", "ask_capabilities"),
    ("what services do you provide", "ask_capabilities"),
    ("can you help me with something", "ask_capabilities"),
    ("what can you do for me", "ask_capabilities"),
    ("tell me your capabilities", "ask_capabilities"),
])

# book flight
data.extend([
    ("i want to book a flight from london to paris", "book_flight"),
    ("book a round trip from new york to tokyo", "book_flight"),
    ("can i get a ticket to dubai next monday", "book_flight"),
    ("change my flight date to tomorrow", "change_flight_date"),
    ("can i move my flight to next week", "change_flight_date"),
    ("reschedule my flight to the 20th of this month", "change_flight_date"),
    ("are there any flights available on friday", "check_flight_availability"),
    ("check flights from berlin to madrid on 25th", "check_flight_availability"),
    ("what flights are available from boston to la next sunday", "check_flight_availability"),
])

# farewell
data.extend([
    ("bye", "farewell"),
    ("goodbye", "farewell"),
    ("see you", "farewell"),
    ("talk to you later", "farewell"),
])


# data.extend(csv_data)

def main():
    data.extend(csv_data)

    texts, labels = zip(*data)

    label_counts = Counter(labels)
    print(label_counts)

    # 训练并保存模型
    classifier = IntentClassifier()
    classifier.train(texts, labels)
    classifier.save_model("intent_classifier.pkl")

    print("Model training complete.")

    return classifier

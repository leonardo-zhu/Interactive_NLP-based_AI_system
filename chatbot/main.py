import train
from chatbot.qa import QuestionAnswering

class TravelBookingChatbot:
    def __init__(self):
        self.name = None
        dataset_path = "../data/dataset.csv"
        self.qa = QuestionAnswering(dataset_path)
        self.intent_classifier = train.main()
        self.intent_classifier.load_model("intent_classifier.pkl")

    def handle_greet(self):
        if not self.name:
            return "Hello! What's your name?"
        else:
            return f"Hello again, {self.name}! How can I assist you?"

    def handle_ask_name(self):
        if self.name:
            return f"Your name is {self.name}."
        else:
            return "I don't know your name yet. Could you tell me your name?"

    def handle_name_update(self, user_input):
        """提取并存储用户名字"""
        self.name = user_input.split()[-1]  # 假设用户最后一个单词是名字
        return f"Nice to meet you, {self.name}!"

    def handle_how_are_you(self):
        return "I'm just a bot, but I'm doing great! How can I help you today?"

    def handle_book_flight(self, user_input):
        # Example simplistic implementation; in practice, connect to a flight booking API
        return "Booking your flight. Details: " + user_input

    def handle_change_flight_date(self, user_input):
        # Example simplistic implementation; in practice, connect to a flight management system
        return "Changing your flight date. Details: " + user_input

    def handle_check_flight_availability(self, user_input):
        # Example simplistic implementation; in practice, check availability via an airline API
        return "Checking flight availability for: " + user_input

    def handle_ask_capabilities(self):
        return "I can help you chat, answer questions, and assist with travel bookings."

    def process_input(self, user_input):
        try:
            # call IntentClassifier to predict intent
            predicted_intent = self.intent_classifier.predict(user_input.lower())

            # handle intent
            if predicted_intent == "greet":
                return self.handle_greet()
            elif "my name is" in user_input.lower() or "i am" in user_input.lower():
                return self.handle_name_update(user_input)
            elif predicted_intent == "ask_name":
                return self.handle_ask_name()
            elif predicted_intent == "ask_how_are_you":
                return self.handle_how_are_you()
            elif predicted_intent == "ask_capabilities":
                return self.handle_ask_capabilities()
            elif predicted_intent == "book_flight":
                return self.handle_book_flight(user_input)
            elif predicted_intent == "change_flight_date":
                return self.handle_change_flight_date(user_input)
            elif predicted_intent == "check_flight_availability":
                return self.handle_check_flight_availability(user_input)
            elif predicted_intent == "dataset_question":  # 处理数据集问题
                return self.qa.answer(user_input)
            else:
                return "I'm sorry, I didn't understand that."
        except ValueError:
            return "I'm not sure what you mean. Could you rephrase that?"

def main():
    chatbot = TravelBookingChatbot()
    print("Welcome to the Travel Booking Chatbot!")
    print("Type 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye! Have a nice day!")
            break

        response = chatbot.process_input(user_input)
        print(f"Chatbot: {response}")


if __name__ == "__main__":
    main()
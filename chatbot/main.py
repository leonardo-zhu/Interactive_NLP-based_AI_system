import re
import train
from qa import QuestionAnswering
from utils import Colors, print_colored

class TravelBookingChatbot:
    def __init__(self):
        self.name = None

        self.flightFrom = None
        self.flightTo = None
        self.flightDate = None
        self.flightPassengers = None

        dataset_path = "./data/dataset.csv"
        self.qa = QuestionAnswering(dataset_path)
        self.intent_classifier = train.main()
        self.intent_classifier.load_model()

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

    def handle_book_flight(self, user_input=None):
        if not self.flightFrom or not self.flightTo or not self.flightDate or not self.flightPassengers:
            self.flightFrom = input("You want to fly from: ").capitalize()
            self.flightTo = input("And to: ").capitalize()
            self.flightDate = input("Please enter the flight date (DD/MM/YYYY): ")

            # if flight date is not in the correct format, ask again
            pattern = r"^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}$"
            while not bool(re.match(pattern, self.flightDate)):
                self.flightDate = input("Please enter the flight date in the correct format (DD/MM/YYYY): ")

            self.flightPassengers = input("How many passengers will be flying? ")

            while not self.flightPassengers.isdigit() or int(self.flightPassengers) <= 0:
                self.flightPassengers = input("Please enter a valid number of passengers: ")

            return f"Booking flight from {self.flightFrom} to {self.flightTo} on {self.flightDate} for {self.flightPassengers} passengers. Is it correct?"

    def handle_change_flight(self, user_input):
        # Example simplistic implementation; in practice, connect to a flight management system
        print_colored(f"Chatbot: Current flight details: {self.handle_check_flight()}\n", Colors.RED)
        print_colored("Chatbot: Would you like to change your flight details?", Colors.RED, '\n')


        print_colored("User: ", Colors.GREEN)
        confirm_change = input().lower()

        # confirm if user wants to change flight


        if self.intent_classifier.predict(confirm_change) == "yes":

            # store previous details
            pre_flightFrom, self.flightFrom = self.flightFrom, None
            pre_flightTo, self.flightTo = self.flightTo, None
            pre_flightDate, self.flightDate = self.flightDate, None
            pre_flightPassengers, self.flightPassengers = self.flightPassengers, None

            self.handle_book_flight()

            # check which details have changed
            changed_details = [
                f"flight from {pre_flightFrom} to {self.flightFrom}" if pre_flightFrom != self.flightFrom else None,
                f"flight to {self.flightTo}" if pre_flightTo != self.flightTo else None,
                f"flight date to {self.flightDate}" if pre_flightDate != self.flightDate else None,
                f"number of passengers to {self.flightPassengers}" if pre_flightPassengers != self.flightPassengers else None,
            ]

            changed_details = [detail for detail in changed_details if detail]

            return "Your flight details have been updated. You have changed " + ", ".join(changed_details) + "."

        return "No problem! Let me know if you need help with anything else."

    def handle_check_flight(self, user_input=None):
        return "Your current details: " + self.flightFrom + " to " + self.flightTo + " on " + self.flightDate + " for " + self.flightPassengers + " passengers."

    def handle_ask_capabilities(self):
        return "You can ask me to book flights, check flight availability, or change your booking details. Just let me know what you need help with!"

    def process_input(self, user_input):
        try:
            # call IntentClassifier to predict intent
            predicted_intent = self.intent_classifier.predict(user_input.lower())

            # handle intent
            if predicted_intent == "greet":
                return self.handle_greet()

            if "my name is" in user_input.lower() or "i am" in user_input.lower():
                return self.handle_name_update(user_input)

            if predicted_intent == "ask_name":
                return self.handle_ask_name()

            if predicted_intent == "ask_how_are_you":
                return self.handle_how_are_you()

            if predicted_intent == "ask_capabilities":
                return self.handle_ask_capabilities()

            if predicted_intent == "book_flight":
                return self.handle_book_flight(user_input)

            if predicted_intent == "change_flight":
                return self.handle_change_flight(user_input)

            if predicted_intent == "check_flight":
                return self.handle_check_flight(user_input)

            if predicted_intent == "yes":
                return "Great! What else can I help you with?"

            if predicted_intent == "dataset_question":
                return self.qa.answer(user_input)

            return "I'm sorry, I didn't understand that."
        except ValueError:
            return "I'm not sure what you mean. Could you rephrase that?"

def main():
    chatbot = TravelBookingChatbot()
    print_colored("Welcome to the Travel Booking Chatbot!\n", Colors.CYAN)
    print_colored("Type 'exit' to end the conversation.\n", Colors.CYAN)

    print()

    while True:
        print_colored("User: ", Colors.GREEN)
        user_input = input().strip()
        if user_input.lower() in ["exit", "quit"]:
            print_colored("Chatbot: Goodbye! Have a nice day!", Colors.BLUE)
            break

        response = chatbot.process_input(user_input)
        print_colored(f"Chatbot: {response}", Colors.RED, '\n')


if __name__ == "__main__":
    main()
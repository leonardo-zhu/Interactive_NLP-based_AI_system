from chatbot.intents import classify_intent
from chatbot.qa import answer_question
from chatbot.handle_dialogue import handle_dialogue

def welcome_message():
    print("Welcome to Mini Chatbot!")
    print("Type 'exit' if you want to end the conversation.\n")

def get_user_input():
    return input("Enter something: ")

# main function
def main():
    welcome_message()
    while True:
        user_input = get_user_input()
        if user_input == "exit":
            print("Chatbot: Goodbye! Have a great day!")
            break
        else:
            print("You said: " + user_input)

        # classify intent
        intent = classify_intent(user_input)

        # 根据意图处理对话逻辑
        if intent == "greeting":
            response = "Hello! How can I help you today?"
        elif intent == "question":
            response = answer_question(user_input)
        elif intent == "small_talk":
            response = handle_dialogue(user_input)
        else:
            response = "I'm sorry, I didn't quite understand that."

        # 输出响应
        print(f"Chatbot: {response}")

# run main function
if __name__ == "__main__":
    main()
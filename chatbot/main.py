from chatbot.intents import IntentClassifier

def welcome_message():
    print("Welcome to Mini Chatbot!")
    print("Type 'exit' if you want to end the conversation.\n")

def get_user_input():
    """获取用户输入"""
    try:
        return input("You: ").strip()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        exit()

# main function
def main():
    welcome_message()
    classifier = IntentClassifier()
    classifier.load_model("intent_classifier.pkl")

    # 聊天循环
    while True:
        # 获取用户输入
        user_input = get_user_input()

        # 检查退出条件
        if user_input.lower() in {"exit", "quit"}:
            print("Chatbot: Goodbye! Have a nice day!")
            break

        # 预测用户意图
        predicted_intent = classifier.predict(user_input)

        print("predicted_intent", predicted_intent)

        response = "Sorry, I didn't understand that."

        # 根据意图生成响应
        if predicted_intent == "greeting":
            response = "Hello! How can I help you today?"
        elif predicted_intent == "farewell":
            response = "Goodbye! Take care!"
        elif predicted_intent == "ask_weather":
            response = "It's sunny outside, a great day to enjoy!"
        elif predicted_intent == "ask_name":
            response = "I'm your friendly chatbot. What's your name?"

        # 输出响应
        print(f"Chatbot: {response}")

# run main function
if __name__ == "__main__":
    main()
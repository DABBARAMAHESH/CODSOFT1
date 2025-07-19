def simple_chatbot():
    print("🤖 Hello! I'm your chatbot. Type 'bye' to exit.")

    while True:
        user_input = input("You: ").lower()

        if "hello" in user_input or "hi" in user_input:
            print("Bot: Hello there! How can I help you today?")

        elif "how are you" in user_input:
            print("Bot: I'm just a bunch of code, but I'm doing great! How about you?")
        elif "your name" in user_input:
            print("Bot: I'm a simple chatbot made by a Python program.")
        elif "help" in user_input:
            print("Bot: Sure! I can respond to greetings, tell you my name, and chat a bit.")
        elif "bye" in user_input or "exit" in user_input:
            print("Bot: Goodbye! Have a great day! 👋")
            break
        else:
            print("Bot: I'm not sure how to respond to that. Try asking something else.")

# Run the chatbot
simple_chatbot()
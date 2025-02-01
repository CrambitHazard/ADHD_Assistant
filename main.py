from llama_utils import load_model, get_response

def chatbot():
    model, tokenizer = load_model()
    print("AI Assistant: Hello! How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("AI Assistant: Goodbye!")
            break
        response = get_response(model, tokenizer, user_input)
        print(f"AI Assistant: {response}")

if __name__ == "__main__":
    chatbot()

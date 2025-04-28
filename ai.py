import google.generativeai as genai

API_KEY = "AIzaSyAarMYFla_9iFsGfZ3_d8UfNhOoGZJu1uc"
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

chat = model.start_chat()

# response = chat.send_message("Hello, how are you?")
# print("Gemini:",response.text)  # Output: "I'm doing well, thank you! How can I assist you today?"

print("Chat with Gemini AI (type 'exit' to quit):")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = chat.send_message(user_input)
    print("Gemini:", response.text)  # Output: Gemini's response to the user's input
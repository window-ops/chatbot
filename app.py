import PySimpleGUI as sg
import transformers
from transformers import Conversation, AutoTokenizer
import torch

# Load the pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-large"
model = transformers.pipeline("conversational", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", pad_token="[PAD]")

# Define the conversation loop
conversation = Conversation()

# Set the theme
sg.theme("DefaultNoMoreNagging")

# Define the GUI layout
layout = [
    [sg.Multiline(size=(80, 20), key="-OUTPUT-", disabled=True)],
    [sg.InputText(key="-INPUT-", size=(74, 1), focus=True, enable_events=True), sg.Button("Send", size=(5, 1), bind_return_key=True)]
]

# Create the GUI window
window = sg.Window("Chatbot", layout)

# Loop to process user input and display bot responses
while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break

    if event == "_INPUT_" or event == "Send":
        # Get user input
        user_input = values["-INPUT-"]

        # Tokenize user input and set attention mask and pad token ID
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs["input_ids"] = inputs["input_ids"].to(model.device)
        inputs["attention_mask"][:, :inputs["input_ids"].shape[1]] = 0

        # Add user input to the conversation
        conversation.add_user_input(user_input, inputs)

        # Generate response using the pre-trained model
        conversation = model(conversation)

        # Get the latest bot response
        bot_response = conversation.generated_responses[-1]

        # Update the conversation history in the GUI
        window["-OUTPUT-"].print(f"You: {user_input}\nChatbot: {bot_response}")

        # Clear the input field
        window["-INPUT-"].update("")

# Close the GUI window
window.close()

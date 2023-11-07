from transformers import AutoModelForCausalLM, AutoTokenizer
from textblob import TextBlob

# import nltk

# Necessary to download components once before usage
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# Loading pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
tokenizer.padding_side = 'left'  # Set padding_side after initializing the tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def correct_text(input_text):
    text_blob = TextBlob(input_text)
    corrected_text = text_blob.correct()
    return str(corrected_text)

# Function for chatbot with text pre-check
def chatbot_response(input_text):
    #Сorrecting the text before processing
    corrected_input_text = correct_text(input_text)

    #Check if the text has been changed
    if corrected_input_text != input_text:
        print("Исправленный текст:", corrected_input_text)

    # Encoding the input text and adding special tokens
    input_ids = tokenizer.encode(corrected_input_text + tokenizer.eos_token + "", return_tensors='pt')

    # Generating model response
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decoding the response and returning it
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response


# Chat with the chatbot
print("Chat-bot activated. Write something to start the conversation (type 'exit' to finish)")
while True:
    # Receiving input from the user
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chat-bot deactivated.")
        break

    # Receiving and displaying chatbot response
    response = chatbot_response(user_input)
    print("Chat-bot: ", response)

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from textblob import TextBlob

# Loading pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def correct_text(input_text):
    text_blob = TextBlob(input_text)
    corrected_text = text_blob.correct()
    return str(corrected_text)

def chatbot_response(input_text):
"""
A function for generating a chatbot response to an input text.

Args:
input_text (str): The input text to generate the response based on.

Returns:
str: The generated chatbot response.
"""
    corrected_input_text = correct_text(input_text)
    if corrected_input_text != input_text:
        st.write("Corrected text:", corrected_input_text)
    input_ids = tokenizer.encode(corrected_input_text + tokenizer.eos_token + "", return_tensors='pt')
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Streamlit interface
st.title("Chatbot with Text Correction")

# Initialize conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ""

user_input = st.text_input("You: ", key="input_text")
if st.button("Send"):
    response = chatbot_response(user_input)

    # Update conversation history
    st.session_state.conversation_history += f"You: {user_input}<br>Chat-bot: {response}<hr>"

    # Display conversation history using st.markdown with HTML formatting
    st.markdown("#### Conversation History:", unsafe_allow_html=True)
    st.markdown(st.session_state.conversation_history, unsafe_allow_html=True)

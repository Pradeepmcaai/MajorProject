#chatbot using microsoft/DialoGPT-medium
# import required libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Define stop words to be removed
stop_words = set(stopwords.words('english'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()



# Define function to preprocess text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenize text into words
    words = word_tokenize(text)

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join words back into text
    text = ' '.join(words)

    return text

# Load the DialoGPT
chatbot_name = "microsoft/DialoGPT-medium"
chatbot_tokenizer = AutoTokenizer.from_pretrained(chatbot_name, padding_side='left')
chatbot_tokenizer.add_special_tokens({'pad_token': chatbot_tokenizer.eos_token})  # Add a new pad token
chatbot_model = AutoModelForCausalLM.from_pretrained(chatbot_name)

# Load the ASC606classification model
# Load the ASC606classification model
model_path = "./twitter_full.pkl"
# Load the model
with open(model_path, 'rb') as file:
    twitter_classifier = pickle.load(file)

# Function to generate chatbot response
def generate_response(input_text):
    bot_input_ids = chatbot_tokenizer.encode(input_text + chatbot_tokenizer.eos_token, return_tensors='pt', padding=True)
    chatbot_output = chatbot_model.generate(bot_input_ids, max_length=1000, num_return_sequences=1)
    chatbot_response = chatbot_tokenizer.decode(chatbot_output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return chatbot_response
# Function to classify the revenue recognition standard using trained model

def classify_standard(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Use the ASC606 classifier model for classification
    predicted_class = twitter_classifier.predict([preprocessed_text])[0]
    return predicted_class


def chat():
    print("Twitter: Hello! How can I assist you today?")
    while True:
        user_input = input("User: ")

        if user_input.lower() == "end chat":
            print("Twitter : Goodbye! Have a great day!")
            break

        if "Twitter" in user_input.lower():
            if "compliance with Twitter" in user_input.lower():
                # Extract the text in quotes
                question = re.search(r'"([^"]*)"', user_input)
                if question:
                    text_to_classify = question.group(1)
                    # Preprocess the text
                    preprocessed_text = preprocess_text(text_to_classify)
                    vectorized_text = vectorizer.transform([preprocessed_text])
                    vectorized_text = vectorized_text.reshape(1, -1)
                    # Use the multinomial classification model
                    classification_result = twitter_classifier.predict(vectorized_text)
                    if classification_result == 1:
                        print("Twitter: Yes, it is in line with ASC606.")
                    else:
                        print("Twitter: No, it is not in line with ASC606.")
                else:
                    print("Twitter: I'm sorry, I couldn't find a valid question in your input.")
            else:
                print("Twitter: How can I assist you with ASC606?")
        else:
            response = generate_response(user_input)
            print("Twitter:", response)

import logging

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

set_global_logging_level(logging.ERROR, ["transformers"])

# Start the chatbot
chat()

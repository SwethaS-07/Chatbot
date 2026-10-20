import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.chat.util import Chat, reflections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

import json
import re
import random
import string



# Data preprocessing
class TextPreprocessing:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text):
        return word_tokenize(text)

    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stopwords]

    def preprocess(self, text):
        text = self.remove_punctuation(text)
        text = text.lower()
        tokens = self.tokenize(text)
        tokens = self.lemmatize(tokens)
        tokens = self.remove_stopwords(tokens)
        return " ".join(tokens)  # Assumed return

class AIPoweredChatBot:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.pipeline = make_pipeline(self.vectorizer,self.model)
        self.preprocessor = TextPreprocessing()
        self.train_data()

    def train_data(self):
        training_sentences = [
            "hello", "hi", "hey", "goodbye", "bye", "how are you", "I am fine", "thanks", "thank you",
            "what is your name", "who created you?", "tell me a joke", "I love you"
        ]

        # FIX: Clean the training sentences exactly like the live input
        cleaned_training_sentences = [
            self.preprocessor.preprocess(sentence) for sentence in training_sentences
        ]

        responses = [
            "Hi there!", "Hi there!", "Hello!", "Goodbye!", "Bye!", "I'm doing great, thanks!", "That's good to hear!",
            "You're welcome!", "You're welcome!",
            "I'm a chatbot!", "I was created by a developer!",
            "Why don't scientists trust atoms? Because they make up everything!", "I love you too!"
        ]

        self.pipeline.fit(cleaned_training_sentences,responses)
    def chatbot_response(self,user_input):
        return self.pipeline.predict([user_input])[0]



# print("Data processing Example")
# sample_text ="hello ,how are you doing today?. I am reading a book."
# preprocessor = TextPreprocessing()
# print("Original Text:",sample_text)
# print("Preprocessing Text",preprocessor.preprocess(sample_text))
#
# print("\nIntermediary Steps")
# example_text = preprocessor.remove_punctuation(sample_text)
# print("Remove Punctuation: ", example_text)
#
# # 2. Tokenize
# print("Tokenize: ", preprocessor.tokenize(example_text))
#
# # 3. Lemmatize (Passes tokens directly as seen in line 48)
# raw_tokens = preprocessor.tokenize(example_text)
# print("Lemmatize: ", preprocessor.lemmatize(raw_tokens))
#
# # 4. Remove Stopwords (Passes tokens directly as seen in line 49)
# print("Remove Stopwords: ", preprocessor.remove_stopwords(preprocessor.lemmatize(raw_tokens)))


#basic rule based chatbot
def chatbot_response(user_input):
    user_input = user_input.lower()
    if "hello" in user_input.lower() or "hi" in user_input.lower():
        return "Hello ,How are you?"
    elif "how are you" in user_input.lower():
        return "I'm just a chatbot. I don't have any feeling"
    elif "your name" in user_input.lower():
        return "I'm a chatbot. what can i do for you?"
    elif "bye" in user_input.lower():
        return "Goodbye! Have a great day"
    else:
        return "I'm sorry, I don't understand your questions"

pairs = [
    [
        r"hi|hello|hey",
        ["Hello, How can I help you?", "Hey there! What's up?", "Hi! how's it going?"]
    ],
    [
        r"what is your name?",
        ["I am a simple chatbot created using Python and NLTK."]
    ],
    [
        r"how are you?",
        ["I'm just a bot, but I'm doing great! How about you?", "I'm good! Thanks for asking."]
    ],
    [
        r"(.*) your name?",
        ["My name is ChatBot. What's yours?", "I am ChatBot, your virtual assistant."]
    ],
    [
        r"(.*) help (.*)",
        ["I can help you with general queries. What do you need help with?", "Sure! Let me know your question."]
    ],
    [
        r"bye|goodbye",
        ["Goodbye! Have a great day!", "See you later! Take care."]
    ],
    [
        r"(.*)",
        ["I'm not sure how to respond to that. Could you rephrase?", "That's interesting! Tell me more."]
    ]
]
chatbot = Chat(pairs,reflections)
print("Basic Rule Based ChatBot")
preprocessor = TextPreprocessing()
ai_bot = AIPoweredChatBot()
print("\n")
print("ChatBot: Hello, How can I help you?")
while True:
    user_input = input("user:").lower()
    preprocessed_text = preprocessor.preprocess(user_input)
    if user_input.lower() == "exit":
        print("ChatBot: GoodBye! Have a great day!")
        break
    # response = chatbot.respond(preprocessed_text) #rule based using nltk chat
    # response = chatbot_response(preprocessed_text) # basic rule based chatbot
    response = ai_bot.chatbot_response(preprocessed_text)  #AI Powered Chatbot
    print("ChatBot:", response)

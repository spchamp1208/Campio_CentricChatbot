import os
import openai
import tkinter as tk
import nltk
import pdfplumber
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.chat.util import Chat, reflections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from random import choice
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key is missing. Please add it to the .env file.")

openai.api_key = api_key

def get_response_from_openai(user_input: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using a chat model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except openai.error.AuthenticationError:
        return "Failed to authenticate with OpenAI API. Check your API key."
    except openai.error.InvalidRequestError as e:
        return f"Invalid request error: {e}"
    except Exception as e:
        return f"Error: {str(e)}"

def setup_chatbot() -> None:
    global chat, vectorizer, classifier, policy_documents
    pairs = [
        [r"my name is (.*)", ["Hello %1, How can I help you today?"]],
        [r"what is your name ?", ["I am a chatbot created for Centric."]],
        [r"how are you ?", ["I'm doing good, How about you?"]],
    ]
    chat = Chat(pairs, reflections)
    vectorizer = TfidfVectorizer()
    classifier = LogisticRegression()
    policy_documents = read_policy_documents()
    train_model()

def read_policy_documents() -> List[str]:
    policy_directory = r'C:\Users\sagar.patel\centric_chatbot_env\policy_documents'
    policy_texts = []

    for filename in os.listdir(policy_directory):
        file_path = os.path.join(policy_directory, filename)
        if filename.endswith('.pdf'):
            pdf_text = read_pdf(file_path)
            policy_texts.append(pdf_text)

    return policy_texts

def read_pdf(file_path: str) -> str:
    text = ''
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def preprocess_input(user_input: str) -> str:
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(user_input)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

def find_most_similar_document(user_input: str, documents: List[str]) -> int:
    tfidf_matrix = vectorizer.fit_transform(documents + [user_input])
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    most_similar_index = similarity_scores.argmax()
    return most_similar_index

def get_document_response(user_input: str, documents: List[str]) -> str:
    preprocessed_input = preprocess_input(user_input)
    most_similar_index = find_most_similar_document(preprocessed_input, documents)
    return documents[most_similar_index]

def check_alerts() -> List[str]:
    alerts = [
        "You have pending system updates.",
        "Your system password will expire in 5 days."
    ]
    return alerts

def get_hr_response(user_input: str) -> Optional[str]:
    if "leave policy" in user_input.lower():
        return "Our leave policy includes 20 days of annual leave."
    return None

wellness_prompts = [
    "Did you learn something new today?",
    "How is your mood today?",
]

def get_response(user_input: str) -> str:
    if "pdf" in user_input.lower() or "document" in user_input.lower():
        return get_document_response(user_input, policy_documents)
    elif "alerts" in user_input.lower():
        return "\n".join(check_alerts())
    elif "how are you" in user_input.lower():
        return choice(wellness_prompts)
    else:
        hr_response = get_hr_response(user_input)
        if hr_response:
            return hr_response
        else:
            return get_response_from_openai(user_input)

def train_model() -> None:
    X_train = vectorizer.fit_transform(["hello", "hi", "how are you"])
    y_train = ["greeting", "greeting", "wellness"]
    classifier.fit(X_train, y_train)

def send_message(event=None) -> None:
    user_input = user_entry.get()
    if user_input:
        bot_response = get_response(user_input)
        chat_box.config(state=tk.NORMAL)
        chat_box.insert(tk.END, f"You: {user_input}\n")
        chat_box.insert(tk.END, f"Bot: {bot_response}\n")
        chat_box.config(state=tk.DISABLED)
        user_entry.delete(0, tk.END)

def create_gui() -> None:
    global root, chat_box, user_entry

    root = tk.Tk()
    root.title("Centric Chatbot")

    def greet_and_ask() -> None:
        chat_box.config(state=tk.NORMAL)
        chat_box.insert(tk.END, "Bot: Hello! How may I assist you today?\n")
        chat_box.config(state=tk.DISABLED)

    header_frame = tk.Frame(root, bg='#2c1a5d', height=50)
    header_frame.pack(fill=tk.X)
    header_label = tk.Label(header_frame, text="Centric Chatbot", fg='white', bg='#2c1a5d', font=('Helvetica', 16, 'bold'))
    header_label.pack(fill=tk.X, pady=10)

    chat_frame = tk.Frame(root, bg='white')
    chat_frame.pack(expand=True, fill=tk.BOTH)
    chat_box = tk.Text(chat_frame, height=20, width=50, bg='white', fg='#2c1a5d')
    chat_box.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    chat_box.config(state=tk.DISABLED)

    footer_frame = tk.Frame(root, bg='#fdb825', height=50)
    footer_frame.pack(fill=tk.X)
    user_entry = tk.Entry(footer_frame, width=50, bg='#fdb825', fg='#2c1a5d', font=('Helvetica', 12))
    user_entry.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.X)
    send_button = tk.Button(footer_frame, text="Send", command=send_message)
    send_button.pack(side=tk.RIGHT, padx=10, pady=10)
    root.bind('<Return>', send_message)

    greet_and_ask()
    root.mainloop()

# Initialize the chatbot before starting the GUI
setup_chatbot()

# Call the create_gui function to start the GUI
create_gui()

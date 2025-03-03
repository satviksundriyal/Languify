from flask import Flask, render_template, request, jsonify
import pickle
import os
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("GEMINI_API_KEY")
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Load trained model and TF-IDF vectorizer for plagiarism detection
model_path = os.path.join(os.getcwd(), 'model.pkl')
tfidf_path = os.path.join(os.getcwd(), 'tfidf_vectorizer.pkl')

model_plagiarism = pickle.load(open(model_path, 'rb'))
tfidf_vectorizer = pickle.load(open(tfidf_path, 'rb'))

def detect_plagiarism(text):
    """Detect plagiarism in the input text."""
    vectorized_text = tfidf_vectorizer.transform([text])
    result = model_plagiarism.predict(vectorized_text)
    return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism Detected"

def correct_grammar(text):
    """Correct grammar using Gemini API."""
    response = model.generate_content(f"Correct this sentence: {text}")
    return response.text if response else text

def translate_text(text, target_language):
    """Translate text using Gemini API."""
    response = model.generate_content(f"Translate this text to {target_language}: {text}")
    return response.text if response else text

def paraphrase_text(text):
    """Paraphrase text using Gemini API."""
    response = model.generate_content(f"Paraphrase this text: {text}")
    return response.text if response else text

def summarize_text(text):
    """Summarize text using Gemini API."""
    response = model.generate_content(f"Summarize this text: {text}")
    return response.text if response else text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    input_text = request.json['text']
    result = detect_plagiarism(input_text)
    return jsonify({"result": result})

@app.route('/correct', methods=['POST'])
def correct():
    input_text = request.json['text']
    corrected_text = correct_grammar(input_text)
    return jsonify({"corrected_text": corrected_text})

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    input_text = data['text']
    target_language = data['language']  # Example: "French", "Hindi", "Spanish"
    translated_text = translate_text(input_text, target_language)
    return jsonify({"translated_text": translated_text})

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    input_text = request.json['text']
    paraphrased_text = paraphrase_text(input_text)
    return jsonify({"paraphrased_text": paraphrased_text})

@app.route('/summarize', methods=['POST'])
def summarize():
    input_text = request.json['text']
    summarized_text = summarize_text(input_text)
    return jsonify({"summarized_text": summarized_text})

if __name__ == "__main__":
    app.run(debug=True) 
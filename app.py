
import os
import dotenv
from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from flask_cors import CORS

# Load environment variables
dotenv.load_dotenv()

# Fetch API Key and Gemini Model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the AI model
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# Define Movie Recommendation Prompt Template
TEMPLATE = """You are a global movie recommendation assistant. 
Based on the user's preferences, suggest movies from different 
countries, languages, and genres.

User’s Movie Preferences:
{input}

Instructions:
1. Recommend **3 movies** based on genre, mood, and region.
2. Include **Title, Year, Country, Genre, and Short Plot Summary**.
3. Suggest **where to watch the movies (Netflix, Prime, Disney+, etc.)**.

Movies:
"""

prompt = PromptTemplate.from_template(TEMPLATE)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for external access

@app.route("/", methods=["GET"])
def home():
    return "Movie Recommendation AI is Running! Use /recommend to get movie suggestions."

@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.json.get("query", "")
    
    if not user_input:
        return jsonify({"error": "Please provide a movie preference."}), 400

    # Generate AI-based movie recommendations
    formatted_prompt = prompt.format(input=user_input)
    response = llm.invoke(formatted_prompt)

    return jsonify({"movies": response.content})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

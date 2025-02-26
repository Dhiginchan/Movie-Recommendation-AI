
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
# Movie Recommendation Template
TEMPLATE = """
You are a movie recommendation assistant. Based on the user's preference, suggest movies.
User's preference: {input}

Movies:
"""
prompt = PromptTemplate.from_template(TEMPLATE)

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.json.get("query", "")
    
    if not user_input:
        return jsonify({"error": "Please provide a movie preference."}), 400

    try:
        # Format prompt with user input
        formatted_prompt = prompt.format(input=user_input)
        response = llm.invoke(formatted_prompt)
        return jsonify({"movies": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# For local testing (use Gunicorn in production)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

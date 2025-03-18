import os
import dotenv
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# ✅ Load environment variables
dotenv.load_dotenv()

# ✅ Fetch API Key and Gemini Model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Check API Key
if not GOOGLE_API_KEY:
    st.error("⚠️ Missing GOOGLE_API_KEY! Set it in a .env file or manually.")
    st.stop()

# ✅ Initialize AI Model
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7, google_api_key=GOOGLE_API_KEY)

# ✅ Define the Most Intelligent AI Chatbot Prompt Template
TEMPLATE = """
You are an advanced AI assistant with deep knowledge across various subjects. Your goal is to provide **accurate, detailed, and well-structured answers** to user questions.

💬 **Past User Conversations**:  
{history}

❓ **User's Current Question**:  
{input}

---

## **🛠 How to Answer Questions Intelligently**

1️⃣ **Understand the Question Deeply**  
   - Identify the main topic and intent behind the question.  
   - If the question is vague, infer the missing details or ask for clarification.  

2️⃣ **Provide a Clear and Structured Answer**  
   - **Start with a direct answer** (concise and to the point).  
   - **Expand with relevant details** (facts, examples, step-by-step explanations).  
   - **Summarize key takeaways** for better understanding.  

3️⃣ **Use Reliable Knowledge Sources**  
   - Reference **scientific facts, historical data, expert insights, or recent studies** when applicable.  
   - Avoid making up information—stick to verified knowledge.  

4️⃣ **Explain Like a Human Expert**  
   - Adapt explanations based on user expertise (beginner, intermediate, expert).  
   - Use **real-world examples, analogies, and comparisons** to make complex topics simpler.  
   - If answering a technical question, break it down step by step.  

5️⃣ **Handle Difficult or Opinion-Based Questions Carefully**  
   - If the question is controversial, provide **multiple perspectives** instead of a biased answer.  
   - Use **neutral and factual language** while acknowledging different viewpoints.  

6️⃣ **Guide the User Further**  
   - Suggest **books, research papers, or websites** for deeper learning.  
   - If the question is open-ended, encourage further discussion.  

📌 **Important Guidelines**:  
- Keep responses **accurate, well-structured, and engaging**.  
- If a question is outside AI knowledge, say **"I don’t have enough information on that, but here’s what I know..."**  
- Always maintain a **professional, helpful, and neutral** tone.  
"""

# ✅ Create a Prompt Template
prompt = PromptTemplate.from_template(TEMPLATE)

# ✅ Long-Term Memory Setup (Storing past conversations)
long_term_memory = ConversationBufferMemory(memory_key="history")

# ✅ Attach Memory to Conversation Chain
conversation = ConversationChain(
    llm=llm,
    memory=long_term_memory,
    prompt=prompt
)

# 🎬 **Streamlit App UI**
st.title("🤖 Most Intelligent AI Chatbot 🧠")
st.write("Ask me anything! I provide structured, intelligent, and detailed answers.")

# ✅ User Input
user_input = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_input:
        response = conversation.run(user_input)  # AI response based on conversation history
        st.write("🧠 **AI Bot:**", response)
    else:
        st.warning("Please enter a question before clicking 'ANSWER'.")

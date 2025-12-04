from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from memory import retrieve_memory
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env! Get free key at https://console.groq.com/keys")

print("Using Groq Llama 3.1 70B – up to 800+ tokens/sec!")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",       
    api_key=GROQ_API_KEY,
    temperature=0.7,
    max_tokens=512,
)

prompt_template = """
You are a friendly, natural, concise voice assistant.
Keep replies short and conversational (1-3 sentences max).

{memories}
User: {user_text}
Assistant:
"""

chat_prompt = ChatPromptTemplate.from_template(prompt_template)
chain = chat_prompt | llm

def generate_reply(user_text: str) -> str:
    user_text = user_text.strip()
    if not user_text:
        return "Sorry, I didn't catch that."

    try:
        memories = retrieve_memory(user_text, k=3)
        response = chain.invoke({"memories": memories, "user_text": user_text})
        return response.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return "Hmm, I'm thinking... try again!"
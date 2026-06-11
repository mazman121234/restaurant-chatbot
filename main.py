from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    with open("public/index.html", "r") as f:
        return f.read()

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a helpful restaurant assistant chatbot. Your role is to help customers with:
- Questions about our menu and food offerings
- Restaurant hours and location
- Reservation and booking inquiries
- Pricing and specials
- Dietary restrictions and allergen information
- General restaurant-related questions

If someone asks a question that is NOT about the restaurant or our services, politely respond with:
"I'm here to help with questions about our restaurant, menu, hours, and reservations. How can I assist you with that?"

Keep responses friendly, concise, and professional. Do NOT answer general knowledge questions like math, trivia, or topics unrelated to the restaurant."""},
                {"role": "user", "content": request.message}
            ],
            max_tokens=500
        )
        
        bot_response = response.choices[0].message.content
        
        return ChatResponse(response=bot_response)
    
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok"}
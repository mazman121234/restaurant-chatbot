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

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list = []

class ChatResponse(BaseModel):
    response: str

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    try:
        with open("public/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: HTML file not found</h1>"
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>"

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        message = request.message.strip()
        
        if not message or len(message) == 0:
            return ChatResponse(response="Hi! How can I help you today?")
        
        if len(message) > 1000:
            return ChatResponse(response="Message too long. Please keep it under 1000 characters.")
        
        messages = [
            {"role": "system", "content": """You are a helpful restaurant assistant. You are in the middle of helping a customer make a reservation.

The customer just responded to a question. Do NOT restart the conversation. Continue naturally from where you left off.

If you asked "how many people" and they said "4", respond: "Perfect! Party of 4. What time would you like to book?"

If you asked "what time" and they said "6pm", respond: "Great! 6pm works. Can I get a name for the reservation?"

CONTINUE THE CONVERSATION. Do not reset. Do not show a greeting. Just continue helping them book.

Accept any input as part of their booking response."""}
        ]
        
        if request.history:
            messages.extend(request.history)
        
        messages.append({"role": "user", "content": message})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.8
        )
        
        bot_response = response.choices[0].message.content
        
        if not bot_response or len(bot_response.strip()) == 0:
            return ChatResponse(response="I'm here to help! What would you like to know?")
        
        return ChatResponse(response=bot_response.strip())
    
    except ValueError as e:
        return ChatResponse(response="Let me help you with that!")
    except Exception as e:
        return ChatResponse(response="Sorry, I'm temporarily unavailable. Please try again in a moment.")

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0"}
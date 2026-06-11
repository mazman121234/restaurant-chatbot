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
            return ChatResponse(response="Please enter a message.")
        
        if len(message) > 1000:
            return ChatResponse(response="Message too long. Please keep it under 1000 characters.")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a helpful restaurant assistant. Be friendly and helpful. Respond to anything the customer says naturally and conversationally.

If they ask about: menu, food, hours, location, reservations, pricing, dietary needs - answer helpfully.

If they say something that might be part of a booking (names, numbers, times, dates, party sizes) - acknowledge it and help them complete their booking.

If they ask something completely unrelated (like math or current events), gently say: "I'm here to help with restaurant questions. What can I help you with?"

Always be conversational and friendly. Never be strict or reject inputs."""},
                {"role": "user", "content": message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        bot_response = response.choices[0].message.content
        
        if not bot_response or len(bot_response.strip()) == 0:
            return ChatResponse(response="I'm here to help! What would you like to know about our restaurant?")
        
        return ChatResponse(response=bot_response.strip())
    
    except ValueError as e:
        return ChatResponse(response="Invalid input. Please try again.")
    except Exception as e:
        return ChatResponse(response="Sorry, I'm temporarily unavailable. Please try again in a moment.")

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0"}
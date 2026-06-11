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
            return ChatResponse(response="Hi! How can I help you today?")
        
        if len(message) > 1000:
            return ChatResponse(response="Message too long. Please keep it under 1000 characters.")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a helpful, friendly restaurant assistant. Your job is to have natural conversations with customers about restaurant-related topics.

IMPORTANT: Accept and respond naturally to ANYTHING the customer types. Never reject input.

- If they ask about menu, food, hours, location, reservations, prices, dietary needs - answer helpfully about the restaurant.
- If they type a name (John, Sarah, etc) - acknowledge it as their name/booking name.
- If they type a number (4, 10, 5, etc) - treat it as a party size or table number. Example: "Got it, party of 4!" or "Table 5, great!"
- If they type a time (6pm, 19:00, etc) - acknowledge it as a booking time.
- If they type random words or letters - just work it into the conversation naturally.
- If they ask something unrelated (math, sports, jokes) - respond with "I'm here to help with restaurant questions!" but still be friendly.

NEVER say "I can only help with restaurant questions." NEVER reject their input. Just respond naturally and conversationally. Be warm, helpful, and flexible."""},
                {"role": "user", "content": message}
            ],
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
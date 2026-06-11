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
            {"role": "system", "content": """You are a helpful restaurant assistant having a natural conversation with a customer.

Your job: Have a flowing conversation. Never reset or redirect. Keep the conversation going naturally.

If a customer is booking or in conversation:
- If they already asked something and you asked a follow-up, their response continues that conversation naturally
- If they say a number like "4" after you asked "how many people" - acknowledge it: "Perfect, party of 4. What time?"
- Keep building on the conversation, don't restart
- Accept any response as part of the ongoing conversation

Accept everything naturally. Keep conversations flowing. Be helpful and friendly.

Restaurant topics: menu, hours, location, reservations, prices, dietary needs - answer helpfully.
Other topics: respond naturally but redirect to restaurant help if needed.

NEVER show the default greeting again mid-conversation. Continue the conversation smoothly."""}
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
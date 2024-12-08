from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from query_rag import run_mistral
from pydantic import BaseModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

class MessageRequest(BaseModel):
    message: str

@app.post('/api/chat/')
async def chat(message: MessageRequest):
    try:
        bot_response = run_mistral(message.message)
        return {"response":bot_response}
    except:
        return {"response":"Something went wront"}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )
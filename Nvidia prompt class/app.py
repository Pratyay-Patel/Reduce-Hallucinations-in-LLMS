from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import temp
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
print("Loading model...")
model, tokenizer = temp.load_model()
print("Model loaded.")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/analyze")
async def analyze(request: PromptRequest):
    try:
        result = temp.analyze_prompt(model, tokenizer, request.prompt)
        # Convert tensor/numpy values to native python types for JSON serialization if needed
        # The existing compute_results in temp.py seems to return lists/floats already,
        # but let's ensure it's safe.
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_index():
    return FileResponse('index.html')

# Serve other static files if needed, but for now just index.html is enough.
# If we had css/js files we would mount them.
# app.mount("/static", StaticFiles(directory="static"), name="static")

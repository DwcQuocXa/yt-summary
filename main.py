from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from youtube_summarizer import YouTubeSummarizer, SummaryType
from dotenv import load_dotenv

# Load environment variables at startup
load_dotenv()

app = FastAPI()
summarizer = YouTubeSummarizer()

class SummaryRequest(BaseModel):
    url: HttpUrl

@app.post("/summarize")
async def summarize_general(request: SummaryRequest):
    try:
        return await summarizer.summarize_video(str(request.url), "general")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/summarize/macro-economy")
async def summarize_macro_economy(request: SummaryRequest):
    try:
        return await summarizer.summarize_video(str(request.url), "macro_economy")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/summarize/crypto")
async def summarize_crypto(request: SummaryRequest):
    try:
        return await summarizer.summarize_video(str(request.url), "crypto")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
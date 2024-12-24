from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from app.youtube_summarizer.youtube_summarizer import YouTubeSummarizer, SummaryType
from dotenv import load_dotenv
from app.email_sender.scheduler import SummaryScheduler
from app.email_sender.email_service import EmailService

# Load environment variables at startup
load_dotenv()

app = FastAPI()
summarizer = YouTubeSummarizer()

# Initialize scheduler
scheduler = SummaryScheduler()
email_service = EmailService()

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

@app.on_event("startup")
async def start_scheduler():
    scheduler.start()

@app.post("/test-email")
async def test_email():
    """Endpoint to test email sending with current summaries"""
    try:
        await scheduler.send_daily_summaries()
        return {"message": "Test email sent successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
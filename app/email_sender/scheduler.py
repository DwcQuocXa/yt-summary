from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from app.email_sender.email_service import EmailService
from app.youtube_summarizer.youtube_summarizer import YouTubeSummarizer

class SummaryScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.email_service = EmailService()
        self.summarizer = YouTubeSummarizer()

    async def send_daily_summaries(self):
        summaries = []
        # Comment out the actual video processing for now
        for video in self.email_service.youtube_urls:
            try:
                summary = await self.summarizer.summarize_video(
                    video["url"], 
                    video["type"]
                )
                summaries.append(summary)
            except Exception as e:
                print(f"Failed to summarize video {video['url']}: {str(e)}")

        # Send email with all summaries
        if summaries:
            await self.email_service.send_summary_email(summaries)

    def start(self):
        # Schedule job to run daily at 8 AM
        self.scheduler.add_job(
            self.send_daily_summaries,
            CronTrigger(hour=8, minute=0)
        )
        self.scheduler.start() 
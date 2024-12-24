import os
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict
from datetime import datetime

class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.sender_password = os.getenv("SENDER_PASSWORD")
        
        # List of recipient emails
        self.recipient_emails = os.getenv("RECIPIENT_EMAILS", "").split(",")
        
        # List of YouTube URLs to summarize daily
        self.youtube_urls = [
            {"url": "https://www.youtube.com/watch?v=-5u3MEpOItc", "type": "macro_economy"},
        ]
        
    async def send_summary_email(self, summaries: List[Dict]):
        try:
            smtp = aiosmtplib.SMTP(
                hostname=self.smtp_server,
                port=465,
                use_tls=True,
            )
            
            await smtp.connect()
            await smtp.login(self.sender_email, self.sender_password)
            
            for recipient in self.recipient_emails:
                # Create a new message for each recipient
                msg = MIMEMultipart()
                msg['From'] = self.sender_email
                msg['To'] = recipient
                msg['Subject'] = f"Daily YouTube Summaries - {datetime.now().strftime('%Y-%m-%d')}"
                
                # Create email body
                email_body = self._create_email_body(summaries)
                msg.attach(MIMEText(email_body, 'html'))
                
                print(f"Sending email to {recipient}")
                await smtp.send_message(msg)
            
            await smtp.quit()
            return True
            
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            return False

    def _create_email_body(self, summaries: List[Dict]) -> str:
        html = """
        <html>
            <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #333;">Daily YouTube Summaries</h1>
                <p>Here are your daily video summaries:</p>
        """
        
        for summary in summaries:
            html += f"""
                <div style="margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
                    <h2 style="color: #2c5282;">
                        <a href="{summary['metadata']['url']}" 
                           style="color: #2b6cb0; text-decoration: none;">
                            {summary['metadata']['url']}
                        </a>
                    </h2>
                    <div style="line-height: 1.6;">{summary['summary']}</div>
                </div>
            """
            
        html += """
            </body>
        </html>
        """
        return html 

    async def send_test_email(self):
        mock_summaries = [
            {
                "metadata": {
                    "url": "https://www.youtube.com/watch?v=test123",
                    "summary_type": "general"
                },
                "summary": """
                <h3>Key Points:</h3>
                <ul>
                    <li>This is a test summary point 1</li>
                    <li>This is a test summary point 2</li>
                    <li>This is a test summary point 3</li>
                </ul>
                <h3>Detailed Summary:</h3>
                <p>This is a mock detailed summary of what would be a YouTube video. 
                It contains multiple paragraphs to simulate a real summary.</p>
                <p>This second paragraph provides additional context and details about the 
                video's content. It helps test how the email formatting handles longer text.</p>
                """
            },
            {
                "metadata": {
                    "url": "https://www.youtube.com/watch?v=test456",
                    "summary_type": "technical"
                },
                "summary": """
                <h3>Technical Overview:</h3>
                <ul>
                    <li>Technical point 1 about some code</li>
                    <li>Technical point 2 about implementation</li>
                    <li>Technical point 3 about best practices</li>
                </ul>
                <p>This is a mock technical summary that would contain code-related or 
                technical content details.</p>
                """
            }
        ]
        
        return await self.send_summary_email(mock_summaries) 
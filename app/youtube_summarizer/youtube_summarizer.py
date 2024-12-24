import os
import re
from typing import Dict, Any, Literal
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from app.youtube_summarizer.prompt import get_prompts

load_dotenv()

SummaryType = Literal["general", "macro_economy", "crypto"]

class YouTubeSummarizer:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key in .env file")
        
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model_name="o1-mini",
            temperature=1
        )
        
        self.prompts = get_prompts()

    async def get_video_id(self, url: str) -> str:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
            return parsed_url.path[1:]
        if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
            return parse_qs(parsed_url.query)['v'][0]
        raise ValueError("Invalid YouTube URL")

    def process_llm_summary(self, llm_summary: str, video_id: str) -> str:
        # First, handle the timestamp markers [XXs]
        timestamp_pattern = r'\[(\d+)s\]'
        def replace_timestamp(match):
            start_time = match.group(1)
            youtube_link = f"https://www.youtube.com/watch?v={video_id}&t={start_time}s"
            return f' <a href="{youtube_link}" target="_blank">[Link]</a>'
        
        processed_text = re.sub(timestamp_pattern, replace_timestamp, llm_summary)
        
        # Then remove any existing markdown-style links that might have been generated
        markdown_pattern = r'\[Link\]\(https://[^)]+\)'
        processed_text = re.sub(markdown_pattern, '[Link]', processed_text)
        
        return processed_text

    async def summarize_video(self, url: str, summary_type: SummaryType = "general") -> Dict[str, Any]:
        try:
            video_id = await self.get_video_id(url)
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['vi'])

            # Create formatted transcript with timestamps
            formatted_transcript = " ".join(
                f"[{int(entry['start'])}s] {entry['text']}"
                for entry in transcript_list
            )

            # Process the entire transcript at once
            prompt_template = self.prompts[summary_type]
            chain = prompt_template | self.llm
            
            summary = await chain.ainvoke({
                "text": formatted_transcript,
                "video_id": video_id,
                "video_url": url,
                "timestamp": "0"  # Default timestamp, will be replaced by actual timestamps in the text
            })
            
            processed_summary = self.process_llm_summary(summary.content, video_id)

            return {
                "summary": processed_summary,
                "metadata": {
                    "video_id": video_id,
                    "url": url,
                    "summary_type": summary_type,
                    "transcript_length": len(formatted_transcript)
                }
            }

        except Exception as e:
            raise Exception(f"Failed to process video: {str(e)}")
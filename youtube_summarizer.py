import os
import re
from typing import Dict, Any, Literal
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

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
        
        self._init_prompts()

    def _init_prompts(self):
        base_example = """
            Ví dụ về một điểm chi tiết tốt với định dạng markdown:
            - **Tác động của AI**: Diễn giả phân tích ảnh hưởng của `machine learning` đến thị trường lao động, nhấn mạnh việc *"47% công việc có nguy cơ bị tự động hóa trong 10 năm tới"*. Quan điểm này được minh họa qua việc dẫn chứng các nghiên cứu từ **Oxford Economics** và đưa ra các ví dụ cụ thể về ngành nghề chịu ảnh hưởng [205s]
        """

        base_format = """
            Vui lòng tạo bản tóm tắt theo cấu trúc markdown sau:

            ## Chủ đề chính 1
            - Chi tiết 1.1: **Phân tích** và giải thích cụ thể về nội dung, trong đó *"trích dẫn lời người nói"* [120s]
            - Chi tiết 1.2: Trích dẫn về `thuật ngữ chuyên môn` và phân tích [180s]
            
            ## Chủ đề chính 2
            - Chi tiết 2.1: Mô tả chi tiết về luận điểm được đề cập [250s]
            - Chi tiết 2.2: Giải thích và phân tích sâu về ý nghĩa [300s]
            
            LƯU Ý QUAN TRỌNG:
            1. PHẢI tóm tắt toàn bộ nội dung từ đầu đến cuối video. Không bỏ sót các phần quan trọng ở giữa hoặc cuối video
            2. Đảm bảo các trích dẫn thời gian được phân bố đều từ đầu đến cuối video
            3. Mỗi chủ đề chính (##) không cần có trích dẫn thời gian
            4. Tất cả các điểm chi tiết (-) PHẢI có trích dẫn thời gian
            5. Mỗi điểm chi tiết nên có độ dài tối thiểu 2-3 câu để đảm bảo đầy đủ thông tin
            6. Tập trung vào việc phân tích và giải thích, không chỉ đơn thuần tóm tắt
            7. Sử dụng ngôn ngữ chuyên nghiệp và mạch lạc
            8. Đảm bảo các trích dẫn thời gian phản ánh đúng nội dung được đề cập
        """

        self.prompts: Dict[str, PromptTemplate] = {
            "general": PromptTemplate(
                template=f"""Với vai trò là một chuyên gia phân tích nội dung, hãy tạo một bản tóm tắt chi tiết và có cấu trúc phân cấp từ nội dung video YouTube sau.

                {base_format}

                {base_example}

                Nội dung ghi chép:
                {{text}}

                Video ID: {{video_id}}
                """,
                input_variables=["text", "video_id"]
            ),
            
            "macro_economy": PromptTemplate(
                template=f"""Với vai trò là một chuyên gia phân tích kinh tế vĩ mô, hãy tạo một bản phân tích chuyên sâu từ nội dung video về tin tức kinh tế.

                {base_format}

                Yêu cầu phân tích bổ sung:
                1. Tập trung vào các chỉ số kinh tế quan trọng (`GDP`, `CPI`, `lạm phát`, `tỷ giá`)
                2. Phân tích tác động đến các ngành nghề và thị trường
                3. Đề xuất các xu hướng và dự báo
                4. Liên hệ với các sự kiện kinh tế toàn cầu
                5. Sử dụng Tiếng Việt cho toàn bộ nội dung với chủ đề chính là `kinh tế vĩ mô`

                Ví dụ về một điểm chi tiết tốt với định dạng markdown:
                - **Tác động lạm phát**: Chuyên gia phân tích chỉ số `CPI` tháng 3, nhấn mạnh *"mức tăng 5.6% so với cùng kỳ năm ngoái"*. Điều này tác động mạnh đến thị trường bất động sản khi **lãi suất vay** tiếp tục được dự báo tăng trong quý tới [180s]

                Nội dung ghi chép:
                {{text}}

                Video ID: {{video_id}}
                """,
                input_variables=["text", "video_id"]
            ),
            
            "crypto": PromptTemplate(
                template=f"""Với vai trò là một chuyên gia phân tích thị trường tiền số, hãy tạo một bản phân tích chuyên sâu từ nội dung video về crypto.

                {base_format}

                Yêu cầu phân tích bổ sung:
                1. Phân tích biến động giá và khối lượng giao dịch
                2. Đánh giá các yếu tố `fundamentals` và tin tức ảnh hưởng
                3. Phân tích `sentiment` thị trường
                4. Các thông tin về `regulatory` và `adoption`
                5. `Technical analysis` nếu có đề cập
                6. Sử dụng Tiếng Việt cho toàn bộ nội dung với chủ đề chính là `crypto`

                Ví dụ về một điểm chi tiết tốt với định dạng markdown:
                - **Phân tích Bitcoin**: Diễn giả đánh giá xu hướng `price action` của BTC, với *"khối lượng giao dịch tăng 40% trong 24h qua"*. Các chỉ báo kỹ thuật như `RSI` và `MACD` cho thấy tín hiệu **tích cực** trong ngắn hạn, đặc biệt khi xét đến tin tức về quỹ `ETF` mới được phê duyệt [245s]

                Nội dung ghi chép:
                {{text}}

                Video ID: {{video_id}}
                """,
                input_variables=["text", "video_id"]
            )
        }

    async def get_video_id(self, url: str) -> str:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
            return parsed_url.path[1:]
        if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
            return parse_qs(parsed_url.query)['v'][0]
        raise ValueError("Invalid YouTube URL")

    def process_llm_summary(self, llm_summary: str, video_id: str) -> str:
        pattern = r'\[(\d+)s\]'

        def replace_match(match):
            start_time = match.group(1)
            youtube_link = f"https://www.youtube.com/watch?v={video_id}&t={start_time}s"
            return f" [Link]({youtube_link})"

        return re.sub(pattern, replace_match, llm_summary)

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
                "video_id": video_id
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
from typing import Dict
from langchain.prompts import PromptTemplate

BASE_EXAMPLE = """
    Ví dụ về cách trình bày nội dung chi tiết với định dạng HTML:
    <div class="key-point">
        <h3>🔍 Phân tích chuyên sâu</h3>
        - <strong>Tác động của AI</strong>: Diễn giả phân tích ảnh hưởng của <code>machine learning</code> đến thị trường lao động, 
          nhấn mạnh việc <em>"47% công việc có nguy cơ bị tự động hóa trong 10 năm tới"</em>.
          <br />
          Quan điểm này được minh họa qua:
          <ul>
              <li>Nghiên cứu từ <strong>Oxford Economics</strong></li>
              <li>Các ví dụ cụ thể về ngành nghề chịu ảnh hưởng</li>
              <li>Dữ liệu thống kê từ các thị trường lớn</li>
          </ul> [205s]
    </div>
"""

BASE_FORMAT = """
    Vui lòng tạo bản tóm tắt theo cấu trúc HTML sau. LƯU Ý: Không sử dụng dấu ``` để bao quanh phần HTML trong câu trả lời:

    <!-- Đây chỉ là ví dụ về cấu trúc. Bạn cần tạo nhiều chủ đề chính (main-topic) hơn dựa trên nội dung thực tế của video -->
    <article class="summary">
        <div class="main-topic">
            <h2>🎯 Chủ đề chính (Ví dụ)</h2>
            <div class="details">
                - <strong>Phân tích chuyên sâu</strong>: Phân tích với <code>thuật ngữ</code> và <em>"trích dẫn quan trọng"</em> <a href="{video_url}&t={timestamp}" target="_blank">[Link]</a>.
                <br />
                  Thêm giải thích chi tiết ở đây. <a href="{video_url}&t=120" target="_blank">[Link]</a>
                <br />
                - <strong>Giải thích và dẫn chứng</strong>: Nội dung phân tích <a href="{video_url}&t=180" target="_blank">[Link]</a>
            </div>
        </div>
    </article>

    HƯỚNG DẪN ĐỊNH DẠNG:
    1. Trả lời trực tiếp bằng HTML, không sử dụng markdown code blocks
    2. Sử dụng thẻ <article> để bao quát toàn bộ nội dung
    3. Các điểm chi tiết nằm trong <div class="details">
    4. Sử dụng <br /> cho xuống dòng thay vì "\\n"
    5. Đảm bảo các thẻ HTML được đóng mở đúng cách

    LƯU Ý QUAN TRỌNG:
    1. PHẢI tạo nhiều chủ đề chính (main-topic) dựa trên nội dung thực tế của video, KHÔNG giới hạn số lượng
    2. Mỗi chủ đề chính cần có ít nhất 3-4 điểm chi tiết để đảm bảo độ sâu của phân tích
    3. PHẢI tóm tắt toàn bộ nội dung từ đầu đến cuối video. Không bỏ sót các phần quan trọng
    4. Đảm bảo các trích dẫn thời gian được phân bố đều từ đầu đến cuối video
    5. Mỗi chủ đề chính h2 không cần có trích dẫn thời gian
    6. Tất cả các điểm chi tiết (-) PHẢI có trích dẫn thời gian
    7. Mỗi điểm chi tiết nên có độ dài tối thiểu 2-3 câu để đảm bảo đầy đủ thông tin
    8. Tập trung vào việc phân tích và giải thích, không chỉ đơn thuần tóm tắt
    9. Sử dụng ngôn ngữ chuyên nghiệp và mạch lạc
    10. Đảm bảo các trích dẫn thời gian phản ánh đúng nội dung được đề cập
    11. Sử dụng biểu tượng (emoji) phù hợp với nội dung của từng chủ đề chính. Ví dụ:
        - 📊 cho phân tích dữ liệu/thống kê
        - 💡 cho ý tưởng/giải pháp
        - 🔍 cho nghiên cứu/phân tích chuyên sâu
        - 📈 cho xu hướng/tăng trưởng
        - 🎯 cho mục tiêu/chiến lược
        - 🔄 cho quy trình/chu trình
        - 💰 cho tài chính/đầu tư
        - 🌐 cho xu hướng toàn cầu
        - ⚠️ cho cảnh báo/rủi ro
        - ✅ cho kết luận/khuyến nghị
"""

def get_prompts() -> Dict[str, PromptTemplate]:
    return {
        "general": PromptTemplate(
            template=f"""Với vai trò là một chuyên gia phân tích nội dung, hãy tạo một bản tóm tắt chi tiết và có cấu trúc phân cấp từ nội dung video YouTube sau.

            {BASE_FORMAT}

            {BASE_EXAMPLE}

            Nội dung ghi chép:
            {{text}}

            Video ID: {{video_id}}
            Video URL: {{video_url}}
            Timestamp: {{timestamp}}
            """,
            input_variables=["text", "video_id", "video_url", "timestamp"]
        ),
        
        "macro_economy": PromptTemplate(
            template=f"""Với vai trò là một chuyên gia phân tích kinh tế vĩ mô, hãy tạo một bản phân tích chuyên sâu từ nội dung video về tin tức kinh tế.

            {BASE_FORMAT}

            Yêu cầu phân tích bổ sung:
            1. Tập trung vào các chỉ số kinh tế quan trọng (`GDP`, `CPI`, `lạm phát`, `tỷ giá`)
            2. Phân tích tác động đến các ngành nghề và thị trường
            3. Đề xuất các xu hướng và dự báo
            4. Liên hệ với các sự kiện kinh tế toàn cầu
            5. Sử dụng Tiếng Việt cho toàn bộ nội dung với chủ đề chính là `kinh tế vĩ mô`

            Ví dụ về một điểm chi tiết tốt với định dạng HTML:
            - <strong>Tác động lạm phát</strong>: Chuyên gia phân tích chỉ số <code>CPI</code> tháng 3, nhấn mạnh <em>"mức tăng 5.6% so với cùng kỳ năm ngoái"</em>. Điều này tác động mạnh đến thị trường bất động sản khi <strong>lãi suất vay</strong> tiếp tục được dự báo tăng trong quý tới [180s]

            Nội dung ghi chép:
            {{text}}

            Video ID: {{video_id}}
            Video URL: {{video_url}}
            Timestamp: {{timestamp}}
            """,
            input_variables=["text", "video_id", "video_url", "timestamp"]
        ),
        
        "crypto": PromptTemplate(
            template=f"""Với vai trò là một chuyên gia phân tích thị trường tiền số, hãy tạo một bản phân tích chuyên sâu từ nội dung video về crypto.

            {BASE_FORMAT}

            Yêu cầu phân tích bổ sung:
            1. Phân tích biến động giá và khối lượng giao dịch
            2. Đánh giá các yếu tố `fundamentals` và tin tức ảnh hưởng
            3. Phân tích `sentiment` thị trường
            4. Các thông tin về `regulatory` và `adoption`
            5. `Technical analysis` nếu có đề cập
            6. Sử dụng Tiếng Việt cho toàn bộ nội dung với chủ đề chính là `crypto`

            Ví dụ về một điểm chi tiết tốt với định dạng HTML:
                - <strong>Phân tích Bitcoin</strong>: Diễn giả đánh giá xu hướng <code>price action</code> của BTC, với <em>"khối lượng giao dịch tăng 40% trong 24h qua"</em>. 
                  Các chỉ báo kỹ thuật như <code>RSI</code> và <code>MACD</code> cho thấy tín hiệu <strong>tích cực</strong> trong ngắn hạn, 
                  đặc biệt khi xét đến tin tức về quỹ <code>ETF</code> mới được phê duyệt [245s]
            Nội dung ghi chép:
            {{text}}

            Video ID: {{video_id}}
            Video URL: {{video_url}}
            Timestamp: {{timestamp}}
            """,
            input_variables=["text", "video_id", "video_url", "timestamp"]
        )
    } 
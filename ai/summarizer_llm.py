# summarizer_llm.py
import os
from openai import OpenAI

# 환경 변수에서 OpenAI API Key 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY 환경 변수가 없습니다. .env 파일을 확인하세요.")

client = OpenAI(api_key=OPENAI_API_KEY)


class LLMSummarizer:
    def __init__(
        self,
        model="gpt-4o-mini",
        max_summary_len=60,
        chunk_threshold=300,
    ):
        """
        LLM 기반 리뷰 요약기.
        model: 사용할 OpenAI 모델
        max_summary_len: 생성할 요약 길이 제한
        chunk_threshold: 리뷰 길이가 너무 길 경우 chunk로 나누기 위한 기준 길이
        """
        self.model = model
        self.max_summary_len = max_summary_len
        self.chunk_threshold = chunk_threshold

    def _summarize_chunk(self, text: str) -> str:
        """
        LLM 한 번 호출로 요약 수행.
        chunk 단위 요약만 담당.
        """

        prompt = f"""
당신은 한국어 리뷰 요약 전문가입니다.

리뷰를 본문 의미를 유지하면서 매끄럽고 자연스럽게 한 문장으로 요약하세요.
- 감정(긍/부정)을 자연스럽게 포함
- 핵심 불만 또는 칭찬을 명확히 표현
- 반복 금지
- 지나친 생략 금지
- 리뷰 쓰는 말투 유지

리뷰:
{text}

요약:"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_summary_len,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"[LLM Error: {e}]"

    def summarize(self, text: str) -> str:
        """
        최종 요약 함수.
        - 짧은 리뷰는 그대로 사용
        - 긴 리뷰는 chunk → partial summary → final summary 구조
        """
        if not text or len(text.strip()) == 0:
            return ""

        # 너무 짧은 리뷰는 요약 불필요
        if len(text.split()) < 5:
            return text.strip()

        # 리뷰 길이가 짧으면 바로 요약
        if len(text) <= self.chunk_threshold:
            return self._summarize_chunk(text)

        # -------------------------
        # 긴 리뷰 chunk 나눠서 처리
        # -------------------------
        sentences = text.replace("?", ".").replace("!", ".").split(".")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

        chunks = []
        chunk = ""
        for s in sentences:
            if len(chunk) + len(s) < self.chunk_threshold:
                chunk += s + ". "
            else:
                chunks.append(chunk.strip())
                chunk = s + ". "
        if chunk:
            chunks.append(chunk.strip())

        # 각 chunk 요약
        partial_summaries = [self._summarize_chunk(c) for c in chunks]

        # partial summary 전체를 다시 요약해서 최종 요약 생성
        final_text = " ".join(partial_summaries)
        final_summary = self._summarize_chunk(final_text)

        return final_summary.strip()


# 테스트 코드
if __name__ == "__main__":
    s = LLMSummarizer()

    review = """
음식은 정말 맛있었습니다. 하지만 배달이 1시간 넘게 걸려 너무 늦었고 
양도 생각보다 적어서 실망스러웠습니다. 사장님은 친절하게 응대해주셨지만 
배달 시간만 좀 빨랐더라면 훨씬 좋았을 것 같습니다.
"""

    print("요약 결과:")
    print(s.summarize(review))
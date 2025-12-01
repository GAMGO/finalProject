from openai import OpenAI
client = OpenAI()

from app.utils.chunk import chunk_reviews

# 1) 청크 요약
def summarize_chunk(text: str):
    prompt = f"""
    아래 리뷰들을 핵심만 요약해줘.\n{text}
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content     # 수정됨
                                               # ["content"] → .content

# 2) 최종 종합 요약
def summarize_reviews(reviews):
    chunks = chunk_reviews(reviews)
    mids = [summarize_chunk(c) for c in chunks]

    final_prompt = f"""
    아래는 청크별 요약이야.
    이를 기반으로 최종 종합 요약을 1줄요약해줘.

    {chr(10).join(mids)}
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}]
    )

    return res.choices[0].message.content  
def chunk_reviews(reviews, chunk_size=10):
    """
    리뷰 리스트를 일정 크기로 청크 분리.
    Review 객체 기준.
    """
    chunks = []
    current = []

    for r in reviews:
        review_text = (r.review_text or "").strip()

        # 중요: rating, sentiment_score, sentiment_label 포함
        current.append(
            f"- rating: {r.rating}, sentiment_score: {r.sentiment_score}, "
            f"sentiment_label: {r.sentiment_label}, text: \"{review_text}\""
        )

        if len(current) >= chunk_size:
            chunks.append("\n".join(current))
            current = []

    if current:
        chunks.append("\n".join(current))

    return chunks
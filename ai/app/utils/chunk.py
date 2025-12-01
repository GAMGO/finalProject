def chunk_reviews(reviews, max_chars=2000):
    """
    reviews: [{"review_text": "...", "rating": 5}, ...]
    """
    chunks = []
    cur = ""

    for r in reviews:
        text = (r.get("review_text") or "").strip()

        if len(cur) + len(text) > max_chars:
            if cur:
                chunks.append(cur)
            cur = text
        else:
            cur += "\n" + text

    if cur:
        chunks.append(cur)

    return chunks
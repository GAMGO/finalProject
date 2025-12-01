# ai/sentiment_batch_reviews.py
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../finalProject/ai
ROOT_DIR = os.path.dirname(CURRENT_DIR)                    # .../finalProject
sys.path.append(ROOT_DIR)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from app.config.database import get_db_connection

MODEL_NAME = "nlp04/korean_sentiment_analysis_kcelectra"

# 감정 라벨 → 긍/부정 매핑
POSITIVE_LABELS = {
    "기쁨(행복한)",
    "고마운",
    "설레는(기대하는)",
    "사랑하는",
    "즐거운(신나는)",
}

NEUTRAL_LABELS = {
    "일상적인",
    "생각이 많은",
}

NEGATIVE_LABELS = {
    "슬픔(우울한)",
    "힘듦(지침)",
    "짜증남",
    "걱정스러운(불안한)",
}


def get_reviews_without_sentiment():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT idx, store_idx, review_text
                FROM store_reviews
                WHERE is_blocked = 0
                  AND review_text IS NOT NULL
                  AND review_text != ''
                  AND sentiment_score IS NULL
                ORDER BY idx ASC
            """
            )
            return cur.fetchall()
    finally:
        conn.close()


def update_review_sentiment(idx, score, label):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE store_reviews
                SET sentiment_score = %s,
                    sentiment_label = %s
                WHERE idx = %s
            """,
                (score, label, idx),
            )
        conn.commit()
    finally:
        conn.close()


def run_sentiment_batch():
    print("🔹 감성 분석 대상 리뷰 로드...")
    rows = get_reviews_without_sentiment()
    if not rows:
        print("⭐ 새로운 리뷰 없음")
        return

    print(f"📌 {len(rows)}개 리뷰 처리 예정")

    print("🔄 KcELECTRA 감성 모델 로드...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,use_safetensors=True,trust_remote_code=True)
    print("DEBUG:", model.config.id2label)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    for idx, store_idx, text in rows:
        # 길이 제한 + truncation 옵션은 호출할 때 넣기
        res = clf(text, truncation=True, max_length=128)[0]

        label = res["label"]          # 예: "기쁨(행복한)"
        prob = float(res["score"])    # softmax 확률 (0~1)

        # 감정 라벨을 [-1, 1] 스코어로 변환
        if label in POSITIVE_LABELS:
            base = 1.0
        elif label in NEGATIVE_LABELS:
            base = -1.0
        else:  # NEUTRAL_LABELS 또는 기타 미정 라벨
            base = 0.0

        # 확률을 곱해서 강도 조절 (원하면 다른 스케일 써도 됨)
        score = base * prob

        update_review_sentiment(idx, score, label)
        print(f"📝 리뷰 {idx} → label={label}, prob={prob:.3f}, score={score:.3f}")

    print("🎉 감성 분석 배치 완료")


if __name__ == "__main__":
    run_sentiment_batch()
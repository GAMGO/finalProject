# ai/sentiment_batch_reviews.py (수정 버전: 병렬 처리 적용)
import os
import sys
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed # ProcessPoolExecutor 추가

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from app.config.database import get_db_connection

MODEL_NAME = "nlp04/korean_sentiment_analysis_kcelectra"
# Railway의 8 vCPU를 활용합니다.
MAX_WORKERS = 8 

# 감정 라벨 설정 (기존과 동일)
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

# DB 함수는 그대로 유지
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
            # 튜플의 리스트를 반환합니다. (idx, store_idx, review_text)
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

# ----------------------------------------------------
# 병렬 처리를 위한 청크 처리 함수
# ----------------------------------------------------

def process_review_chunk(chunk_rows: List[tuple]):
    """
    단일 프로세스에서 리뷰 데이터 청크를 처리하고 DB에 업데이트합니다.
    모델 로드는 각 프로세스 내부에서 한 번만 수행됩니다.
    """
    pid = os.getpid()
    
    try:
        print(f"[Process {pid}] 🔄 모델 로드 시작...")
        
        # 모델을 각 프로세스 내부에서 로드해야 직렬화 오류를 피할 수 있습니다.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # 8GB 메모리이므로 load_in_8bit=True 옵션은 일단 제거하고 기본 로드합니다.
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, use_safetensors=True, trust_remote_code=True)
        
        # GPU 사용 불가능하므로 device를 -1 (CPU)로 명시합니다.
        clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1, 
        )
        print(f"[Process {pid}] ✅ 모델 로드 완료. 리뷰 {len(chunk_rows)}개 처리 시작.")

        processed_count = 0
        
        for idx, store_idx, text in chunk_rows:
            # 모델 추론
            res = clf(text, truncation=True, max_length=128)[0]

            label = res["label"]
            prob = float(res["score"])

            # 감정 라벨을 [-1, 1] 스코어로 변환
            if label in POSITIVE_LABELS:
                base = 1.0
            elif label in NEGATIVE_LABELS:
                base = -1.0
            else:  # NEUTRAL_LABELS 또는 기타 미정 라벨
                base = 0.0

            score = base * prob

            # DB 업데이트
            update_review_sentiment(idx, score, label)
            
            processed_count += 1
            if processed_count % 1000 == 0:
                 print(f"[Process {pid}] 리뷰 {processed_count}/{len(chunk_rows)} 처리 및 DB 업데이트 완료.")

        print(f"[Process {pid}] 🎉 청크 처리 완료. 총 {processed_count}개.")
        return processed_count

    except Exception as e:
        print(f"[Process {pid}] ❌ 프로세스 실행 중 치명적 오류: {e}")
        # 오류 발생 시 0을 반환하여 최종 집계에 영향 없도록 합니다.
        return 0

# ----------------------------------------------------
# 메인 배치 실행 함수 수정
# ----------------------------------------------------

def run_sentiment_batch():
    print("🔹 감성 분석 대상 리뷰 로드...")
    rows = get_reviews_without_sentiment()
    if not rows:
        print("⭐ 새로운 리뷰 없음")
        return

    TOTAL_REVIEWS = len(rows)
    print(f"📌 {TOTAL_REVIEWS}개 리뷰 처리 예정 (병렬 처리 사용).")

    # 1. 리뷰 데이터를 청크로 분할
    # MAX_WORKERS 만큼의 청크로 나눕니다.
    chunk_size = TOTAL_REVIEWS // MAX_WORKERS + 1 
    chunks = [rows[i:i + chunk_size] for i in range(0, TOTAL_REVIEWS, chunk_size)]
    print(f"🔬 리뷰를 {len(chunks)}개의 청크(약 {chunk_size}개/청크)로 분할했습니다.")
    
    # 2. ProcessPoolExecutor를 사용하여 병렬 실행
    total_processed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # map 대신 submit을 사용해 로그 출력을 실시간으로 확인합니다.
        futures = [executor.submit(process_review_chunk, chunk) for chunk in chunks]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                total_processed += result
            except Exception as e:
                print(f"❌ 병렬 처리 중 메인 스레드 오류 발생: {e}")
    
    # 3. 결과 출력
    if total_processed == TOTAL_REVIEWS:
        print(f"\n🎉 감성 분석 배치 완료! 총 {total_processed}개 리뷰 처리 완료.")
    else:
        print(f"\n⚠️ 감성 분석 배치 완료. {TOTAL_REVIEWS}개 중 {total_processed}개 리뷰 처리 완료. 오류가 발생했을 수 있습니다.")


if __name__ == "__main__":
    # torch.cuda.is_available() 검사는 병렬 처리에 불필요하므로 제거합니다.
    print(f"Device set to use cpu (Max Workers: {MAX_WORKERS})")
    run_sentiment_batch()
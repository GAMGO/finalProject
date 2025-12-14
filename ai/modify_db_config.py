# modify_db_config.py (ai 루트 폴더에 위치)

import sys
import os

# Docker 컨테이너 내부 경로: /app/config/database.py
DB_FILE_PATH = "app/config/database.py"

# 핵심 문자열 정의 (앞뒤 공백 무시)
# 이 문자열이 database.py 파일에 반드시 존재해야 합니다.
UNCOMMENTED_CORE = 'cursorclass=pymysql.cursors.DictCursor'
COMMENTED_CORE = '# cursorclass=pymysql.cursors.DictCursor'


def modify_config(action):
    """
    database.py 파일에서 DictCursor 줄을 주석 처리하거나 주석을 해제합니다.
    줄 단위 비교 대신, 파일 전체에서 핵심 문자열을 찾아 대체하는 방식으로 안정성을 높입니다.
    action: 'comment' 또는 'uncomment'
    """
    
    # 1. 파일 경로 확인
    if not os.path.exists(DB_FILE_PATH):
        print(f"❌ 오류: 파일 {DB_FILE_PATH}을 찾을 수 없습니다. 경로를 확인하세요.")
        sys.exit(1)

    try:
        # 2. 파일 전체 내용 읽기
        with open(DB_FILE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()

        is_modified = False

        if action == 'comment':
            # 주석 처리: 원본 문자열이 파일에 있는지 확인 후 대체
            if UNCOMMENTED_CORE in content:
                # 띄어쓰기를 보존하며 치환하기 위해 줄 전체를 치환하는 것이 아니라 핵심 부분만 치환
                content = content.replace(UNCOMMENTED_CORE, COMMENTED_CORE, 1) # 한 번만 치환
                is_modified = True
        
        elif action == 'uncomment':
            # 주석 해제: 주석 처리된 문자열이 파일에 있는지 확인 후 대체
            if COMMENTED_CORE in content:
                content = content.replace(COMMENTED_CORE, UNCOMMENTED_CORE, 1) # 한 번만 치환
                is_modified = True
        
        # 3. 파일 쓰기
        if is_modified:
            with open(DB_FILE_PATH, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ database.py: {action} 작업 완료. DictCursor 설정이 변경되었습니다.")
        else:
            print(f"⚠️ database.py: {action}할 내용이 이미 적용되어 있었습니다. 건너뜁니다. (파일 수정 없음)")

    except Exception as e:
        print(f"❌ 파일 수정 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python modify_db_config.py [comment|uncomment]")
        sys.exit(1)
    
    action = sys.argv[1]
    if action in ['comment', 'uncomment']:
        modify_config(action)
    else:
        print("잘못된 인자입니다. 'comment' 또는 'uncomment'를 사용하세요.")
        sys.exit(1)
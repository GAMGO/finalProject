# modify_db_config.py (ai 루트 폴더에 위치)

import sys
import os

# Docker 컨테이너 내부 경로: /app/config/database.py
DB_FILE_PATH = "app/config/database.py"

# 원본 코드
TARGET_LINE_UNCOMMENTED = '        cursorclass=pymysql.cursors.DictCursor  # ★ 핵심 포인트\n'

# 주석 처리된 코드 (쉼표 유무에 주의해야 함)
# 사용자가 제공한 코드에서는 이 줄이 get_db_connection() 함수에서 마지막 인자처럼 보이지만,
# 혹시 get_conn() 함수를 사용하고 있다면 해당 함수 내에서 return 전에 conn.cursor() 호출 시 적용됩니다.
# 사용자가 제공한 database.py 코드의 get_db_connection() 함수를 기준으로 작성합니다.

def modify_config(action):
    """
    database.py 파일에서 DictCursor 줄을 주석 처리하거나 주석을 해제합니다.
    action: 'comment' 또는 'uncomment'
    """
    
    if not os.path.exists(DB_FILE_PATH):
        print(f"❌ 오류: 파일 {DB_FILE_PATH}을 찾을 수 없습니다. 경로를 확인하세요.")
        sys.exit(1)

    try:
        with open(DB_FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        is_modified = False

        for line in lines:
            stripped_line = line.strip()

            if action == 'comment':
                # 주석이 없는 상태일 때 주석 처리
                if stripped_line == TARGET_LINE_UNCOMMENTED.strip():
                    new_lines.append(f'        # {TARGET_LINE_UNCOMMENTED.strip()}\n')
                    is_modified = True
                else:
                    new_lines.append(line)
            
            elif action == 'uncomment':
                # 주석이 있는 상태일 때 주석 해제
                if stripped_line == f'# {TARGET_LINE_UNCOMMENTED.strip()}':
                    new_lines.append(f'        {TARGET_LINE_UNCOMMENTED.strip()}\n')
                    is_modified = True
                else:
                    new_lines.append(line)
            
            else:
                new_lines.append(line)


        if is_modified:
            with open(DB_FILE_PATH, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"✅ database.py: {action} 작업 완료.")
        else:
            print(f"⚠️ database.py: {action}할 내용이 이미 적용되어 있었습니다. 건너뜁니다.")

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
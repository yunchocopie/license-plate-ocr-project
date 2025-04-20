import os
import unicodedata
from zipfile import ZipFile

# zip 파일 내부에 있는 파일을 파일명 인코딩 오류 없이 안전하게 추출하기 위함
# 시도할 인코딩 목록 (우선순위별)
CANDIDATE_ENCODINGS = ['utf-8', 'euc-kr', 'cp949', 'iso-8859-1']

# 이름 디코딩 시도 함수
def try_decode_cp437_name(name_bytes):
    for encoding in CANDIDATE_ENCODINGS:
        try:
            return name_bytes.decode(encoding), encoding # 성공시 디코딩된 문자열과 인코딩 이름 반
        except UnicodeDecodeError:
            continue
    return None, None  # 실패

def extract_zip_safely(zip_path, extract_to):
    with ZipFile(zip_path, 'r') as zf: # zip 파일을 열어서 내부를 순회
        for info in zf.infolist():
            try:
                if info.is_dir(): # 디렉토리는 스킵
                    continue

                # 이름 디코딩 (깨져도 원래 이름 유지)
                try:
                    raw_bytes = info.filename.encode('cp437') # zip 내부의 이름을 원래 바이트로 복원
                    decoded_name = raw_bytes.decode('utf-8')  # or use your try_decode_cp437_name
                except Exception:
                    decoded_name = info.filename

                safe_name = unicodedata.normalize('NFC', decoded_name) # 한글 조합 문제 해결(ㄱ ㅣ ㅁ -> 김)

                target_path = os.path.join(extract_to, safe_name)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                with zf.open(info) as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())

            except Exception as e:
                print(f"⚠️ 무시된 항목: {info.filename} → {e}")
                continue


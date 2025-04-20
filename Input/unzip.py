import os
import unicodedata

# 시도할 인코딩 목록 (우선순위별)
CANDIDATE_ENCODINGS = ['utf-8', 'euc-kr', 'cp949', 'iso-8859-1']

def try_decode_cp437_name(name_bytes):
    for encoding in CANDIDATE_ENCODINGS:
        try:
            return name_bytes.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return None, None  # 실패

def extract_zip_safely(zip_path, extract_to):
    from zipfile import ZipFile

    with ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            try:
                if info.is_dir():
                    continue

                # 이름 디코딩 (깨져도 원래 이름 유지)
                try:
                    raw_bytes = info.filename.encode('cp437')
                    decoded_name = raw_bytes.decode('utf-8')  # or use your try_decode_cp437_name
                except Exception:
                    decoded_name = info.filename

                safe_name = unicodedata.normalize('NFC', decoded_name)

                target_path = os.path.join(extract_to, safe_name)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                with zf.open(info) as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())

            except Exception as e:
                print(f"⚠️ 무시된 항목: {info.filename} → {e}")
                continue


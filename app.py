import streamlit as st
from Input.plate_detector import uploaded_files

st.set_page_config(page_title="번호판 인식기 테스트", layout="centered")
st.title("📂 번호판 자동 인식기 (로컬 저장용)")

input_files = st.file_uploader(
    "이미지 여러 개 또는 ZIP 파일을 업로드하세요",
    type=["jpg", "jpeg", "png", "zip"],
    accept_multiple_files=True
)

if input_files:
    saved_paths = uploaded_files(input_files, save_dir='outputs')
    st.success(f"{len(saved_paths)}개의 번호판 이미지가 'outputs/' 폴더에 저장되었습니다.")

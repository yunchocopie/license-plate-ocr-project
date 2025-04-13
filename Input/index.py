import streamlit as st
from InputData import inputData

upload_images = st.file_uploader("이미지를 업로드하세요", type=['jpg','jpeg', 'zip', 'png'], accept_multiple_files=True)

if upload_images:
    inputData(upload_images)
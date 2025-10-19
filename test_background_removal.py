import cv2
import numpy as np
from src.preprocessing.character_segmentation import CharacterSegmentation
from src.preprocessing.image_processor import ImageProcessor
import config

# 테스트 이미지 로드 (샘플 번호판 이미지 필요)
print("=== 배경 제거 테스트 ===")
print(f"ENABLE_CHAR_SEGMENTATION: {config.ENABLE_CHAR_SEGMENTATION}")
print(f"CHAR_SEGMENTATION_MODE: {config.CHAR_SEGMENTATION_MODE}")

# 간단한 테스트 이미지 생성 (실제로는 번호판 이미지 사용)
test_image = np.ones((80, 320, 3), dtype=np.uint8) * 200  # 회색 배경

# CharacterSegmentation 테스트
char_seg = CharacterSegmentation(plate_type='general')
print(f"\n문자 필터 파라미터: {char_seg.filter_params}")

# ImageProcessor 테스트
processor = ImageProcessor()

# 실제 이미지가 있다면 로드
import os
sample_dir = os.path.join(config.DATA_DIR, "raw")
if os.path.exists(sample_dir):
    files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png'))]
    if files:
        img_path = os.path.join(sample_dir, files[0])
        test_image = cv2.imread(img_path)
        print(f"\n테스트 이미지 로드: {img_path}")
        print(f"이미지 크기: {test_image.shape}")

        # 문자 영역 추출 테스트
        result = char_seg.extract_character_regions(test_image)
        print(f"\n문자 추출 결과:")
        print(f"  - 성공: {result['success']}")
        print(f"  - 검출 문자 수: {result.get('num_characters', 0)}")
        if not result['success']:
            print(f"  - 에러: {result.get('error', 'Unknown')}")

        # visualize_steps_opencv_method 테스트
        steps = processor.visualize_steps_opencv_method(test_image)
        print(f"\n시각화 단계: {list(steps.keys())}")

        # 배경 제거 관련 단계 확인
        if 'final_ocr_input' in steps:
            print("✅ final_ocr_input 단계 존재 (배경 제거 성공)")
        else:
            print("❌ final_ocr_input 단계 없음 (배경 제거 실패)")
    else:
        print(f"샘플 이미지 없음: {sample_dir}")
else:
    print(f"데이터 디렉토리 없음: {sample_dir}")

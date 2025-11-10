"""
업로드된 번호판 이미지 테스트 스크립트

이미지 전처리 파이프라인의 각 단계를 시각화하고 문제점을 찾습니다.
"""

import cv2
import numpy as np
import os
from pathlib import Path

# 프로젝트 루트 디렉토리 설정
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.warp import process_plate_warp, select_template
from src.preprocessing.pipelines import preprocess_plate_image
from src.ocr.slot_classifier import SlotClassifier, recognize_plate_slots
import config

def save_debug_image(image, name, output_dir="data/debug/test_uploaded"):
    """디버그 이미지 저장"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.png")
    cv2.imwrite(output_path, image)
    print(f"✓ Saved: {output_path}")
    return output_path

def test_uploaded_plate(image_path):
    """업로드된 번호판 이미지 테스트"""

    print(f"\n{'='*60}")
    print(f"테스트 시작: {image_path}")
    print(f"{'='*60}\n")

    # 1. 이미지 로드
    print("Step 1: 이미지 로드")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        return

    print(f"  - 원본 이미지 크기: {image.shape}")
    save_debug_image(image, "01_original")

    # 2. 템플릿 선택
    print("\nStep 2: 템플릿 선택")
    h, w = image.shape[:2]
    ratio = w / h
    print(f"  - 종횡비: {ratio:.2f} (width={w}, height={h})")

    template_meta = select_template(w, h)
    print(f"  - 선택된 템플릿: {template_meta.plate_type}")
    print(f"  - 템플릿 크기: {template_meta.size}")
    print(f"  - 슬롯 개수: {len(template_meta.slots)}")

    # 3. 워프 및 원근 변환
    print("\nStep 3: 워프 및 원근 변환")
    try:
        warped, template_meta = process_plate_warp(image, bbox=None, debug=False)
        print(f"  - 워프 후 크기: {warped.shape}")
        print(f"  - 모서리 신뢰도: {template_meta.corners_confidence:.2f}")
        save_debug_image(warped, "02_warped")
    except Exception as e:
        print(f"  ❌ 워프 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 전처리 파이프라인
    print("\nStep 4: 적응형 전처리 파이프라인")
    try:
        processed, meta = preprocess_plate_image(
            warped,
            template_meta,
            blur_threshold=config.BLUR_THRESHOLD,
            contrast_threshold=config.CONTRAST_THRESHOLD,
            noise_threshold=config.NOISE_THRESHOLD,
            adaptive_enabled=config.ADAPTIVE_PREPROCESSING
        )
        print(f"  - 품질 지표:")
        print(f"    * Blur: {meta['quality']['blur']:.2f}")
        print(f"    * Contrast: {meta['quality']['contrast']:.2f}")
        print(f"    * Noise: {meta['quality']['noise']:.2f}")
        print(f"  - 선택된 파이프라인: {meta['pipeline']}")
        save_debug_image(processed, "03_preprocessed")
    except Exception as e:
        print(f"  ❌ 전처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 슬롯 추출 및 시각화
    print("\nStep 5: 슬롯 추출")
    try:
        from src.ocr.slot_classifier import extract_slots

        slot_crops = extract_slots(processed, template_meta.slots)
        print(f"  - 추출된 슬롯: {len(slot_crops)}개")

        # 슬롯 이미지 저장
        for i, (slot, crop) in enumerate(zip(template_meta.slots, slot_crops)):
            slot_name = f"04_slot_{i:02d}_{slot.name}"
            save_debug_image(crop, slot_name)
            print(f"    * {slot.name}: {crop.shape}")
    except Exception as e:
        print(f"  ❌ 슬롯 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. OCR 인식 (슬롯 기반)
    print("\nStep 6: 슬롯 기반 OCR 인식")
    try:
        classifier = SlotClassifier(
            model_path=config.SLOT_CLASSIFIER_MODEL,
            use_easyocr=config.USE_EASYOCR_FOR_SLOTS
        )

        text, chars, probs = recognize_plate_slots(processed, template_meta, classifier)

        print(f"  - 인식된 문자열: '{text}'")
        print(f"  - 슬롯별 결과:")
        for slot, char, prob in zip(template_meta.slots, chars, probs):
            print(f"    * {slot.name}: '{char}' (신뢰도: {prob:.3f})")
    except Exception as e:
        print(f"  ❌ OCR 인식 실패: {e}")
        import traceback
        traceback.print_exc()

    # 7. 슬롯 시각화 (바운딩 박스)
    print("\nStep 7: 슬롯 시각화")
    try:
        vis_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR) if len(processed.shape) == 2 else processed.copy()

        for i, slot in enumerate(template_meta.slots):
            # 슬롯 박스 그리기
            cv2.rectangle(vis_image, (slot.x, slot.y), (slot.x + slot.w, slot.y + slot.h), (0, 255, 0), 2)
            # 슬롯 이름 표시
            cv2.putText(vis_image, slot.name, (slot.x, slot.y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        save_debug_image(vis_image, "05_slots_visualization")
    except Exception as e:
        print(f"  ❌ 시각화 실패: {e}")

    print(f"\n{'='*60}")
    print("테스트 완료!")
    print(f"{'='*60}\n")
    print(f"디버그 이미지 저장 위치: data/debug/test_uploaded/")

if __name__ == "__main__":
    # 업로드된 이미지 경로
    image_path = "docs/test_plate.jpg"

    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일이 없습니다: {image_path}")
        print("업로드된 이미지를 docs/test_plate.jpg로 저장해주세요.")
    else:
        test_uploaded_plate(image_path)

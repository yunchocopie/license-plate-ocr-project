"""
전처리 파이프라인 전용 테스트 스크립트

OCR 없이 이미지 전처리 단계만 테스트합니다.
"""

import cv2
import numpy as np
import os
from pathlib import Path

# 프로젝트 루트 디렉토리 설정
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.warp import (
    process_plate_warp,
    select_template,
    detect_plate_corners
)
from src.preprocessing.pipelines import preprocess_plate_image, select_pipeline
from src.preprocessing.quality import analyze_quality

def save_debug_image(image, name, output_dir="data/debug/test_preprocessing"):
    """디버그 이미지 저장"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.png")
    cv2.imwrite(output_path, image)
    print(f"✓ Saved: {output_path}")
    return output_path

def draw_text_on_image(image, text, position=(10, 30), color=(0, 255, 0)):
    """이미지에 텍스트 오버레이"""
    img_copy = image.copy()
    if len(img_copy.shape) == 2:  # Grayscale
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
    cv2.putText(img_copy, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img_copy

def test_preprocessing_pipeline(image_path):
    """전처리 파이프라인 전용 테스트"""

    print(f"\n{'='*60}")
    print(f"전처리 파이프라인 테스트: {image_path}")
    print(f"{'='*60}\n")

    # 1. 이미지 로드
    print("Step 1: 이미지 로드")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        return

    print(f"  ✓ 원본 이미지 크기: {image.shape}")
    save_debug_image(image, "01_original")

    # 2. 템플릿 선택
    print("\nStep 2: 템플릿 선택")
    h, w = image.shape[:2]
    ratio = w / h
    print(f"  - 이미지 크기: {w} x {h}")
    print(f"  - 종횡비: {ratio:.2f}")

    template_meta = select_template(w, h)
    print(f"  ✓ 선택된 템플릿: {template_meta.plate_type}")
    print(f"  ✓ 템플릿 크기: {template_meta.size}")
    print(f"  ✓ 슬롯 개수: {len(template_meta.slots)}")

    # 3. 모서리 검출
    print("\nStep 3: 모서리 검출")
    try:
        corners, confidence = detect_plate_corners(image, debug=False)
        print(f"  ✓ 모서리 검출 신뢰도: {confidence:.2f}")
        print(f"  - 모서리 좌표:")
        for i, corner in enumerate(corners):
            print(f"    {i+1}: ({corner[0]:.1f}, {corner[1]:.1f})")

        # 모서리 시각화
        vis_corners = image.copy()
        cv2.polylines(vis_corners, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        for i, corner in enumerate(corners):
            cv2.circle(vis_corners, tuple(corner.astype(int)), 5, (0, 0, 255), -1)
            cv2.putText(vis_corners, str(i+1), tuple((corner + 10).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        save_debug_image(vis_corners, "02_detected_corners")
    except Exception as e:
        print(f"  ❌ 모서리 검출 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 워프 및 원근 변환
    print("\nStep 4: 워프 및 원근 변환")
    try:
        warped, template_meta = process_plate_warp(image, bbox=None, debug=False)
        print(f"  ✓ 워프 후 크기: {warped.shape}")
        print(f"  ✓ 목표 크기: {template_meta.size}")
        save_debug_image(warped, "03_warped")
    except Exception as e:
        print(f"  ❌ 워프 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 품질 분석
    print("\nStep 5: 이미지 품질 분석")
    try:
        quality = analyze_quality(warped)
        print(f"  품질 지표:")
        print(f"    - Blur (선명도): {quality['blur']:.2f} (>100 권장)")
        print(f"    - Contrast (대비): {quality['contrast']:.2f} (>20 권장)")
        print(f"    - Noise (노이즈): {quality['noise']:.2f} (<35 권장)")

        # 임계값 설정 (config에서 가져오기)
        blur_threshold = 120.0
        contrast_threshold = 20.0
        noise_threshold = 35.0

        # 품질 판정
        issues = []
        if quality['blur'] < blur_threshold:
            issues.append("흐림")
        if quality['contrast'] < contrast_threshold:
            issues.append("저대비")
        if quality['noise'] > noise_threshold:
            issues.append("노이즈")

        if issues:
            print(f"  ⚠️  감지된 품질 문제: {', '.join(issues)}")
        else:
            print(f"  ✓ 품질 양호")

        # 파이프라인 선택
        pipeline_key = select_pipeline(
            quality,
            blur_threshold=blur_threshold,
            contrast_threshold=contrast_threshold,
            noise_threshold=noise_threshold,
            adaptive_enabled=True
        )
        print(f"  ✓ 선택된 파이프라인: {pipeline_key}")

    except Exception as e:
        print(f"  ❌ 품질 분석 실패: {e}")
        import traceback
        traceback.print_exc()

    # 6. 전처리 파이프라인 적용
    print("\nStep 6: 적응형 전처리 파이프라인")
    try:
        processed, meta = preprocess_plate_image(
            warped,
            template_meta,
            blur_threshold=120.0,
            contrast_threshold=20.0,
            noise_threshold=35.0,
            adaptive_enabled=True
        )
        print(f"  ✓ 적용된 파이프라인: {meta['pipeline']}")
        save_debug_image(processed, "04_preprocessed")

        # 전처리 전후 비교
        comparison = np.hstack([
            cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped,
            processed
        ])
        save_debug_image(comparison, "05_before_after_comparison")

    except Exception as e:
        print(f"  ❌ 전처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. 슬롯 위치 시각화
    print("\nStep 7: 슬롯 위치 시각화")
    try:
        # 전처리된 이미지를 BGR로 변환
        if len(processed.shape) == 2:
            vis_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = processed.copy()

        # 슬롯 박스 그리기
        for i, slot in enumerate(template_meta.slots):
            # 슬롯 박스
            cv2.rectangle(vis_image,
                         (slot.x, slot.y),
                         (slot.x + slot.w, slot.y + slot.h),
                         (0, 255, 0), 2)
            # 슬롯 이름
            cv2.putText(vis_image, slot.name,
                       (slot.x, slot.y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            # 슬롯 번호
            cv2.putText(vis_image, str(i),
                       (slot.x + 2, slot.y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        save_debug_image(vis_image, "06_slots_visualization")
        print(f"  ✓ 슬롯 {len(template_meta.slots)}개 시각화 완료")

        # 슬롯별 상세 정보
        print(f"\n  슬롯 상세 정보:")
        for i, slot in enumerate(template_meta.slots):
            print(f"    [{i}] {slot.name:12s}: x={slot.x:3d}, y={slot.y:3d}, w={slot.w:3d}, h={slot.h:3d}")

    except Exception as e:
        print(f"  ❌ 슬롯 시각화 실패: {e}")
        import traceback
        traceback.print_exc()

    # 8. 개별 슬롯 추출
    print("\nStep 8: 개별 슬롯 이미지 추출")
    try:
        slot_dir = "data/debug/test_preprocessing/slots"
        os.makedirs(slot_dir, exist_ok=True)

        for i, slot in enumerate(template_meta.slots):
            # 슬롯 영역 추출
            crop = processed[slot.y:slot.y + slot.h, slot.x:slot.x + slot.w]

            # 파일명
            slot_filename = f"slot_{i:02d}_{slot.name}.png"
            slot_path = os.path.join(slot_dir, slot_filename)

            # 저장
            cv2.imwrite(slot_path, crop)
            print(f"    [{i}] {slot.name:12s}: {crop.shape} → {slot_filename}")

        print(f"  ✓ 슬롯 이미지 저장: {slot_dir}/")

    except Exception as e:
        print(f"  ❌ 슬롯 추출 실패: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print("✓ 전처리 파이프라인 테스트 완료!")
    print(f"{'='*60}\n")
    print(f"📂 디버그 이미지 저장 위치:")
    print(f"   data/debug/test_preprocessing/")
    print(f"   data/debug/test_preprocessing/slots/")

if __name__ == "__main__":
    import sys

    # 이미지 경로 지정
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 기본 경로: 샘플 이미지
        image_path = "docs/tmp_plate6.jpg"

    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일이 없습니다: {image_path}")
        print("\n사용법:")
        print(f"  python test_preprocessing_only.py <image_path>")
        print(f"\n예시:")
        print(f"  python test_preprocessing_only.py docs/tmp_plate6.jpg")
    else:
        test_preprocessing_pipeline(image_path)

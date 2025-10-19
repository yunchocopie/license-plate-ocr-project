import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import config

"""
문자 영역 정밀 추출 모듈

Contour 기반으로 번호판 이미지에서 문자 영역만 추출하여
배경, 테두리, 노이즈를 제거함으로써 OCR 정확도를 향상시킵니다.

참고: https://ssam2s.tistory.com/5
한국 번호판 특성에 맞게 최적화
"""

class CharacterSegmentation:
    """Contour 기반 문자 영역 추출 클래스"""

    def __init__(self, plate_type: str = 'general'):
        """
        CharacterSegmentation 초기화

        Args:
            plate_type: 번호판 타입 ('general', 'commercial', 'electric' 등)
        """
        self.plate_type = plate_type

        # 번호판 타입별 문자 필터링 파라미터
        self.char_filters = {
            'general': {  # 일반 번호판: 03마7893
                'MINAREA': 30,        # 최소 면적 완화 (80 → 30)
                'MINWIDTH': 5,        # 최소 폭 완화 (8 → 5)
                'MINHEIGHT': 10,      # 최소 높이 완화 (15 → 10)
                'MINRATIO': 0.2,      # 비율 범위 확대 (0.3 → 0.2)
                'MAXRATIO': 1.5,      # 비율 범위 확대 (1.2 → 1.5)
                'MIN_CHARS': 4,       # 최소 문자 수 완화 (5 → 4)
                'MAX_ANGLE': 15,
                'CHAR_SPACING_MIN': 2,
                'CHAR_SPACING_MAX': 50  # 간격 범위 확대 (40 → 50)
            },
            'general_3digit': {  # 3자리 번호판: 145하1937
                'MINAREA': 30,
                'MINWIDTH': 5,
                'MINHEIGHT': 10,
                'MINRATIO': 0.2,
                'MAXRATIO': 1.5,
                'MIN_CHARS': 4,
                'MAX_ANGLE': 15,
                'CHAR_SPACING_MIN': 2,
                'CHAR_SPACING_MAX': 50
            },
            'commercial': {  # 영업용: 경기37바2120
                'MINAREA': 30,
                'MINWIDTH': 5,
                'MINHEIGHT': 10,
                'MINRATIO': 0.2,
                'MAXRATIO': 1.5,
                'MIN_CHARS': 4,
                'MAX_ANGLE': 15,
                'CHAR_SPACING_MIN': 2,
                'CHAR_SPACING_MAX': 50
            },
            'electric': {  # 전기차
                'MINAREA': 30,
                'MINWIDTH': 5,
                'MINHEIGHT': 10,
                'MINRATIO': 0.2,
                'MAXRATIO': 1.5,
                'MIN_CHARS': 4,
                'MAX_ANGLE': 15,
                'CHAR_SPACING_MIN': 2,
                'CHAR_SPACING_MAX': 50
            },
            'motorcycle': {  # 이륜차 (작은 크기)
                'MINAREA': 20,
                'MINWIDTH': 4,
                'MINHEIGHT': 8,
                'MINRATIO': 0.2,
                'MAXRATIO': 1.5,
                'MIN_CHARS': 3,
                'MAX_ANGLE': 15,
                'CHAR_SPACING_MIN': 1,
                'CHAR_SPACING_MAX': 40
            },
            'default': {  # 기본값
                'MINAREA': 30,
                'MINWIDTH': 5,
                'MINHEIGHT': 10,
                'MINRATIO': 0.2,
                'MAXRATIO': 1.5,
                'MIN_CHARS': 4,
                'MAX_ANGLE': 15,
                'CHAR_SPACING_MIN': 2,
                'CHAR_SPACING_MAX': 50
            }
        }

        # 현재 타입의 필터 가져오기
        self.filter_params = self.char_filters.get(
            plate_type,
            self.char_filters['default']
        )

    def apply_morphology(self, image: np.ndarray) -> np.ndarray:
        """
        Morphology 연산 적용 (TopHat, BlackHat)

        TopHat: 밝은 문자 강조
        BlackHat: 어두운 배경 제거

        Args:
            image: 그레이스케일 이미지

        Returns:
            개선된 이미지
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 구조 요소 생성 (3x3 정사각형)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # TopHat: 밝은 영역(문자) 강조
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # BlackHat: 어두운 영역(배경) 추출
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # 결합: 원본 + TopHat - BlackHat
        enhanced = cv2.add(gray, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)

        return enhanced

    def preprocess_for_contour(self, image: np.ndarray) -> np.ndarray:
        """
        Contour 검출을 위한 전처리

        Args:
            image: 입력 이미지

        Returns:
            전처리된 이진 이미지
        """
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # ⭐ 단순화된 전처리 (문자 보존 우선)

        # 1. 약한 노이즈 제거
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        # 2. CLAHE 대비 향상 (약하게)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)

        # 3. 적응형 이진화 (Otsu보다 안전)
        binary = cv2.adaptiveThreshold(
            contrast,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # 문자가 흰색
            21,  # block_size 크게 (더 관대하게)
            3    # C 값 낮게 (더 많은 영역 검출)
        )

        # 4. 최소한의 모폴로지 (문자 연결만)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def filter_character_contours(self, contours: List, image_shape: Tuple) -> List:
        """
        문자 영역으로 추정되는 Contour만 필터링

        Args:
            contours: 검출된 모든 contour
            image_shape: 이미지 크기 (h, w)

        Returns:
            필터링된 contour 리스트
        """
        filtered = []
        h, w = image_shape[:2]

        params = self.filter_params

        for contour in contours:
            # 면적 계산
            area = cv2.contourArea(contour)

            # 최소 면적 필터
            if area < params['MINAREA']:
                continue

            # 바운딩 박스 계산
            x, y, width, height = cv2.boundingRect(contour)

            # 크기 필터
            if width < params['MINWIDTH'] or height < params['MINHEIGHT']:
                continue

            # 가로/세로 비율 필터 (한글 문자는 정사각형에 가까움)
            ratio = width / height if height > 0 else 0
            if not (params['MINRATIO'] <= ratio <= params['MAXRATIO']):
                continue

            # 이미지 경계 필터 완화 (번호판 문자는 경계에 가까울 수 있음)
            # margin = 3
            # if x < margin or y < margin or (x + width) > (w - margin) or (y + height) > (h - margin):
            #     continue

            # 너무 큰 contour 제외 (전체 영역일 가능성)
            if area > (h * w * 0.5):
                continue

            filtered.append({
                'contour': contour,
                'area': area,
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'ratio': ratio
            })

        return filtered

    def create_clean_image_with_white_background(self, image: np.ndarray, character_contours: List) -> np.ndarray:
        """
        ⭐ 배경을 흰색으로 제거하고 문자만 남김

        마스크를 생성하여 문자 영역만 남기고 나머지는 흰색(255)으로 채움

        Args:
            image: 원본 이미지 (그레이스케일 또는 컬러)
            character_contours: 검출된 문자 contour 리스트 (dict 형태)

        Returns:
            배경이 흰색이고 문자만 남은 그레이스케일 이미지
        """
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # ⭐ 배경 제거: 마스크 생성
        # 1. 흰색(255) 배경 이미지 생성
        white_background = np.ones_like(gray) * 255

        # 2. 문자 영역 마스크 생성 (검은색 배경)
        mask = np.zeros_like(gray)

        # 3. 각 문자 contour를 마스크에 그리기 (흰색으로)
        for char_info in character_contours:
            contour = char_info['contour']
            # Contour 내부를 흰색(255)으로 채움
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # 4. 마스크를 사용하여 문자만 추출
        # mask가 255인 곳은 원본 이미지, 0인 곳은 흰색 배경
        clean_image = np.where(mask == 255, gray, white_background)

        return clean_image.astype(np.uint8)

    def create_clean_roi_image(self, image: np.ndarray, best_group: List) -> Dict:
        """
        문자 영역만 추출하고 배경을 흰색으로 제거한 ROI 이미지 생성

        Args:
            image: 원본 이미지
            best_group: 선택된 문자 그룹

        Returns:
            dict: {
                'clean_image': 배경이 흰색인 이미지,
                'roi_coords': ROI 좌표,
                'character_boxes': 문자 박스 리스트
            }
        """
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape

        # 1. 배경 제거된 이미지 생성
        clean_full = self.create_clean_image_with_white_background(image, best_group)

        # 2. ROI 영역 계산
        char_boxes = []
        min_x = float('inf')
        min_y = float('inf')
        max_x = 0
        max_y = 0

        for char_info in best_group:
            x, y, w_char, h_char = char_info['x'], char_info['y'], char_info['width'], char_info['height']
            char_boxes.append((x, y, w_char, h_char))

            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w_char)
            max_y = max(max_y, y + h_char)

        # 3. 여백 추가 (약간의 패딩)
        padding = 5
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(w, max_x + padding)
        max_y = min(h, max_y + padding)

        # 4. ROI 크롭
        clean_roi = clean_full[min_y:max_y, min_x:max_x]

        return {
            'clean_image': clean_roi,
            'clean_full_image': clean_full,
            'roi_coords': (min_x, min_y, max_x, max_y),
            'character_boxes': char_boxes
        }

    def group_character_candidates(self, filtered_contours: List) -> List[List]:
        """
        문자 후보들을 그룹화하여 번호판 문자열 찾기

        Args:
            filtered_contours: 필터링된 contour 리스트

        Returns:
            그룹화된 contour 리스트의 리스트
        """
        if not filtered_contours:
            return []

        # X 좌표 기준으로 정렬
        sorted_contours = sorted(filtered_contours, key=lambda c: c['x'])

        params = self.filter_params
        groups = []
        current_group = [sorted_contours[0]]

        for i in range(1, len(sorted_contours)):
            prev = sorted_contours[i - 1]
            curr = sorted_contours[i]

            # 이전 문자와의 거리 계산
            distance = curr['x'] - (prev['x'] + prev['width'])

            # 높이 차이 확인 (같은 라인에 있는지)
            height_diff = abs(curr['y'] - prev['y'])
            avg_height = (curr['height'] + prev['height']) / 2

            # 같은 그룹 조건
            if (params['CHAR_SPACING_MIN'] <= distance <= params['CHAR_SPACING_MAX'] and
                height_diff < avg_height * 0.5):
                current_group.append(curr)
            else:
                # 새 그룹 시작
                if len(current_group) >= params['MIN_CHARS']:
                    groups.append(current_group)
                current_group = [curr]

        # 마지막 그룹 추가
        if len(current_group) >= params['MIN_CHARS']:
            groups.append(current_group)

        return groups

    def extract_character_regions_from_binary(self, binary_image: np.ndarray, original_image: np.ndarray) -> Dict:
        """
        이진 이미지에서 문자 영역 추출 (C단계 → D단계 연결용)

        Args:
            binary_image: C5 단계에서 넘어온 이진 이미지
            original_image: 원본 이미지 (ROI 추출 및 시각화용)

        Returns:
            dict: extract_character_regions와 동일한 형식
        """
        if binary_image is None or binary_image.size == 0:
            return {
                'success': False,
                'roi_image': None,
                'original_with_boxes': None,
                'character_boxes': [],
                'num_characters': 0
            }

        # 1. C5 이진 이미지에서 직접 Contour 검출 (전처리 스킵)
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 2. 문자 영역 필터링
        filtered = self.filter_character_contours(contours, binary_image.shape)

        if not filtered:
            return {
                'success': False,
                'roi_image': None,
                'original_with_boxes': None,
                'character_boxes': [],
                'num_characters': 0,
                'error': 'No character contours found'
            }

        # 3. 문자 그룹화
        groups = self.group_character_candidates(filtered)

        if not groups:
            return {
                'success': False,
                'roi_image': None,
                'original_with_boxes': None,
                'character_boxes': [],
                'num_characters': 0,
                'error': 'No valid character groups found'
            }

        # 4. 가장 큰 그룹 선택 (번호판 문자열)
        best_group = max(groups, key=len)

        # 5. ROI 이미지 생성 (원본 이미지 사용)
        clean_result = self.create_clean_roi_image(original_image, best_group)

        # 6. 시각화용 박스 그리기
        if len(original_image.shape) == 3:
            display_image = original_image.copy()
        else:
            display_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        for char_info in best_group:
            x, y, w, h = char_info['x'], char_info['y'], char_info['width'], char_info['height']
            # 개별 문자 박스 (녹색)
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ROI 전체 박스 (빨간색)
        min_x, min_y, max_x, max_y = clean_result['roi_coords']
        cv2.rectangle(display_image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

        return {
            'success': True,
            'roi_image': clean_result['clean_image'],
            'clean_full_image': clean_result['clean_full_image'],
            'original_with_boxes': display_image,
            'character_boxes': clean_result['character_boxes'],
            'num_characters': len(best_group),
            'roi_coords': clean_result['roi_coords']
        }

    def extract_character_regions(self, image: np.ndarray) -> Dict:
        """
        문자 영역 추출 메인 메서드

        Args:
            image: 입력 번호판 이미지

        Returns:
            dict: {
                'success': 성공 여부,
                'roi_image': 문자 영역만 크롭된 이미지,
                'original_with_boxes': 박스가 그려진 원본 이미지 (디버깅용),
                'character_boxes': 각 문자의 좌표 리스트,
                'num_characters': 검출된 문자 수
            }
        """
        if image is None or image.size == 0:
            return {
                'success': False,
                'roi_image': None,
                'original_with_boxes': None,
                'character_boxes': [],
                'num_characters': 0
            }

        # 1. Contour 검출을 위한 전처리
        binary = self.preprocess_for_contour(image)

        # 2. Contour 검출
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 3. 문자 영역 필터링
        filtered = self.filter_character_contours(contours, image.shape)

        if not filtered:
            return {
                'success': False,
                'roi_image': None,
                'original_with_boxes': None,
                'character_boxes': [],
                'num_characters': 0,
                'error': 'No character contours found'
            }

        # 4. 문자 그룹화
        groups = self.group_character_candidates(filtered)

        if not groups:
            return {
                'success': False,
                'roi_image': None,
                'original_with_boxes': None,
                'character_boxes': [],
                'num_characters': 0,
                'error': 'No valid character groups found'
            }

        # 5. 가장 큰 그룹 선택 (번호판 문자열)
        best_group = max(groups, key=len)

        # 6. ⭐ 핵심: 배경을 흰색으로 제거하고 문자만 남김
        clean_result = self.create_clean_roi_image(image, best_group)

        # 7. 시각화용 박스 그리기
        if len(image.shape) == 3:
            display_image = image.copy()
        else:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for char_info in best_group:
            x, y, w, h = char_info['x'], char_info['y'], char_info['width'], char_info['height']
            # 개별 문자 박스 (녹색)
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ROI 전체 박스 (빨간색)
        min_x, min_y, max_x, max_y = clean_result['roi_coords']
        cv2.rectangle(display_image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

        return {
            'success': True,
            'roi_image': clean_result['clean_image'],  # ⭐ 배경이 흰색인 깨끗한 이미지
            'clean_full_image': clean_result['clean_full_image'],  # 전체 크기 배경 제거 이미지
            'original_with_boxes': display_image,
            'character_boxes': clean_result['character_boxes'],
            'num_characters': len(best_group),
            'roi_coords': clean_result['roi_coords']
        }

    def segment_plate(self, image: np.ndarray) -> np.ndarray:
        """
        번호판 문자 영역 추출 (간단한 인터페이스)

        Args:
            image: 입력 번호판 이미지

        Returns:
            문자 영역만 크롭된 이미지 (실패 시 원본 반환)
        """
        result = self.extract_character_regions(image)

        if result['success'] and result['roi_image'] is not None:
            return result['roi_image']
        else:
            # 실패 시 원본 반환
            return image

    def visualize_segmentation(self, image: np.ndarray) -> Dict:
        """
        문자 추출 과정 시각화 (배경 제거 포함)

        Args:
            image: 입력 이미지

        Returns:
            dict: 각 단계별 이미지
        """
        steps = {}

        # 1. 원본
        if len(image.shape) == 3:
            steps['1_original'] = image.copy()
        else:
            steps['1_original'] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 2. Morphology (TopHat/BlackHat)
        enhanced = self.apply_morphology(image)
        steps['2_morphology'] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) if len(enhanced.shape) == 2 else enhanced

        # 3. 전처리된 이진 이미지
        binary = self.preprocess_for_contour(image)
        steps['3_binary'] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # 4. 문자 영역 추출
        result = self.extract_character_regions(image)
        if result['success']:
            # 4a. 검출된 문자 박스
            steps['4_detected_boxes'] = result['original_with_boxes']

            # 4b. ⭐ 배경 제거된 전체 이미지
            if 'clean_full_image' in result:
                clean_full = result['clean_full_image']
                steps['5_background_removed'] = cv2.cvtColor(clean_full, cv2.COLOR_GRAY2BGR) if len(clean_full.shape) == 2 else clean_full

            # 4c. ⭐ ROI 크롭 + 배경 제거
            roi_clean = result['roi_image']
            steps['6_roi_clean'] = cv2.cvtColor(roi_clean, cv2.COLOR_GRAY2BGR) if len(roi_clean.shape) == 2 else roi_clean

        return steps

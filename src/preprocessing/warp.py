"""
템플릿 워프 & 슬롯 정의 모듈

번호판 모서리 검출, 원근 변환(템플릿 워프), 판형별 슬롯 좌표 정의를 담당합니다.
docs/ocr_structured_pipeline_plan.md 기반 구현
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional

# 번호판 타입 정의
PlateType = Literal["ONE_LINE", "TWO_LINE", "TWO_LINE_SMALL", "MOTORCYCLE"]


@dataclass
class Slot:
    """슬롯 (문자 영역) 정의"""
    name: str      # 슬롯 이름 (예: "region1", "number1", "hangul", etc.)
    x: int         # 좌상단 x 좌표
    y: int         # 좌상단 y 좌표
    w: int         # 너비
    h: int         # 높이


@dataclass
class TemplateMeta:
    """템플릿 메타데이터"""
    plate_type: PlateType                # 판형 타입
    size: Tuple[int, int]                # 워프된 이미지 크기 (width, height)
    slots: List[Slot]                    # 슬롯 리스트
    corners_confidence: float = 1.0      # 모서리 검출 신뢰도


# 템플릿 정의 (판형별 크기 및 슬롯 좌표)
# 좌표는 워프된 이미지를 기준으로 하드코딩

def get_one_line_template() -> TemplateMeta:
    """
    1행 번호판 템플릿 (예: 12가3456)
    크기: 520x110 픽셀
    구조: [숫자2자리][한글1자][숫자4자리]
    """
    width, height = 520, 110
    char_width = 60
    char_height = 80
    spacing = 10
    start_x = 20
    start_y = 15

    slots = []
    slot_names = ["num1", "num2", "hangul", "num3", "num4", "num5", "num6"]

    for i, name in enumerate(slot_names):
        x = start_x + i * (char_width + spacing)
        slots.append(Slot(name=name, x=x, y=start_y, w=char_width, h=char_height))

    return TemplateMeta(
        plate_type="ONE_LINE",
        size=(width, height),
        slots=slots
    )


def get_two_line_template() -> TemplateMeta:
    """
    2행 번호판 템플릿 (예: 경기79 / 사4711)
    크기: 340x180 픽셀
    구조:
      1행: [지역명2-4자][숫자2자리]
      2행: [한글1자][숫자4자리]
    """
    width, height = 340, 180

    # 1행 슬롯 (지역명 + 숫자)
    region_width = 120
    region_height = 60
    region_x = 20
    region_y = 15

    num_width = 50
    num_height = 60
    num_y = 15

    # 2행 슬롯 (한글 + 숫자)
    char_width = 55
    char_height = 70
    char_y = 100
    start_x = 30
    spacing = 8

    slots = [
        # 1행
        Slot(name="region", x=region_x, y=region_y, w=region_width, h=region_height),
        Slot(name="top_num1", x=160, y=num_y, w=num_width, h=num_height),
        Slot(name="top_num2", x=220, y=num_y, w=num_width, h=num_height),

        # 2행
        Slot(name="hangul", x=start_x, y=char_y, w=char_width, h=char_height),
        Slot(name="num1", x=start_x + (char_width + spacing) * 1, y=char_y, w=char_width, h=char_height),
        Slot(name="num2", x=start_x + (char_width + spacing) * 2, y=char_y, w=char_width, h=char_height),
        Slot(name="num3", x=start_x + (char_width + spacing) * 3, y=char_y, w=char_width, h=char_height),
        Slot(name="num4", x=start_x + (char_width + spacing) * 4, y=char_y, w=char_width, h=char_height),
    ]

    return TemplateMeta(
        plate_type="TWO_LINE",
        size=(width, height),
        slots=slots
    )


def get_two_line_small_template() -> TemplateMeta:
    """
    2행 소형 번호판 템플릿 (예: 02노 / 3454)
    크기: 280x260 픽셀
    종횡비가 1.1 정도로 정사각형에 가까움

    패턴: 상단(2자리 숫자 + 한글) / 하단(4자리 숫자)
    예시: 02노3454, 01고8109
    """
    width, height = 280, 260

    char_width = 50
    char_height = 65
    spacing = 6

    # 1행 (숫자 2자리 + 한글) - 지역명 없음
    top_y = 20
    top_start_x = 40  # 3개 문자를 중앙 정렬

    # 2행 (숫자 4자리)
    bottom_y = 150
    bottom_start_x = 25

    slots = [
        # 1행 (지역명 없음, 숫자 2자리 + 한글만)
        Slot(name="top_num1", x=top_start_x + (char_width + spacing) * 0, y=top_y, w=char_width, h=char_height),
        Slot(name="top_num2", x=top_start_x + (char_width + spacing) * 1, y=top_y, w=char_width, h=char_height),
        Slot(name="hangul", x=top_start_x + (char_width + spacing) * 2, y=top_y, w=char_width, h=char_height),

        # 2행
        Slot(name="num1", x=bottom_start_x + (char_width + spacing) * 0, y=bottom_y, w=char_width, h=char_height),
        Slot(name="num2", x=bottom_start_x + (char_width + spacing) * 1, y=bottom_y, w=char_width, h=char_height),
        Slot(name="num3", x=bottom_start_x + (char_width + spacing) * 2, y=bottom_y, w=char_width, h=char_height),
        Slot(name="num4", x=bottom_start_x + (char_width + spacing) * 3, y=bottom_y, w=char_width, h=char_height),
    ]

    return TemplateMeta(
        plate_type="TWO_LINE_SMALL",
        size=(width, height),
        slots=slots
    )


def get_motorcycle_template() -> TemplateMeta:
    """
    이륜차 번호판 템플릿 (예: 서울 / 가1234)
    크기: 200x220 픽셀
    세로로 긴 형태
    """
    width, height = 200, 220

    # 1행 (지역명)
    region_width = 140
    region_height = 70
    region_x = 30
    region_y = 20

    # 2행 (한글 + 숫자)
    char_width = 38
    char_height = 60
    bottom_y = 130
    spacing = 5
    start_x = 10

    slots = [
        # 1행
        Slot(name="region", x=region_x, y=region_y, w=region_width, h=region_height),

        # 2행
        Slot(name="hangul", x=start_x, y=bottom_y, w=char_width, h=char_height),
        Slot(name="num1", x=start_x + (char_width + spacing) * 1, y=bottom_y, w=char_width, h=char_height),
        Slot(name="num2", x=start_x + (char_width + spacing) * 2, y=bottom_y, w=char_width, h=char_height),
        Slot(name="num3", x=start_x + (char_width + spacing) * 3, y=bottom_y, w=char_width, h=char_height),
        Slot(name="num4", x=start_x + (char_width + spacing) * 4, y=bottom_y, w=char_width, h=char_height),
    ]

    return TemplateMeta(
        plate_type="MOTORCYCLE",
        size=(width, height),
        slots=slots
    )


def detect_plate_corners(image: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, float]:
    """
    번호판 모서리 검출

    Args:
        image: 입력 이미지 (번호판 ROI)
        debug: 디버그 모드 활성화

    Returns:
        corners: 4개의 모서리 좌표 (좌상, 우상, 우하, 좌하) - shape (4, 2)
        confidence: 모서리 검출 신뢰도 (0.0~1.0)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 1. 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Canny 엣지 검출
    edges = cv2.Canny(blurred, 50, 150)

    # 3. 윤곽선 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # 실패 시 이미지 경계를 모서리로 사용
        h, w = image.shape[:2]
        return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32), 0.0

    # 4. 가장 큰 윤곽선 찾기
    largest_contour = max(contours, key=cv2.contourArea)

    # 5. 윤곽선을 사각형으로 근사
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 6. 사각형이 아니면 최소 외접 사각형 사용
    if len(approx) != 4:
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        approx = box.reshape(-1, 1, 2).astype(np.int32)
        confidence = 0.5
    else:
        confidence = 1.0

    # 7. 좌표 정렬 (좌상, 우상, 우하, 좌하)
    corners = approx.reshape(-1, 2).astype(np.float32)
    corners = order_points(corners)

    if debug:
        debug_img = image.copy()
        cv2.drawContours(debug_img, [corners.astype(np.int32)], -1, (0, 255, 0), 2)
        cv2.imshow("Detected Corners", debug_img)
        cv2.waitKey(0)

    return corners, confidence


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    4개의 점을 (좌상, 우상, 우하, 좌하) 순서로 정렬

    Args:
        pts: 4개의 점 좌표 (4, 2)

    Returns:
        ordered: 정렬된 좌표 (4, 2)
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # 합이 가장 작은 점 = 좌상, 가장 큰 점 = 우하
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 차이가 가장 작은 점 = 우상, 가장 큰 점 = 좌하
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def select_template(width: float, height: float) -> TemplateMeta:
    """
    번호판 크기 비율을 기반으로 템플릿 선택

    Args:
        width: 번호판 너비
        height: 번호판 높이

    Returns:
        template_meta: 선택된 템플릿 메타데이터
    """
    if width == 0 or height == 0:
        # 기본값
        return get_one_line_template()

    ratio = width / height

    # 종횡비 기준
    # 1행: ~4.7 (520/110 = 4.73)
    # 2행: ~1.9 (340/180 = 1.89)
    # 소형 2행: ~1.1 (280/260 = 1.08)
    # 이륜차: ~0.9 (200/220 = 0.91) - 세로로 긴 형태

    if ratio > 3.5:
        # 1행 번호판 (가로로 길다)
        return get_one_line_template()
    elif ratio > 1.5:
        # 일반 2행 번호판
        return get_two_line_template()
    elif ratio > 1.0:
        # 소형 2행 번호판
        return get_two_line_small_template()
    else:
        # 이륜차 번호판 (세로로 길다)
        return get_motorcycle_template()


def warp_plate(image: np.ndarray, corners: np.ndarray, template_meta: TemplateMeta) -> np.ndarray:
    """
    번호판 이미지를 템플릿 크기로 원근 변환

    Args:
        image: 입력 이미지
        corners: 4개의 모서리 좌표 (4, 2)
        template_meta: 템플릿 메타데이터

    Returns:
        warped: 원근 변환된 이미지
    """
    width, height = template_meta.size

    # 목표 좌표 (템플릿 크기의 직사각형)
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # 원근 변환 행렬 계산
    matrix = cv2.getPerspectiveTransform(corners, dst_points)

    # 원근 변환 적용
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped


def process_plate_warp(image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None,
                       debug: bool = False) -> Tuple[np.ndarray, TemplateMeta]:
    """
    번호판 이미지 전체 워프 프로세스

    Args:
        image: 입력 이미지 (번호판 ROI)
        bbox: 바운딩 박스 (x1, y1, x2, y2) - None이면 전체 이미지 사용
        debug: 디버그 모드

    Returns:
        warped_image: 워프된 이미지
        template_meta: 템플릿 메타데이터 (corners_confidence 포함)
    """
    # 1. 모서리 검출
    corners, confidence = detect_plate_corners(image, debug=debug)

    # 2. 템플릿 선택
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
    else:
        height, width = image.shape[:2]

    template_meta = select_template(width, height)
    template_meta.corners_confidence = confidence

    # 3. 원근 변환
    warped = warp_plate(image, corners, template_meta)

    return warped, template_meta

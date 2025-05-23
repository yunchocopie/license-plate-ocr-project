
import cv2
import numpy as np
import math

class PerspectiveCorrection:
    def __init__(self):
        pass

    def correct(self, image): # 입력은 그레이스케일 이미지로 가정
        if image is None or image.size == 0:
            return image

        # 입력 이미지가 uint8이 아니면 변환
        if image.dtype != np.uint8:
            img_to_process = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_to_process = image.copy()

        # 이진화: 번호판 영역을 잘 찾기 위함.
        # 번호판 글자색에 따라 THRESH_BINARY 또는 THRESH_BINARY_INV 사용
        # 현재 이미지는 녹색 배경에 흰색 글자 -> 글자가 흰색(255)이 되도록
        # Adaptive thresholding이 더 강인할 수 있음
        binary = cv2.adaptiveThreshold(img_to_process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 5) # block_size, C 값 조절 가능
        # _, binary = cv2.threshold(img_to_process, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image # 원본 그레이스케일 이미지 반환

        max_contour = max(contours, key=cv2.contourArea)
        img_area = img_to_process.shape[0] * img_to_process.shape[1]

        # 윤곽선 면적이 너무 작거나, 이미지의 대부분을 차지하면(잘못된 윤곽선일 가능성) 원본 반환
        if not (0.01 < cv2.contourArea(max_contour) / img_area < 0.85):
            return image

        rect = cv2.minAreaRect(max_contour)
        angle = rect[2]
        box_points = cv2.boxPoints(rect)
        box = np.intp(box_points) # np.int0 대신 np.intp 사용 (플랫폼 의존적이지 않은 정수형)


        # 회전 각도가 매우 작으면 (예: +/- 2도 이내) 회전 및 원근 변환 생략 고려
        # 또는 회전만 수행하고 원근 변환은 건너뛰기
        rotated_image = img_to_process # 초기값은 원본 그레이스케일

        # 회전 보정 (극단적인 각도가 아니면 수행)
        # OpenCV의 각도는 -90 ~ 0 범위. 세워진 직사각형이면 0에 가까움. 눕혀진 직사각형이면 -90에 가까움.
        # 우리가 원하는 것은 글자가 수평이 되는 것이므로, 각도가 -45보다 작으면 (즉, 더 많이 누워있으면) 90을 더함.
        # 각도가 절대값 2도 이상일 때만 회전 수행
        if abs(angle) > 2.0 and abs(angle - 90.0) > 2.0 : # 이미 거의 수평/수직인 경우 제외
            if angle < -45.0:
                angle += 90.0
            

        dst_pts = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        # 원근 변환 행렬 계산 및 적용 (회전된 그레이스케일 이미지에 적용)
        try:
            M = cv2.getPerspectiveTransform(src_pts_ordered, dst_pts)
            warped = cv2.warpPerspective(rotated_image, M, (max_width, max_height),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=255) # 배경 흰색
            return warped
        except Exception as e:
            print(f"Error in getPerspectiveTransform or warpPerspective: {e}")
            return rotated_image


    def order_points(self, pts):
        # (기존 order_points 함수는 대부분 잘 작동하지만, 극단적인 경우 오류 가능성 있음)
        # x 좌표와 y 좌표를 기준으로 정렬하는 더 강인한 방법 사용 가능
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left
        return rect

    # detect_rotation_angle 함수는 현재 직접 사용되지 않으므로 생략하거나 유지 가능
import cv2
import numpy as np
import config

class Normalize:
    def __init__(self, target_size=None):
        self.target_size = target_size or config.PLATE_SIZE # (width, height)

    def normalize(self, image): # 입력은 그레이스케일 이미지로 가정
        if image is None or image.size == 0:
            # target_size는 (width, height) 이므로, numpy 배열 생성 시 (height, width) 사용
            return np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)

        # 이미지가 uint8이 아니면 변환
        if image.dtype != np.uint8:
            norm_image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            norm_image = image.copy()

        # 현재 이미지 크기
        h, w = norm_image.shape[:2]
        target_w, target_h = self.target_size

        if h == 0 or w == 0: # 입력 이미지 크기가 0이면 빈 이미지 반환
             return np.zeros((target_h, target_w), dtype=np.uint8)


        # 비율에 따라 크기 조정
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # OpenCV 리사이즈는 (width, height) 순서
        try:
            resized = cv2.resize(norm_image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4)
        except cv2.error as e:
            print(f"Error resizing image: {e}. Original shape: {(h,w)}, Target new shape: {(new_h, new_w)}")
            return np.zeros((target_h, target_w), dtype=np.uint8)


        # 패딩 추가하여 목표 크기 맞춤
        # 패딩 색상은 EasyOCR이 글자를 잘 인식하도록. 일반적으로 흰색(255) 또는 검은색(0).
        # 번호판 배경과 대비되는 색 또는 중립적인 회색(128)도 고려 가능.
        # 현재는 흰색(255) 사용.
        pad_color = 255
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        normalized = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=pad_color)
        return normalized

    # adaptive_normalize 와 remove_border 함수는 필요에 따라 유지 또는 수정
    # adaptive_normalize는 현재 코드에서 사용되지 않는 것으로 보임

"""
데이터 증강 파이프라인

번호판 이미지를 다양한 조건으로 증강하여 학습 데이터 확보
- 밝기 조정
- 반사/글레어
- 야간 조건
- 흐림
- 노이즈
- 회전/변형
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import random


class PlateAugmentation:
    """번호판 이미지 증강 클래스"""

    def __init__(self, seed=None):
        """
        증강 클래스 초기화

        Args:
            seed: 랜덤 시드 (재현성을 위해)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def adjust_brightness(self, image: np.ndarray, factor: float = None) -> np.ndarray:
        """
        밝기 조정

        Args:
            image: 입력 이미지
            factor: 밝기 계수 (0.5~2.0, None이면 랜덤)

        Returns:
            밝기 조정된 이미지
        """
        if factor is None:
            factor = random.uniform(0.5, 2.0)

        # HSV로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        hsv = hsv.astype(np.uint8)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def add_glare(self, image: np.ndarray, intensity: float = None) -> np.ndarray:
        """
        반사/글레어 추가

        Args:
            image: 입력 이미지
            intensity: 반사 강도 (0.0~1.0, None이면 랜덤)

        Returns:
            반사가 추가된 이미지
        """
        if intensity is None:
            intensity = random.uniform(0.3, 0.8)

        h, w = image.shape[:2]

        # 랜덤 위치에 타원형 반사 생성
        center_x = random.randint(int(w * 0.2), int(w * 0.8))
        center_y = random.randint(int(h * 0.2), int(h * 0.8))
        axes = (random.randint(w // 4, w // 2), random.randint(h // 4, h // 2))

        # 반사 마스크 생성
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 1, -1)

        # 가우시안 블러로 부드럽게
        mask = cv2.GaussianBlur(mask, (51, 51), 20)

        # 반사 적용
        result = image.astype(np.float32)
        white = np.ones_like(result) * 255
        result = result * (1 - mask[:, :, np.newaxis] * intensity) + white * mask[:, :, np.newaxis] * intensity

        return np.clip(result, 0, 255).astype(np.uint8)

    def simulate_night(self, image: np.ndarray, darkness: float = None) -> np.ndarray:
        """
        야간 조건 시뮬레이션

        Args:
            image: 입력 이미지
            darkness: 어둠 정도 (0.0~1.0, None이면 랜덤)

        Returns:
            야간 조건이 적용된 이미지
        """
        if darkness is None:
            darkness = random.uniform(0.4, 0.8)

        # 전체 밝기 감소
        result = self.adjust_brightness(image, factor=1.0 - darkness)

        # 노이즈 추가 (저조도 노이즈)
        noise = np.random.normal(0, 10 * darkness, image.shape).astype(np.float32)
        result = result.astype(np.float32) + noise
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def add_blur(self, image: np.ndarray, kernel_size: int = None) -> np.ndarray:
        """
        흐림 추가

        Args:
            image: 입력 이미지
            kernel_size: 블러 커널 크기 (None이면 랜덤)

        Returns:
            흐림이 추가된 이미지
        """
        if kernel_size is None:
            kernel_size = random.choice([3, 5, 7, 9])

        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def add_noise(self, image: np.ndarray, noise_type: str = 'gaussian') -> np.ndarray:
        """
        노이즈 추가

        Args:
            image: 입력 이미지
            noise_type: 노이즈 타입 ('gaussian', 'salt_pepper')

        Returns:
            노이즈가 추가된 이미지
        """
        if noise_type == 'gaussian':
            mean = 0
            std = random.uniform(5, 20)
            noise = np.random.normal(mean, std, image.shape).astype(np.float32)
            result = image.astype(np.float32) + noise
            return np.clip(result, 0, 255).astype(np.uint8)

        elif noise_type == 'salt_pepper':
            result = image.copy()
            prob = random.uniform(0.01, 0.05)

            # Salt
            salt_mask = np.random.random(image.shape[:2]) < prob / 2
            result[salt_mask] = 255

            # Pepper
            pepper_mask = np.random.random(image.shape[:2]) < prob / 2
            result[pepper_mask] = 0

            return result

        return image

    def rotate(self, image: np.ndarray, angle: float = None) -> np.ndarray:
        """
        회전

        Args:
            image: 입력 이미지
            angle: 회전 각도 (None이면 랜덤 -15~15도)

        Returns:
            회전된 이미지
        """
        if angle is None:
            angle = random.uniform(-15, 15)

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # 회전 행렬
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 회전 적용
        result = cv2.warpAffine(image, M, (w, h),
                               borderMode=cv2.BORDER_REPLICATE)

        return result

    def perspective_transform(self, image: np.ndarray, intensity: float = None) -> np.ndarray:
        """
        원근 변환

        Args:
            image: 입력 이미지
            intensity: 변형 강도 (0.0~1.0, None이면 랜덤)

        Returns:
            원근 변환된 이미지
        """
        if intensity is None:
            intensity = random.uniform(0.1, 0.3)

        h, w = image.shape[:2]

        # 소스 포인트 (원본 코너)
        src_points = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ])

        # 대상 포인트 (변형된 코너)
        offset = int(min(w, h) * intensity)
        dst_points = np.float32([
            [random.randint(0, offset), random.randint(0, offset)],
            [w - 1 - random.randint(0, offset), random.randint(0, offset)],
            [w - 1 - random.randint(0, offset), h - 1 - random.randint(0, offset)],
            [random.randint(0, offset), h - 1 - random.randint(0, offset)]
        ])

        # 변환 행렬
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # 변환 적용
        result = cv2.warpPerspective(image, M, (w, h),
                                     borderMode=cv2.BORDER_REPLICATE)

        return result

    def augment_random(self, image: np.ndarray, num_augmentations: int = 3) -> np.ndarray:
        """
        랜덤 증강 조합

        Args:
            image: 입력 이미지
            num_augmentations: 적용할 증강 개수

        Returns:
            증강된 이미지
        """
        augmentation_funcs = [
            lambda img: self.adjust_brightness(img),
            lambda img: self.add_glare(img),
            lambda img: self.simulate_night(img),
            lambda img: self.add_blur(img),
            lambda img: self.add_noise(img, 'gaussian'),
            lambda img: self.add_noise(img, 'salt_pepper'),
            lambda img: self.rotate(img),
            lambda img: self.perspective_transform(img)
        ]

        # 랜덤하게 선택
        selected_funcs = random.sample(augmentation_funcs, min(num_augmentations, len(augmentation_funcs)))

        result = image.copy()
        for func in selected_funcs:
            result = func(result)

        return result

    def augment_pipeline(self, image: np.ndarray, config: Dict) -> np.ndarray:
        """
        설정 기반 증강 파이프라인

        Args:
            image: 입력 이미지
            config: 증강 설정 딕셔너리

        Returns:
            증강된 이미지
        """
        result = image.copy()

        if config.get('brightness', False):
            result = self.adjust_brightness(result, config.get('brightness_factor'))

        if config.get('glare', False):
            result = self.add_glare(result, config.get('glare_intensity'))

        if config.get('night', False):
            result = self.simulate_night(result, config.get('darkness'))

        if config.get('blur', False):
            result = self.add_blur(result, config.get('blur_kernel'))

        if config.get('noise', False):
            result = self.add_noise(result, config.get('noise_type', 'gaussian'))

        if config.get('rotate', False):
            result = self.rotate(result, config.get('rotation_angle'))

        if config.get('perspective', False):
            result = self.perspective_transform(result, config.get('perspective_intensity'))

        return result


# 미리 정의된 증강 프리셋
AUGMENTATION_PRESETS = {
    'light': {
        'brightness': True, 'brightness_factor': None,
        'blur': True, 'blur_kernel': 3
    },
    'medium': {
        'brightness': True, 'brightness_factor': None,
        'glare': True, 'glare_intensity': 0.4,
        'blur': True, 'blur_kernel': 5,
        'rotate': True, 'rotation_angle': None
    },
    'heavy': {
        'brightness': True, 'brightness_factor': None,
        'glare': True, 'glare_intensity': None,
        'night': True, 'darkness': 0.6,
        'blur': True, 'blur_kernel': 7,
        'noise': True, 'noise_type': 'gaussian',
        'rotate': True, 'rotation_angle': None,
        'perspective': True, 'perspective_intensity': 0.2
    },
    'night': {
        'night': True, 'darkness': None,
        'blur': True, 'blur_kernel': 5,
        'noise': True, 'noise_type': 'gaussian'
    },
    'glare_only': {
        'glare': True, 'glare_intensity': None
    }
}

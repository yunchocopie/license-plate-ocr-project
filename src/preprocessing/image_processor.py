import cv2
import numpy as np
from .blur_correction import BlurCorrection
from .perspective import PerspectiveCorrection
from .normalize import Normalize
from ..utils.metrics import assess_image_quality, determine_processing_level
import config

"""
이미지 기본 처리 모듈

이 모듈은 번호판 이미지 전처리의 주요 클래스를 제공합니다.
흐림 보정, 기울기 보정, 비율 정규화 등의 작업을 순차적으로 적용합니다.

1단계 개선: 이미지 품질 평가 기반 적응형 전처리 추가
- 이미지 품질을 자동으로 평가하여 적절한 처리 수준 결정
- 고품질 이미지는 최소 처리, 저품질 이미지는 전체 처리 적용
"""

class ImageProcessor:
    """번호판 이미지 전처리를 위한 클래스"""

    def __init__(self):
        """ImageProcessor 클래스 초기화"""
        self.blur_corrector = BlurCorrection(
            kernel_size=config.BLUR_KERNEL_SIZE,
            sigma=config.BLUR_SIGMA
        )
        self.perspective_corrector = PerspectiveCorrection()

        Args:
            image (numpy.ndarray): BGR 형식의 원본 번호판 이미지
            perform_denoising (bool): 노이즈 제거 수행 여부
            perform_blur_correction (bool): 흐림 보정 수행 여부
            perform_perspective_correction (bool): 원근 보정 수행 여부
            perform_normalization (bool): 정규화 수행 여부
            perform_enhancement (bool): 대비 향상 수행 여부

        Returns:

        """
        if image is None or image.size == 0:
            print("Warning: Input image to ImageProcessor is empty.")
            # 빈 이미지를 반환하기보다, 에러를 발생시키거나 기본 이미지를 반환하는 것이 좋을 수 있습니다.
            # 여기서는 config.PLATE_SIZE에 맞춰 빈 이미지를 생성합니다.
            return np.zeros(config.PLATE_SIZE[::-1], dtype=np.uint8) # (height, width)

        processed_img = image.copy()

        # 1. 그레이스케일 변환
        if len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        elif len(processed_img.shape) == 2:
            gray = processed_img.copy()

    def visualize_steps(self, image):
        steps = {'original': image.copy()}

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        steps['gray'] = gray.copy()

        denoised = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
        steps['denoised'] = denoised.copy()

        # 흐림 보정 (약하게 또는 선택적으로)
        # deblurred = self.blur_corrector.correct(denoised.copy())
        # steps['deblurred'] = deblurred.copy()
        # current_for_blur = denoised # 만약 deblurred를 다음 단계에 쓴다면 current_for_blur = deblurred

        # 원근 보정 (선택적으로)
        # warped = self.perspective_corrector.correct(current_for_blur.copy()) # 이전 단계 결과 사용
        # steps['warped'] = warped.copy()
        # current_for_normalize = warped

        # 여기서는 단순화된 파이프라인으로 시각화
        # 원근/흐림보정은 기본 비활성화이므로, 이 시각화에서는 생략하거나 조건부로 추가
        normalized = self.normalizer.normalize(denoised.copy()) # denoised 결과를 정규화
        steps['normalized'] = normalized.copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


import cv2
import numpy as np
from .blur_correction import BlurCorrection
from .perspective import PerspectiveCorrection
from .normalize import Normalize
import config

"""
이미지 기본 처리 모듈

이 모듈은 번호판 이미지 전처리의 주요 클래스를 제공합니다.
흐림 보정, 기울기 보정, 비율 정규화 등의 작업을 순차적으로 적용합니다.
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
        self.normalizer = Normalize(target_size=config.PLATE_SIZE)

    def process(self, image,
                perform_denoising=True,
                perform_blur_correction=False, # 기본적으로 비활성화 (현재 이미지에는 불필요할 수 있음)
                perform_perspective_correction=False, # 기본적으로 비활성화 (현재 이미지에는 불필요할 수 있음)
                perform_normalization=True,
                perform_enhancement=True):
        """
        번호판 이미지 전처리 실행

        Args:
            image (numpy.ndarray): BGR 형식의 원본 번호판 이미지
            perform_denoising (bool): 노이즈 제거 수행 여부
            perform_blur_correction (bool): 흐림 보정 수행 여부
            perform_perspective_correction (bool): 원근 보정 수행 여부
            perform_normalization (bool): 정규화 수행 여부
            perform_enhancement (bool): 대비 향상 수행 여부

        Returns:
            numpy.ndarray: 전처리된 그레이스케일 번호판 이미지 (EasyOCR 입력용)
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
        else:
            print(f"Warning: Unexpected image shape {processed_img.shape} for grayscale conversion.")
            # 기본적으로 첫 채널 사용 시도 또는 에러 처리
            if len(processed_img.shape) == 3 and processed_img.shape[2] == 1:
                 gray = processed_img[:,:,0]
            else: # 그래도 모르겠으면 일단 원본으로
                 gray = processed_img


        current_img = gray

        # 2. 노이즈 제거 (필요한 경우에만, 약하게)
        if perform_denoising:
            # fastNlMeansDenoising의 h 파라미터는 노이즈 제거 강도입니다. 너무 높으면 디테일 손실.
            current_img = cv2.fastNlMeansDenoising(current_img, None, h=5, templateWindowSize=7, searchWindowSize=21)

        # 3. 흐림 보정 (선택적, 현재 이미지에는 불필요)
        if perform_blur_correction:
            # BlurCorrection.correct 메서드가 그레이스케일 이미지를 받도록 수정되었다고 가정
            current_img = self.blur_corrector.correct(current_img)

        # 4. 기울기/원근 보정 (선택적, 현재 이미지에는 불필요)
        # PerspectiveCorrection.correct는 내부적으로 이진화를 할 수 있으나,
        # 그레이스케일 이미지를 받아 처리하고 그레이스케일 결과를 반환하도록 하는 것이 좋음.
        # 또는 여기서 이진화 후 전달 -> 결과도 이진화 이미지.
        # 현재는 그레이스케일 이미지를 그대로 전달한다고 가정.
        if perform_perspective_correction:
            corrected_perspective = self.perspective_corrector.correct(current_img.copy()) # 복사본 전달
            # 보정 결과가 유효한 경우에만 업데이트 (예: 너무 작아지지 않은 경우)
            if corrected_perspective.shape[0] > 10 and corrected_perspective.shape[1] > 30:
                current_img = corrected_perspective


        # 5. 비율 정규화 (EasyOCR에 적절한 크기로)
        if perform_normalization:
            # Normalize.normalize는 그레이스케일 이미지를 입력으로 받고,
            # 정규화된 그레이스케일 이미지 반환
            current_img = self.normalizer.normalize(current_img)

        # 6. 최종 대비 향상 (CLAHE) - EasyOCR 성능 향상에 도움
        if perform_enhancement:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            current_img = clahe.apply(current_img)

        # 7. EasyOCR은 uint8 타입의 이미지를 선호합니다.
        if current_img.dtype != np.uint8:
            print(f"Warning: Image dtype is {current_img.dtype}, converting to uint8.")
            if np.max(current_img) <= 1.0 and (current_img.dtype == np.float32 or current_img.dtype == np.float64) : # 0-1 범위 float
                current_img = (current_img * 255).astype(np.uint8)
            else: # 다른 float 범위 또는 다른 타입
                current_img = np.clip(current_img, 0, 255).astype(np.uint8)

        # 최종적으로 그레이스케일 이미지를 반환 (EasyOCR은 그레이스케일 이미지도 잘 처리)
        # 만약 특정 상황에서 이진화된 이미지가 더 좋다면 여기서 이진화:
        # _, final_image = cv2.threshold(current_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # return final_image
        return current_img

    def apply_individual(self, image, blur=False, perspective=False, normalize=True, enhance=True):
        # ... (이 메서드는 process 메서드의 플래그를 사용하여 유사하게 구현 가능)
        return self.process(image,
                            perform_denoising=True, # 기본적인 노이즈제거는 하는것이 좋음
                            perform_blur_correction=blur,
                            perform_perspective_correction=perspective,
                            perform_normalization=normalize,
                            perform_enhancement=enhance)


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
        enhanced = clahe.apply(normalized.copy())
        steps['enhanced_clahe'] = enhanced.copy()

        # 최종 결과 (process 메서드의 기본 설정과 유사하게)
        final_processed_image = self.process(image.copy(), perform_blur_correction=False, perform_perspective_correction=False)
        steps['final_easyocr_input'] = final_processed_image

        return steps
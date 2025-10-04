import cv2
import numpy as np
from .blur_correction import BlurCorrection
from .perspective import PerspectiveCorrection
from .normalize import Normalize
from .advanced_enhancement import AdvancedImageProcessor
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
        self.advanced_processor = AdvancedImageProcessor()  # 고급 전처리 프로세서
        
        # 번호판 타입별 최적화 설정
        self.plate_type_configs = {
            'general': {
                'target_size': config.PLATE_SIZES.get('general', (320, 80)),
                'enhancement_strength': 1.2,  # 적당한 대비 향상
                'noise_reduction': 'light',   # 가벼운 노이즈 제거
                'sharpening': True,          # 샤프닝 적용
                'char_separation': True      # 문자 분리 최적화
            },
            'general_3digit': {
                'target_size': config.PLATE_SIZES.get('general_3digit', (360, 80)),
                'enhancement_strength': 1.3,  # 조금 더 강한 대비
                'noise_reduction': 'medium',  # 중간 노이즈 제거
                'sharpening': True,
                'char_separation': True
            },
            'commercial': {
                'target_size': config.PLATE_SIZES.get('commercial', (400, 80)),
                'enhancement_strength': 1.4,  # 노란 배경 대비 강화
                'noise_reduction': 'medium',
                'sharpening': True,
                'char_separation': False     # 지역명+번호 연결형
            }
        }

    def process(self, image):
        """
        새로운 두 단계 파이프라인으로 번호판 이미지 전처리 실행
        A단계: 번호판 탐지·정합 단계 + B단계: OCR 최적화 단계

        Args:
            image (numpy.ndarray): BGR 형식의 원본 번호판 이미지

        Returns:
            numpy.ndarray: 전처리된 그레이스케일 번호판 이미지 (EasyOCR 입력용)
        """
        if image is None or image.size == 0:
            print("Warning: Input image to ImageProcessor is empty.")
            return np.zeros(config.PLATE_SIZE[::-1], dtype=np.uint8)

        # === A단계: 번호판 탐지·정합 단계 ===

        # A1. 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # A2. 가우시안 블러 (ksize=5×5, σ≈1.0)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

        # A3. 원근 보정 (워핑) - 블러 결과 바로 사용
        perspective_corrected = self.perspective_corrector.correct(blurred.copy())
        if perspective_corrected is not None and perspective_corrected.shape[0] > 10 and perspective_corrected.shape[1] > 30:
            warped_image = perspective_corrected
        else:
            warped_image = blurred.copy()

        # A4. 적응형 임계값 (탐지용) - 원근 보정 후 적용
        adaptive_detection = cv2.adaptiveThreshold(
            warped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 10  # block_size=25, C=10
        )

        # A단계 최종 결과
        a_stage_final = adaptive_detection.copy()

        # === B단계: OCR 최적화 단계 (A단계 최종 결과 사용) ===

        # B1. 미세 가우시안 블러 (3×3) - A단계 최종 결과에 적용
        warped_blur = cv2.GaussianBlur(a_stage_final, (3, 3), 0)

        # B2. 적응형 임계값 (인식용, 더 보수적으로)
        adaptive_ocr = cv2.adaptiveThreshold(
            warped_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8  # block_size=15, C=8 (더 보수적)
        )

        return adaptive_ocr

    def apply_individual(self, image, blur=False, perspective=False, normalize=True, enhance=True):
        """레거시 호환성을 위한 메서드 - 새로운 두 단계 파이프라인 사용"""
        return self.process(image)


    def visualize_steps(self, image):
        steps = {'original': image.copy()}

        # 1. 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Streamlit 표시를 위해 3채널로 변환 (RGB 형태)
            gray_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            gray = image.copy()
            # 이미 그레이스케일인 경우도 3채널로 변환
            if len(gray.shape) == 2:
                gray_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                gray_display = gray.copy()
        steps['gray'] = gray_display

        # 2. 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
        denoised_display = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        steps['denoised'] = denoised_display

        # 3. 흐림 보정 (약하게 또는 선택적으로)
        deblurred = self.blur_corrector.correct(denoised.copy())
        deblurred_display = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2RGB)
        steps['deblurred'] = deblurred_display

        # 4. 원근 보정
        warped = self.perspective_corrector.correct(deblurred.copy())
        # 보정 결과가 유효한지 확인
        if warped is not None and warped.shape[0] > 10 and warped.shape[1] > 30:
            perspective_fixed = warped
        else:
            perspective_fixed = deblurred.copy()
        perspective_display = cv2.cvtColor(perspective_fixed, cv2.COLOR_GRAY2RGB)
        steps['perspective_corrected'] = perspective_display

        # 5. 정규화 (리사이즈)
        normalized = self.normalizer.normalize(denoised.copy()) # denoised 결과를 정규화
        normalized_display = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        steps['normalized'] = normalized_display

        # 6. 대비 향상 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized.copy())
        enhanced_display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        steps['enhanced_clahe'] = enhanced_display

        # 7. 최종 전처리 결과 (EasyOCR 입력용)
        final_processed_image = self.process(image.copy())

        # 최종 결과도 3채널로 변환하여 표시
        final_display = cv2.cvtColor(final_processed_image, cv2.COLOR_GRAY2RGB)
        steps['final_easyocr_input'] = final_display


        return steps

    def visualize_steps_opencv_method(self, image):
        """
        전체 전처리 파이프라인 시각화 (UI 표시용 + OCR 엔진 전처리 포함)
        A단계: 번호판 탐지·정합 단계 + B단계: OCR 최적화 단계 + C단계: OCR 엔진 내부 전처리
        """
        steps = {'original': image.copy()}

        # === A단계: 번호판 탐지·정합 단계 ===

        # A1. 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        steps['gray'] = gray_display

        # A2. 가우시안 블러 (ksize=5×5, σ≈1.0)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        blurred_display = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
        steps['gaussian_blur'] = blurred_display

        # A3. 원근 보정 (워핑) - 블러 결과 바로 사용
        perspective_corrected = self.perspective_corrector.correct(blurred.copy())
        if perspective_corrected is not None and perspective_corrected.shape[0] > 10 and perspective_corrected.shape[1] > 30:
            warped_image = perspective_corrected
        else:
            warped_image = blurred.copy()

        warped_display = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2RGB)
        steps['warped'] = warped_display

        # A4. 적응형 임계값 (탐지용) - 원근 보정 후 적용
        adaptive_detection = cv2.adaptiveThreshold(
            warped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 10  # block_size=25, C=10
        )
        adaptive_detection_display = cv2.cvtColor(adaptive_detection, cv2.COLOR_GRAY2RGB)
        steps['adaptive_detection'] = adaptive_detection_display

        # A5. 윤곽선 검출 → 후보 박스 필터링 (시각화용)
        contours, _ = cv2.findContours(adaptive_detection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 윤곽선이 그려진 이미지 생성
        contour_image = adaptive_detection.copy()
        contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        steps['contours'] = contour_image

        # A단계 최종 결과 = A4 적응형 임계값 결과 (A5는 시각화용)
        a_stage_final = adaptive_detection.copy()

        # === B단계: OCR 최적화 단계 (A단계 최종 결과 사용) ===

        # B1. 미세 가우시안 블러 (3×3) - A단계 최종 결과에 적용
        warped_blur = cv2.GaussianBlur(a_stage_final, (3, 3), 0)
        warped_blur_display = cv2.cvtColor(warped_blur, cv2.COLOR_GRAY2RGB)
        steps['ocr_blur'] = warped_blur_display

        # B2. 적응형 임계값 (인식용, 더 보수적으로) - B단계 최종
        adaptive_ocr = cv2.adaptiveThreshold(
            warped_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8  # block_size=15, C=8 (더 보수적)
        )
        adaptive_ocr_display = cv2.cvtColor(adaptive_ocr, cv2.COLOR_GRAY2RGB)
        steps['adaptive_ocr'] = adaptive_ocr_display

        # === C단계: OCR 엔진 내부 추가 전처리 (process_standard 시뮬레이션) ===

        # C1. 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(adaptive_ocr, None, h=5, templateWindowSize=7, searchWindowSize=21)
        denoised_display = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        steps['ocr_denoised'] = denoised_display

        # C2. 대비 향상 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        enhanced_display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        steps['ocr_enhanced'] = enhanced_display

        # C3. 샤프닝
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        sharpened_display = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        steps['ocr_sharpened'] = sharpened_display

        # C4. 최종 이진화
        binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        binary_display = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        steps['ocr_binary'] = binary_display

        # C5. 모폴로지 연산 (최종)
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned_display = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        steps['ocr_final'] = cleaned_display

        return steps
    
    def process_advanced(self, image, quality_mode: str = 'balanced') -> dict:
        """
        고급 전처리 파이프라인 (PRD 요구사항 적용)
        
        Args:
            image: 입력 번호판 이미지
            quality_mode: 'fast', 'balanced', 'high_quality' 중 선택
            
        Returns:
            dict: 처리된 이미지와 분석 정보
        """
        if image is None or image.size == 0:
            return {
                'processed_image': np.zeros(config.PLATE_SIZE[::-1], dtype=np.uint8),
                'analysis': {},
                'processing_steps': [],
                'quality_metrics': {}
            }
        
        # 품질 모드별 설정
        if quality_mode == 'fast':
            settings = {
                'enable_super_resolution': False,
                'enable_deblurring': True,
                'enable_contrast_enhancement': True,
                'enable_illumination_normalization': True
            }
        elif quality_mode == 'high_quality':
            settings = {
                'enable_super_resolution': True,
                'enable_deblurring': True,
                'enable_contrast_enhancement': True,
                'enable_illumination_normalization': True
            }
        else:  # balanced
            settings = {
                'enable_super_resolution': True,
                'enable_deblurring': True,
                'enable_contrast_enhancement': True,
                'enable_illumination_normalization': False  # 로컬 환경에서 성능 고려
            }
        
        # 고급 전처리 실행
        result = self.advanced_processor.process_advanced(image, **settings)
        
        # 기본 정규화 추가 적용
        if result['processed_image'] is not None:
            final_image = self.normalizer.normalize(result['processed_image'])
            result['processed_image'] = final_image
            
            # 품질 메트릭 계산
            if len(image.shape) == 3:
                original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                original_gray = image
                
            quality_metrics = self.advanced_processor.get_quality_metrics(
                original_gray, final_image
            )
            result['quality_metrics'] = quality_metrics
        
        return result
    
    def auto_enhance(self, image) -> np.ndarray:
        """
        이미지 상태를 자동 분석하여 최적의 전처리 적용
        
        Args:
            image: 입력 이미지
            
        Returns:
            최적화된 이미지
        """
        if image is None or image.size == 0:
            return np.zeros(config.PLATE_SIZE[::-1], dtype=np.uint8)
        
        # 이미지 품질 자동 분석
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 블러 정도 측정
        blur_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 대비 측정
        contrast = np.std(gray)
        
        # 해상도 측정
        h, w = gray.shape
        resolution_score = h * w
        
        # 조건에 따른 모드 선택
        if blur_measure < 50 and contrast < 30:
            # 매우 흐리고 대비가 낮음 -> 고품질 모드
            mode = 'high_quality'
        elif resolution_score < 2000:
            # 저해상도 -> 슈퍼해상도 필요
            mode = 'high_quality'
        elif blur_measure > 200 and contrast > 50:
            # 양호한 품질 -> 빠른 모드
            mode = 'fast'
        else:
            # 일반적인 경우 -> 균형 모드
            mode = 'balanced'
        
        result = self.process_advanced(image, quality_mode=mode)
        return result['processed_image']

    def process_standard(self, image) -> np.ndarray:
        """
        표준 전처리 파이프라인 - 모든 이미지에 동일하게 적용

        Args:
            image: 입력 이미지

        Returns:
            전처리된 이미지
        """
        if image is None or image.size == 0:
            return np.zeros(config.PLATE_SIZE[::-1], dtype=np.uint8)

        # 1. 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. 크기 정규화 (320x80)
        resized = cv2.resize(gray, config.PLATE_SIZE, interpolation=cv2.INTER_CUBIC)

        # 3. 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(resized, None, h=5, templateWindowSize=7, searchWindowSize=21)

        # 4. 대비 향상 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 5. 샤프닝
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # 6. 적응형 이진화 (추가 단계로 OCR 성능 향상)
        binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

        # 7. 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 8. 다시 3채널로 변환 (EasyOCR 호환성을 위해)
        if len(cleaned.shape) == 2:
            final_image = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        else:
            final_image = cleaned

        return final_image
    
    def process_for_plate_type(self, image, plate_type='general'):
        """
        번호판 타입에 최적화된 전처리 (제공된 이미지 구조 기반)
        
        Args:
            image: 입력 번호판 이미지
            plate_type: 'general', 'general_3digit', 'commercial' 등
            
        Returns:
            전처리된 이미지
        """
        if image is None or image.size == 0:
            return np.zeros((80, 320), dtype=np.uint8)
        
        # 타입별 설정 가져오기
        config_key = plate_type if plate_type in self.plate_type_configs else 'general'
        type_config = self.plate_type_configs[config_key]
        
        # 1. 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 2. 타입별 크기 조정
        target_size = type_config['target_size']
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_CUBIC)
        
        # 3. 노이즈 제거 (타입별 강도)
        noise_level = type_config['noise_reduction']
        if noise_level == 'light':
            denoised = cv2.fastNlMeansDenoising(resized, None, h=3, templateWindowSize=7, searchWindowSize=21)
        elif noise_level == 'medium':
            denoised = cv2.fastNlMeansDenoising(resized, None, h=5, templateWindowSize=7, searchWindowSize=21)
        else:
            denoised = resized.copy()
        
        # 4. 대비 향상 (타입별 강도)
        strength = type_config['enhancement_strength']
        enhanced = cv2.convertScaleAbs(denoised, alpha=strength, beta=10)
        
        # 5. 적응형 히스토그램 평활화 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(enhanced)
        
        # 6. 샤프닝 (선택적)
        if type_config.get('sharpening', False):
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(equalized, -1, kernel)
            # 샤프닝 강도 조절
            result = cv2.addWeighted(equalized, 0.7, sharpened, 0.3, 0)
        else:
            result = equalized
        
        # 7. 문자 분리 최적화 (일반 번호판용)
        if type_config.get('char_separation', False):
            # 모폴로지 연산으로 문자 분리 개선
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        
        return result
    
    def optimize_for_korean_chars(self, image):
        """
        한국어 문자 인식에 특화된 전처리
        (마, 바, 루, 아, 하 등 제공된 이미지의 한글 최적화)
        """
        if image is None or image.size == 0:
            return np.zeros((80, 320), dtype=np.uint8)
        
        # 1. 기본 전처리
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 2. 한글 문자 특성 고려한 이진화
        # 적응형 임계값으로 다양한 조명 조건에 대응
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 3. 한글 획 두께 정규화
        # 모폴로지 연산으로 한글 특성 강화
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        
        # 4. 작은 노이즈 제거
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        # 5. 한글 문자 간격 최적화
        # 세로 방향 연결 강화 (한글 특성)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        result = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, vertical_kernel)
        
        return result
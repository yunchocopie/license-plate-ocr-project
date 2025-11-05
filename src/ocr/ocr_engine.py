import cv2
import numpy as np
# import easyocr  # 백엔드로 이동
from .backends import EasyOCRBackend, OCRBackend
import torch
import os
import uuid
from pathlib import Path
from .text_postprocess import TextPostProcessor # 상대 경로 유지
from .korean_plate_postprocessor import KoreanPlatePostProcessor # 한국 번호판 전용 후처리기
from ..classification.plate_classifier import PlateClassifier # 번호판 분류기 추가
from ..preprocessing.image_processor import ImageProcessor # 고급 이미지 전처리 추가
from ..utils.logger import setup_logger
import config # config 파일 임포트

# 로거 설정
logger = setup_logger(__name__)

class OCREngine:
    def __init__(self, languages=None, gpu=None, allowed_chars=None, model_storage_directory=None, download_enabled=None):
        self.languages = languages if languages is not None else config.OCR_LANGUAGES
        self.gpu = gpu if gpu is not None else config.OCR_GPU

        # GPU 사용 여부 (백엔드에서 실제 확인 수행)
        # 여기서는 설정만 저장

        # EasyOCR은 문자열 리스트를 허용 문자로 받음
        self.allowed_chars = allowed_chars if allowed_chars is not None else config.OCR_ALLOWED_CHARS
        self.model_storage_directory = model_storage_directory if model_storage_directory is not None else config.MODEL_DIR
        self.download_enabled = False  # 로컬 모델 사용을 위해 다운로드 비활성화

        # 모델 파일 확인은 백엔드에서 처리

        # OCR 백엔드 초기화 (기본: EasyOCR)
        backend_type = getattr(config, 'OCR_BACKEND', 'easyocr').lower()
        
        if backend_type == 'easyocr':
            self.backend = EasyOCRBackend(
                languages=self.languages,
                gpu=self.gpu,
                model_storage_directory=self.model_storage_directory,
                download_enabled=self.download_enabled,
                allowed_chars=self.allowed_chars
            )
            logger.info(f"OCR 백엔드 초기화: EasyOCR")
        else:
            raise ValueError(f"지원하지 않는 OCR 백엔드: {backend_type}")
        
        # 하위 호환성을 위한 reader 속성
        self.reader = self.backend.reader if hasattr(self.backend, 'reader') else None
        self.post_processor = TextPostProcessor(allowed_chars=self.allowed_chars)
        self.korean_postprocessor = KoreanPlatePostProcessor()  # 한국 번호판 전용 후처리기
        self.plate_classifier = PlateClassifier()  # 번호판 분류기 초기화
        self.image_processor = ImageProcessor()  # 고급 이미지 전처리기 초기화

    def recognize_with_confidence(self, image, min_confidence=None, use_char_segmentation=None, plate_type='general'):
        """
        OCR 인식 (신뢰도 포함)

        Args:
            image: 입력 이미지
            min_confidence: 최소 신뢰도
            use_char_segmentation: Contour 방식 사용 여부 (None이면 적응형)
            plate_type: 번호판 타입

        Returns:
            tuple: (인식된 텍스트, 신뢰도)
        """
        min_confidence = min_confidence if min_confidence is not None else config.MIN_OCR_CONFIDENCE

        if image is None or image.size == 0:
            return "", 0.0

        # 이미지 크기 검증 및 전처리
        height, width = image.shape[:2]
        print(f"입력 이미지 크기: {width}x{height}")

        # 너무 작은 이미지는 확대
        if width < 100 or height < 30:
            print("이미지가 너무 작습니다. 확대 처리중...")
            scale_factor = max(100/width, 30/height) * 2  # 2배 추가 확대
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            print(f"확대 후 크기: {new_width}x{new_height}")

        # 전처리 방식 선택
        if use_char_segmentation is None:
            # 적응형: 이미지 품질에 따라 자동 선택
            if config.ENABLE_CHAR_SEGMENTATION:
                result = self.image_processor.process_adaptive(image, plate_type=plate_type)
                processed_image = result['processed_image']
                print(f"적응형 전처리: {result['method']} 방식 선택됨")
            else:
                processed_image = self.image_processor.process_standard(image)
                print("표준 전처리 적용")
        elif use_char_segmentation:
            # Contour 방식 강제 사용
            processed_image = self.image_processor.process_with_char_segmentation(image, plate_type=plate_type)
            print("Contour 기반 문자 추출 방식 사용")
        else:
            # 기존 방식 강제 사용
            processed_image = self.image_processor.process_standard(image)
            print("표준 전처리 사용")

        try:
            # 이미지 크기에 따라 EasyOCR 파라미터 조정
            print(f"OCR 처리 시작: 이미지 크기 {width}x{height}")
            print(f"허용 문자 수: {len(self.allowed_chars)}")
            print(f"허용 문자 샘플: {self.allowed_chars[:50]}...")

            if height < 50 or width < 150:  # 작은 이미지
                print("작은 이미지용 OCR 설정 사용")
                results = self.backend.recognize(
                    processed_image,
                    detail=1,
                    allowlist=self.allowed_chars,
                    paragraph=False,
                    width_ths=0.05,  # 매우 작은 문자도 인식 (더 낮게)
                    height_ths=0.05,
                    text_threshold=0.3,  # 텍스트 감지 임계값 더 낮춤
                    low_text=0.1,  # 낮은 품질 텍스트도 허용
                    decoder='beamsearch'
                )
            else:
                print("일반 이미지용 OCR 설정 사용")
                results = self.backend.recognize(
                    processed_image,
                    detail=1,
                    allowlist=self.allowed_chars,
                    paragraph=False,
                    width_ths=0.1,  # 텍스트 폭 임계값 더 낮춤
                    height_ths=0.1,  # 텍스트 높이 임계값 더 낮춤
                    text_threshold=0.3,  # 텍스트 감지 임계값 낮춤
                    low_text=0.1,  # 낮은 품질 텍스트도 허용
                    decoder='beamsearch'  # 더 정확한 디코딩
                )

            print(f"EasyOCR 원시 결과 개수: {len(results) if results else 0}")
            if results:
                for i, result in enumerate(results):
                    bbox, text, confidence = result
                    print(f"  결과 {i+1}: 텍스트='{text}', 신뢰도={confidence:.3f}")

        except Exception as e:
            logger.error(f"OCR Error: {e}", exc_info=True)
            return "", 0.0

        if not results:
            logger.warning("EasyOCR 결과 없음 - 예비 방법들 시도")
            # 디버깅용으로 전처리된 이미지 저장
            try:
                debug_filename = f"debug_ocr_{uuid.uuid4().hex[:8]}.jpg"
                debug_path = Path(config.DEBUG_DIR) / debug_filename
                cv2.imwrite(str(debug_path), processed_image)
                logger.info(f"디버그 이미지 저장: {debug_path}")
            except Exception as e:
                logger.error(f"디버그 이미지 저장 실패: {e}")

            # 예비 방법 0: 허용문자 제한 없이 시도
            logger.info("허용문자 제한 없이 OCR 시도")
            try:
                no_filter_results = self.backend.recognize(
                    processed_image,
                    detail=1,
                    paragraph=False,
                    width_ths=0.05,
                    height_ths=0.05,
                    text_threshold=0.2,
                    low_text=0.1,
                    decoder='beamsearch'
                )
                print(f"허용문자 제한 없는 결과: {len(no_filter_results) if no_filter_results else 0}개")
                if no_filter_results:
                    for i, result in enumerate(no_filter_results):
                        bbox, text, confidence = result
                        print(f"  제한없음 결과 {i+1}: 텍스트='{text}', 신뢰도={confidence:.3f}")
                    results = no_filter_results  # 허용문자 제한 없는 결과 사용
            except Exception as e:
                print(f"허용문자 제한 없는 OCR 실패: {e}")

            # 예비 방법 1: 다른 전처리로 재시도
            if not results:
                backup_results = self._try_backup_ocr(processed_image)
                if backup_results:
                    results = backup_results
                else:
                    return "", 0.0

        # 작은 이미지의 경우 신뢰도 기준을 낮춤
        actual_min_confidence = min_confidence
        if height < 50 or width < 150:  # 작은 이미지
            actual_min_confidence = max(0.1, min_confidence - 0.2)  # 신뢰도 기준 완화
            print(f"작은 이미지 감지: 신뢰도 기준을 {min_confidence} -> {actual_min_confidence}로 낮춤")
        
        # 신뢰도 필터링 및 좌표 기준 정렬
        filtered_results = [r for r in results if r[2] >= actual_min_confidence]
        
        # 번호판 텍스트를 좌표 순서대로 정렬 (왼쪽에서 오른쪽으로)
        if len(filtered_results) > 1:
            filtered_results.sort(key=lambda x: x[0][0][0])  # X 좌표 기준 정렬

        if not filtered_results:
            return "", 0.0

        texts = [r[1] for r in filtered_results]
        confidences = [r[2] for r in filtered_results]

        combined_text = "".join(texts) # 번호판은 공백 없이 합치는 것이 나을 수 있음
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        processed_text = self.post_processor.process(combined_text)
        return processed_text, avg_confidence
    
    def _try_backup_ocr(self, image):
        """예비 OCR 방법들을 시도"""
        logger.info("예비 OCR 방법 시도 중...")
        
        # 방법 1: 이진화 + 노이즈 제거
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # 적응형 이진화
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # 노이즈 제거
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 다시 3채널로 변환
            if len(cleaned.shape) == 2:
                cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            
            print("이진화 전처리로 OCR 재시도")
            results = self.backend.recognize(cleaned, detail=1, allowlist=self.allowed_chars, 
                                         paragraph=False, width_ths=0.1, height_ths=0.1)
            if results:
                print(f"예비 방법 성공: {len(results)}개 텍스트 발견")
                return results
                
        except Exception as e:
            print(f"예비 OCR 방법 1 실패: {e}")
        
        # 방법 2: 강한 대비 증가
        try:
            enhanced = cv2.convertScaleAbs(image, alpha=2.0, beta=0)  # 대비 2배 증가
            print("대비 강화로 OCR 재시도")
            results = self.backend.recognize(enhanced, detail=1, allowlist=self.allowed_chars,
                                         paragraph=False, width_ths=0.05, height_ths=0.05)
            if results:
                print(f"예비 방법 2 성공: {len(results)}개 텍스트 발견")
                return results
                
        except Exception as e:
            print(f"예비 OCR 방법 2 실패: {e}")
            
        return None

    def recognize_korean_license_plate(self, image):
        # 이 함수는 recognize_with_confidence를 사용하므로 별도 수정은 적음
        # 다만, TextPostProcessor의 format_korean_license_plate가 중요
        text, confidence = self.recognize_with_confidence(image)
        plate_text = self.post_processor.format_korean_license_plate(text)
        return plate_text
    
    def recognize_with_classification(self, image, min_confidence=None):
        """
        번호판 인식과 동시에 타입 분류 수행

        Args:
            image: 번호판 이미지 (numpy array)
            min_confidence: 최소 OCR 신뢰도

        Returns:
            dict: {
                'text': 인식된 텍스트,
                'confidence': OCR 신뢰도,
                'classification': 분류 결과,
                'validation': 유효성 검사 결과,
                'preprocessing_info': 전처리 정보
            }
        """
        min_confidence = min_confidence if min_confidence is not None else config.MIN_OCR_CONFIDENCE

        # 번호판 분류 먼저 수행 (원본 이미지 사용 - 색상 분석을 위해)
        classification = self.plate_classifier.classify_plate(image, "")
        plate_type = classification['type']

        # 타입별 전처리 적용
        from ..preprocessing.image_processor import ImageProcessor
        processor = ImageProcessor()
        processed_image = processor.process_for_plate_type(image, plate_type)

        # 2행 번호판 처리
        if plate_type.name in ['MOTORCYCLE', 'CONSTRUCTION']:
            h, w = processed_image.shape
            mid_h = h // 2

            # 상단과 하단 별도 OCR
            top_text, top_conf = self._recognize_single(processed_image[:mid_h, :], min_confidence)
            bottom_text, bottom_conf = self._recognize_single(processed_image[mid_h:, :], min_confidence)

            # 결합
            text = top_text + " " + bottom_text
            confidence = (top_conf + bottom_conf) / 2
        else:
            # 단일 OCR
            text, confidence = self._recognize_single(processed_image, min_confidence)

        # 한국 번호판 전용 후처리 적용
        processed_text = self.korean_postprocessor.process_by_plate_type(text, classification['type'])

        # 형식 유효성 검사
        validation = self.korean_postprocessor.validate_format(processed_text, classification['type'])

        return {
            'text': processed_text,
            'confidence': confidence,
            'classification': classification,
            'validation': validation,
            'preprocessing_info': {'mode': 'type_optimized', 'plate_type': plate_type.name}
        }

    def _recognize_single(self, image, min_confidence):
        """
        단일 이미지 영역에 대한 OCR 수행

        Args:
            image: 전처리된 이미지
            min_confidence: 최소 신뢰도

        Returns:
            tuple: (인식된 텍스트, 신뢰도)
        """
        if image is None or image.size == 0:
            return "", 0.0

        try:
            # EasyOCR 실행
            results = self.backend.recognize(
                image,
                detail=1,
                allowlist=self.allowed_chars,
                paragraph=False,
                width_ths=0.1,
                height_ths=0.1,
                text_threshold=0.3,
                low_text=0.1,
                decoder='beamsearch'
            )

            if not results:
                return "", 0.0

            # 신뢰도 필터링 및 좌표 기준 정렬
            filtered_results = [r for r in results if r[2] >= min_confidence]

            # 번호판 텍스트를 좌표 순서대로 정렬 (왼쪽에서 오른쪽으로)
            if len(filtered_results) > 1:
                filtered_results.sort(key=lambda x: x[0][0][0])  # X 좌표 기준 정렬

            if not filtered_results:
                return "", 0.0

            texts = [r[1] for r in filtered_results]
            confidences = [r[2] for r in filtered_results]

            combined_text = "".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return combined_text, avg_confidence

        except Exception as e:
            logger.error(f"OCR Error in _recognize_single: {e}", exc_info=True)
            return "", 0.0
    
    def _post_process_by_type(self, text, plate_type):
        """번호판 타입에 따른 추가 후처리"""
        from ..classification.plate_classifier import PlateType
        
        # 기본 후처리
        processed_text = self.post_processor.format_korean_license_plate(text)
        
        # 타입별 추가 처리
        if plate_type == PlateType.ELECTRIC:
            # 전기차 번호판: EV 표기 제거
            processed_text = processed_text.replace('EV', '').strip()
        elif plate_type == PlateType.DIPLOMATIC:
            # 외교관용: 외교, 영사 문구 처리
            processed_text = processed_text.replace('외교', '').replace('영사', '').strip()
        elif plate_type == PlateType.MILITARY:
            # 군용: 국, 육, 해, 공, 합 문자 유지
            pass
        elif plate_type == PlateType.CONSTRUCTION:
            # 건설기계: 지역명과 하이픈 처리
            pass
        
        return processed_text
    
    def recognize_korean_plate_optimized(self, image, plate_type='general', min_confidence=None):
        """
        한국 번호판에 최적화된 인식 (제공된 이미지 구조 기반)
        
        Args:
            image: 번호판 이미지
            plate_type: 'general', 'general_3digit', 'commercial' 등
            min_confidence: 최소 신뢰도
            
        Returns:
            dict: 인식 결과와 상세 정보
        """
        if min_confidence is None:
            min_confidence = config.MIN_OCR_CONFIDENCE
            
        # 번호판 타입에 최적화된 전처리
        optimized_image = self.image_processor.process_for_plate_type(image, plate_type)
        
        # 한국어 문자 특화 전처리
        korean_optimized = self.image_processor.optimize_for_korean_chars(optimized_image)
        
        # 두 가지 전처리 결과로 OCR 수행
        results = []
        
        # 1. 타입 최적화 이미지로 인식
        text1, conf1 = self.recognize_with_confidence(optimized_image, min_confidence, use_advanced_preprocessing=False)
        if text1:
            results.append((text1, conf1, 'type_optimized'))
        
        # 2. 한국어 특화 이미지로 인식
        text2, conf2 = self.recognize_with_confidence(korean_optimized, min_confidence, use_advanced_preprocessing=False)
        if text2:
            results.append((text2, conf2, 'korean_optimized'))
        
        # 3. 원본 이미지로 백업 인식
        if not results:
            text3, conf3 = self.recognize_with_confidence(image, min_confidence)
            if text3:
                results.append((text3, conf3, 'original'))
        
        # 최상의 결과 선택
        if results:
            # 신뢰도가 높은 순으로 정렬
            results.sort(key=lambda x: x[1], reverse=True)
            best_text, best_conf, method = results[0]
            
            # 한국 번호판 후처리 적용
            from ..classification.plate_classifier import PlateType, PlateClassifier
            classifier = PlateClassifier()
            classification = classifier.classify_plate(image, best_text)
            
            final_text = self.korean_postprocessor.process_by_plate_type(
                best_text, classification['type']
            )
            
            # 검증
            validation = self.korean_postprocessor.validate_format(
                final_text, classification['type']
            )
            
            return {
                'text': final_text,
                'confidence': best_conf,
                'method': method,
                'raw_results': results,
                'classification': classification,
                'validation': validation,
                'preprocessing_used': [method]
            }
        
        return {
            'text': '',
            'confidence': 0.0,
            'method': 'failed',
            'raw_results': [],
            'classification': None,
            'validation': {'is_valid': False, 'errors': ['인식 실패']},
            'preprocessing_used': []
        }

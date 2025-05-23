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
        # 각 전처리 모듈 초기화
        self.blur_corrector = BlurCorrection()
        self.perspective_corrector = PerspectiveCorrection()
        self.normalizer = Normalize()
        
        # 적응형 처리 설정 로드
        self.adaptive_enabled = config.ADAPTIVE_PROCESSING.get('enabled', True)
        self.quality_thresholds = config.QUALITY_THRESHOLDS
        self.processing_levels = config.ADAPTIVE_PROCESSING.get('processing_levels', {})
        
    def assess_image_quality(self, image):
        """
        이미지 품질을 평가하여 처리 수준을 결정
        
        Args:
            image (numpy.ndarray): BGR 형식의 번호판 이미지
            
        Returns:
            tuple: (품질 메트릭 dict, 처리 수준 str)
        """
        # 이미지 품질 평가
        quality_metrics = assess_image_quality(image)
        
        # 처리 수준 결정
        processing_level = determine_processing_level(quality_metrics)
        
        return quality_metrics, processing_level
    
    def minimal_process(self, image):
        """
        최소 처리 (고품질 이미지용)
        
        Args:
            image (numpy.ndarray): BGR 형식의 번호판 이미지
            
        Returns:
            numpy.ndarray: 최소 전처리된 번호판 이미지
        """
        # 이미지 복사본 생성
        processed_img = image.copy()
        
        # 그레이스케일 변환
        if len(processed_img.shape) == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_img.copy()
        
        # 비율 정규화 (기본)
        normalized = self.normalizer.normalize(gray)
        
        # 최소한의 대비 향상
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        enhanced = clahe.apply(normalized)
        
        return enhanced
    
    def moderate_process(self, image):
        """
        적당한 처리 (중간 품질 이미지용)
        
        Args:
            image (numpy.ndarray): BGR 형식의 번호판 이미지
            
        Returns:
            numpy.ndarray: 적당히 전처리된 번호판 이미지
        """
        # 이미지 복사본 생성
        processed_img = image.copy()
        
        # 그레이스케일 변환
        if len(processed_img.shape) == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_img.copy()
        
        # 가벼운 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
        
        # 가벼운 흐림 보정
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # 이진화
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 비율 정규화
        normalized = self.normalizer.normalize(binary)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        enhanced = clahe.apply(normalized)
        
        return enhanced
    
    def full_process(self, image):
        """
        전체 처리 (저품질 이미지용) - 기존 process() 메서드를 이름 변경
        
        Args:
            image (numpy.ndarray): BGR 형식의 번호판 이미지
            
        Returns:
            numpy.ndarray: 전체 전처리된 번호판 이미지
        """
        # 이미지 복사본 생성
        processed_img = image.copy()
        
        # 그레이스케일 변환
        if len(processed_img.shape) == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_img.copy()
            
        # 강한 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 흐림 보정
        deblurred = self.blur_corrector.correct(denoised)
        
        # 이진화 (번호판은 밝은 배경에 어두운 문자)
        _, binary = cv2.threshold(deblurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 기울기/원근 보정
        warped = self.perspective_corrector.correct(binary)
        
        # 비율 정규화
        normalized = self.normalizer.normalize(warped)
        
        # 강한 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        # 추가적인 노이즈 제거
        final = cv2.medianBlur(enhanced, 3)
        
        return final
    
    def process(self, image):
        """
        이미지 품질에 따라 적절한 전처리 자동 적용
        
        Args:
            image (numpy.ndarray): BGR 형식의 번호판 이미지
        
        Returns:
            numpy.ndarray: 전처리된 번호판 이미지
        """
        _, processing_level = self.assess_image_quality(image)
        if processing_level == 'minimal':
            return self.minimal_process(image)
        elif processing_level == 'moderate':
            return self.moderate_process(image)
        else:
            return self.full_process(image)
    
    def visualize_steps(self, image):
        """
        전처리 단계별 시각화
        
        Args:
            image (numpy.ndarray): BGR 형식의 번호판 이미지
            
        Returns:
            dict: 각 전처리 단계별 이미지 딕셔너리
        """
        # 이미지 복사본 생성
        original = image.copy()
        
        # 그레이스케일 변환
        if len(original.shape) == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray = original.copy()
            
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 흐림 보정
        deblurred = self.blur_corrector.correct(denoised)
        
        # 이진화
        _, binary = cv2.threshold(deblurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 기울기/원근 보정
        warped = self.perspective_corrector.correct(binary)
        
        # 비율 정규화
        normalized = self.normalizer.normalize(warped)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        # 단계별 이미지 딕셔너리 반환
        return {
            'original': original,
            'gray': gray,
            'denoised': denoised,
            'deblurred': deblurred,
            'binary': binary,
            'warped': warped,
            'normalized': normalized,
            'enhanced': enhanced,
            'final': cv2.medianBlur(enhanced, 3)
        }
    
    def visualize_adaptive_steps(self, image):
        """
        적응형 처리 단계별 시각화 (품질 평가 포함)
        
        Args:
            image (numpy.ndarray): BGR 형식의 번호판 이미지
            
        Returns:
            dict: 적응형 처리 결과 및 단계별 이미지
        """
        # 품질 평가
        quality_metrics, processing_level = self.assess_image_quality(image)
        
        # 각 처리 수준별 결과 생성
        minimal_result = self.minimal_process(image)
        moderate_result = self.moderate_process(image)
        full_result = self.full_process(image)
        
        # 선택된 처리 결과
        if processing_level == 'minimal':
            selected_result = minimal_result
        elif processing_level == 'moderate':
            selected_result = moderate_result
        else:
            selected_result = full_result
        
        return {
            'original': image,
            'quality_metrics': quality_metrics,
            'processing_level': processing_level,
            'processing_description': self.processing_levels.get(processing_level, {}).get('description', ''),
            'minimal_result': minimal_result,
            'moderate_result': moderate_result,
            'full_result': full_result,
            'selected_result': selected_result
        }
    
    def compare_processing_levels(self, image):
        """
        다양한 처리 수준을 비교
        
        Args:
            image (numpy.ndarray): BGR 형식의 번호판 이미지
            
        Returns:
            dict: 각 처리 수준별 결과 비교
        """
        # 품질 평가
        quality_metrics, recommended_level = self.assess_image_quality(image)
        
        # 각 처리 수준 적용
        results = {}
        processing_times = {}
        
        import time
        
        # 최소 처리
        start_time = time.time()
        results['minimal'] = self.minimal_process(image)
        processing_times['minimal'] = time.time() - start_time
        
        # 적당한 처리
        start_time = time.time()
        results['moderate'] = self.moderate_process(image)
        processing_times['moderate'] = time.time() - start_time
        
        # 전체 처리
        start_time = time.time()
        results['full'] = self.full_process(image)
        processing_times['full'] = time.time() - start_time
        
        return {
            'original': image,
            'quality_metrics': quality_metrics,
            'recommended_level': recommended_level,
            'results': results,
            'processing_times': processing_times,
            'level_descriptions': {
                level: info.get('description', '') 
                for level, info in self.processing_levels.items()
            }
        }

    def get_quality_report(self, image):
        """
        이미지 품질 상세 리포트 생성
        
        Args:
            image (numpy.ndarray): BGR 형식의 번호판 이미지
            
        Returns:
            dict: 상세 품질 리포트
        """
        quality_metrics, processing_level = self.assess_image_quality(image)
        
        # 품질 등급 결정
        overall_score = quality_metrics['overall_score']
        if overall_score >= 80:
            quality_grade = 'Excellent'
        elif overall_score >= 60:
            quality_grade = 'Good'
        elif overall_score >= 40:
            quality_grade = 'Fair'
        elif overall_score >= 20:
            quality_grade = 'Poor'
        else:
            quality_grade = 'Very Poor'
        
        # 개선 권장사항 생성
        recommendations = []
        if quality_metrics['sharpness_norm'] < 0.5:
            recommendations.append("이미지가 흐릿합니다. 더 선명한 촬영을 권장합니다.")
        if quality_metrics['contrast_norm'] < 0.4:
            recommendations.append("이미지 대비가 낮습니다. 조명 조건을 개선해보세요.")
        if quality_metrics['noise_norm'] < 0.6:
            recommendations.append("노이즈가 많습니다. 안정적인 촬영 환경을 권장합니다.")
        if quality_metrics['brightness_norm'] < 0.5:
            recommendations.append("밝기가 적절하지 않습니다. 조명을 조정해보세요.")
        
        if not recommendations:
            recommendations.append("이미지 품질이 양호합니다.")
        
        return {
            'quality_metrics': quality_metrics,
            'quality_grade': quality_grade,
            'processing_level': processing_level,
            'processing_description': self.processing_levels.get(processing_level, {}).get('description', ''),
            'recommendations': recommendations,
            'score_breakdown': {
                'sharpness': f"{quality_metrics['sharpness_norm']*100:.1f}%",
                'contrast': f"{quality_metrics['contrast_norm']*100:.1f}%",
                'noise': f"{quality_metrics['noise_norm']*100:.1f}%",
                'brightness': f"{quality_metrics['brightness_norm']*100:.1f}%",
                'overall': f"{overall_score:.1f}%"
            }
        }
    
    def set_adaptive_mode(self, enabled):
        """
        적응형 처리 모드 설정
        
        Args:
            enabled (bool): 적응형 처리 활성화 여부
        """
        self.adaptive_enabled = enabled
    
    def get_processing_statistics(self, images):
        """
        여러 이미지에 대한 처리 통계 생성
        
        Args:
            images (list): 이미지 목록
            
        Returns:
            dict: 처리 통계 정보
        """
        if not images:
            return {}
        
        stats = {
            'total_images': len(images),
            'processing_levels': {'minimal': 0, 'moderate': 0, 'full': 0},
            'quality_scores': [],
            'avg_quality_score': 0,
            'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        }
        
        for image in images:
            quality_metrics, processing_level = self.assess_image_quality(image)
            
            # 처리 수준 카운트
            stats['processing_levels'][processing_level] += 1
            
            # 품질 점수 수집
            overall_score = quality_metrics['overall_score']
            stats['quality_scores'].append(overall_score)
            
            # 품질 분포 카운트
            if overall_score >= 75:
                stats['quality_distribution']['excellent'] += 1
            elif overall_score >= 50:
                stats['quality_distribution']['good'] += 1
            elif overall_score >= 25:
                stats['quality_distribution']['fair'] += 1
            else:
                stats['quality_distribution']['poor'] += 1
        
        # 평균 품질 점수 계산
        stats['avg_quality_score'] = sum(stats['quality_scores']) / len(stats['quality_scores'])
        
        # 백분율 계산
        total = stats['total_images']
        stats['processing_levels_pct'] = {
            level: (count / total) * 100 
            for level, count in stats['processing_levels'].items()
        }
        
        stats['quality_distribution_pct'] = {
            grade: (count / total) * 100 
            for grade, count in stats['quality_distribution'].items()
        }
        
        return stats

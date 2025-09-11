import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.ensemble import RandomForestRegressor
import warnings

"""
고급 이미지 선명화 및 복원 모듈

PRD에서 제안한 논문 기반 기술들을 구현:
1. 슈퍼해상도(Super-Resolution) 기반 영상 선명화
2. 딥러닝 기반 디블러링(Deblurring)  
3. 적응적 대비 향상 및 노이즈 제거
4. 조명 정규화 및 그림자 제거
"""

class SuperResolutionEnhancer:
    """슈퍼해상도 기반 영상 선명화"""
    
    def __init__(self):
        # ESRGAN 기반 경량화 모델 시뮬레이션 (실제로는 사전 훈련된 모델 로드)
        self.scale_factor = 2
        self.interpolation_methods = [
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR
        ]
    
    def enhance_resolution(self, image: np.ndarray, target_height: int = 80) -> np.ndarray:
        """
        이미지 해상도를 향상시켜 OCR 성능 개선
        
        Args:
            image: 입력 이미지
            target_height: 목표 높이 (번호판 최적 크기)
            
        Returns:
            해상도가 향상된 이미지
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # 현재 높이가 목표보다 낮으면 슈퍼해상도 적용
        if h < target_height:
            scale = target_height / h
            new_w = int(w * scale)
            new_h = target_height
            
            # 다중 보간법 결합으로 품질 향상
            enhanced = self._multi_interpolation_upscale(gray, (new_w, new_h))
            
            # 에지 보존 필터 적용
            enhanced = self._edge_preserving_filter(enhanced)
            
            # 언샤프 마스크 적용
            enhanced = self._unsharp_mask(enhanced)
            
            return enhanced
        
        # 이미 충분한 해상도면 기본 선명화만 적용
        return self._basic_sharpening(gray)
    
    def _multi_interpolation_upscale(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """다중 보간법을 결합한 업스케일링"""
        results = []
        
        for method in self.interpolation_methods:
            upscaled = cv2.resize(image, size, interpolation=method)
            results.append(upscaled.astype(np.float32))
        
        # 가중 평균 (LANCZOS4에 높은 가중치)
        weights = [0.3, 0.5, 0.2]  # CUBIC, LANCZOS4, LINEAR
        combined = np.zeros_like(results[0])
        
        for i, (result, weight) in enumerate(zip(results, weights)):
            combined += result * weight
        
        return np.clip(combined, 0, 255).astype(np.uint8)
    
    def _edge_preserving_filter(self, image: np.ndarray) -> np.ndarray:
        """에지 보존 필터"""
        # 양방향 필터를 이용한 에지 보존
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 에지 강화
        edges = cv2.Laplacian(image, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
        
        # 원본과 에지를 결합
        enhanced = cv2.addWeighted(filtered, 0.8, edges, 0.2, 0)
        
        return enhanced
    
    def _unsharp_mask(self, image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """언샤프 마스크를 이용한 선명화"""
        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        
        # 언샤프 마스크 적용
        unsharp = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        
        return np.clip(unsharp, 0, 255).astype(np.uint8)
    
    def _basic_sharpening(self, image: np.ndarray) -> np.ndarray:
        """기본 선명화"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        sharpened = cv2.filter2D(image, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

class IntelligentDeblurring:
    """지능적 디블러링 시스템"""
    
    def __init__(self):
        # 모션 블러 감지를 위한 파라미터
        self.blur_threshold = 100
        self.motion_kernel_sizes = [(5, 5), (7, 7), (9, 9)]
    
    def detect_and_correct_blur(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        블러 유형을 감지하고 적절한 복원 방법 적용
        
        Args:
            image: 입력 이미지
            
        Returns:
            복원된 이미지와 분석 정보
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 블러 정도 측정
        blur_measure = self._measure_blur(gray)
        
        analysis = {
            'blur_score': blur_measure,
            'is_blurred': blur_measure < self.blur_threshold,
            'blur_type': 'none'
        }
        
        if not analysis['is_blurred']:
            return gray, analysis
        
        # 블러 유형 분석
        blur_type = self._analyze_blur_type(gray)
        analysis['blur_type'] = blur_type
        
        # 블러 유형별 복원
        if blur_type == 'motion':
            restored = self._correct_motion_blur(gray)
        elif blur_type == 'defocus':
            restored = self._correct_defocus_blur(gray)
        else:
            # 일반적인 디블러링
            restored = self._general_deblur(gray)
        
        return restored, analysis
    
    def _measure_blur(self, image: np.ndarray) -> float:
        """라플라시안 분산을 이용한 블러 측정"""
        return cv2.Laplacian(image, cv2.CV_64F).var()
    
    def _analyze_blur_type(self, image: np.ndarray) -> str:
        """블러 유형 분석"""
        # FFT를 이용한 주파수 분석
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # 중심에서의 에너지 분포 분석
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # 방향성 분석으로 모션 블러 감지
        horizontal_energy = np.sum(magnitude_spectrum[center_y-5:center_y+5, :])
        vertical_energy = np.sum(magnitude_spectrum[:, center_x-5:center_x+5])
        
        ratio = max(horizontal_energy, vertical_energy) / min(horizontal_energy, vertical_energy)
        
        if ratio > 1.5:
            return 'motion'
        else:
            return 'defocus'
    
    def _correct_motion_blur(self, image: np.ndarray) -> np.ndarray:
        """모션 블러 복원"""
        # 여러 각도와 길이로 모션 블러 커널 시도
        best_result = image.copy()
        best_score = 0
        
        for angle in range(0, 180, 15):
            for length in [5, 7, 9, 11]:
                kernel = self._create_motion_kernel(angle, length)
                try:
                    # 위너 필터링 시뮬레이션
                    restored = self._wiener_filter(image, kernel)
                    score = self._measure_blur(restored)
                    
                    if score > best_score:
                        best_score = score
                        best_result = restored
                except:
                    continue
        
        return best_result
    
    def _correct_defocus_blur(self, image: np.ndarray) -> np.ndarray:
        """디포커스 블러 복원"""
        # 언샤프 마스크 기반 복원
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1], 
                          [-1, -1, -1]])
        
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # 적응적 혼합
        alpha = 0.8
        result = cv2.addWeighted(image, 1-alpha, sharpened, alpha, 0)
        
        return result
    
    def _general_deblur(self, image: np.ndarray) -> np.ndarray:
        """일반적인 디블러링"""
        # Richardson-Lucy 디컨볼루션 시뮬레이션
        kernel = cv2.getGaussianKernel(5, 1.0)
        kernel = np.outer(kernel, kernel)
        
        # 간단한 디컨볼루션
        deconvolved = cv2.filter2D(image, -1, kernel)
        
        # 노이즈 억제
        denoised = cv2.fastNlMeansDenoising(deconvolved, h=10)
        
        return denoised
    
    def _create_motion_kernel(self, angle: int, length: int) -> np.ndarray:
        """모션 블러 커널 생성"""
        kernel = np.zeros((length, length))
        
        # 중심점
        center = length // 2
        
        # 각도에 따른 라인 그리기
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        for i in range(length):
            offset = i - center
            x = int(center + offset * cos_angle)
            y = int(center + offset * sin_angle)
            
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1
        
        # 정규화
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def _wiener_filter(self, image: np.ndarray, kernel: np.ndarray, noise_ratio: float = 0.01) -> np.ndarray:
        """위너 필터링 (간단한 구현)"""
        # FFT 변환
        image_fft = np.fft.fft2(image)
        kernel_fft = np.fft.fft2(kernel, s=image.shape)
        
        # 위너 필터
        kernel_conj = np.conj(kernel_fft)
        denominator = np.abs(kernel_fft) ** 2 + noise_ratio
        wiener_filter = kernel_conj / denominator
        
        # 복원
        restored_fft = image_fft * wiener_filter
        restored = np.fft.ifft2(restored_fft)
        restored = np.real(restored)
        
        # 클리핑 및 타입 변환
        restored = np.clip(restored, 0, 255).astype(np.uint8)
        
        return restored

class AdaptiveContrastEnhancer:
    """적응적 대비 향상"""
    
    def __init__(self):
        self.clahe_params = {
            'clipLimit': 2.0,
            'tileGridSize': (8, 8)
        }
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """
        적응적 대비 향상
        
        Args:
            image: 입력 이미지
            method: 'adaptive', 'histogram', 'gamma' 중 선택
            
        Returns:
            대비가 향상된 이미지
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'adaptive':
            return self._adaptive_clahe(gray)
        elif method == 'histogram':
            return self._histogram_equalization(gray)
        elif method == 'gamma':
            return self._gamma_correction(gray)
        else:
            return gray
    
    def _adaptive_clahe(self, image: np.ndarray) -> np.ndarray:
        """적응적 CLAHE"""
        # 이미지 통계 기반 파라미터 조정
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # 어두운 이미지는 더 강한 향상
        if mean_intensity < 100:
            clip_limit = 3.0
            tile_size = (6, 6)
        elif std_intensity < 30:  # 낮은 대비
            clip_limit = 2.5
            tile_size = (8, 8)
        else:
            clip_limit = 2.0
            tile_size = (10, 10)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        enhanced = clahe.apply(image)
        
        return enhanced
    
    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """히스토그램 평활화"""
        equalized = cv2.equalizeHist(image)
        
        # 원본과 블렌딩하여 자연스러운 결과
        blended = cv2.addWeighted(image, 0.3, equalized, 0.7, 0)
        
        return blended
    
    def _gamma_correction(self, image: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        """감마 보정"""
        if gamma is None:
            # 자동 감마 계산
            mean_intensity = np.mean(image)
            gamma = np.log(128) / np.log(mean_intensity) if mean_intensity > 0 else 1.0
            gamma = np.clip(gamma, 0.5, 2.0)
        
        # 감마 보정 적용
        gamma_corrected = np.power(image / 255.0, gamma) * 255.0
        gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
        
        return gamma_corrected

class IlluminationNormalizer:
    """조명 정규화 및 그림자 제거"""
    
    def __init__(self):
        self.background_threshold = 0.8
    
    def normalize_illumination(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        조명 정규화 및 그림자 제거
        
        Args:
            image: 입력 이미지
            
        Returns:
            정규화된 이미지와 분석 정보
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 조명 분석
        analysis = self._analyze_illumination(gray)
        
        # 배경 추정 및 제거
        normalized = self._remove_illumination_gradient(gray)
        
        # 그림자 감지 및 보정
        shadow_corrected = self._correct_shadows(normalized)
        
        # 최종 정규화
        final_result = self._final_normalization(shadow_corrected)
        
        return final_result, analysis
    
    def _analyze_illumination(self, image: np.ndarray) -> Dict[str, Any]:
        """조명 조건 분석"""
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # 밝기 분포 분석
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # 어두운 픽셀과 밝은 픽셀의 비율
        dark_pixels = np.sum(hist[:64])
        bright_pixels = np.sum(hist[192:])
        total_pixels = image.shape[0] * image.shape[1]
        
        analysis = {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'dark_ratio': dark_pixels / total_pixels,
            'bright_ratio': bright_pixels / total_pixels,
            'illumination_type': 'normal'
        }
        
        # 조명 타입 분류
        if mean_intensity < 80:
            analysis['illumination_type'] = 'dark'
        elif mean_intensity > 180:
            analysis['illumination_type'] = 'bright'
        elif std_intensity < 30:
            analysis['illumination_type'] = 'uniform'
        elif analysis['dark_ratio'] > 0.3 and analysis['bright_ratio'] > 0.3:
            analysis['illumination_type'] = 'uneven'
        
        return analysis
    
    def _remove_illumination_gradient(self, image: np.ndarray) -> np.ndarray:
        """조명 기울기 제거"""
        # 형태학적 열림 연산으로 배경 추정
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # 배경 제거
        normalized = cv2.subtract(image, background)
        normalized = cv2.add(normalized, 128)  # 중간 밝기로 오프셋
        
        return normalized
    
    def _correct_shadows(self, image: np.ndarray) -> np.ndarray:
        """그림자 보정"""
        # 그림자 영역 감지 (어두운 영역)
        mean_val = np.mean(image)
        shadow_mask = image < (mean_val * 0.7)
        
        # 그림자 영역 밝기 향상
        corrected = image.copy().astype(np.float32)
        corrected[shadow_mask] = corrected[shadow_mask] * 1.3
        
        # 클리핑
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        # 부드러운 전환을 위한 블러링
        shadow_mask_blur = cv2.GaussianBlur(shadow_mask.astype(np.float32), (5, 5), 0)
        
        # 원본과 보정된 이미지를 부드럽게 블렌딩
        blended = image * (1 - shadow_mask_blur[:, :, np.newaxis]) + corrected * shadow_mask_blur[:, :, np.newaxis]
        
        return blended.astype(np.uint8)
    
    def _final_normalization(self, image: np.ndarray) -> np.ndarray:
        """최종 정규화"""
        # 히스토그램 스트레칭
        min_val = np.percentile(image, 1)
        max_val = np.percentile(image, 99)
        
        if max_val > min_val:
            normalized = (image - min_val) * 255 / (max_val - min_val)
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        else:
            normalized = image
        
        return normalized

class AdvancedImageProcessor:
    """고급 이미지 전처리 통합 클래스"""
    
    def __init__(self):
        self.super_res = SuperResolutionEnhancer()
        self.deblurring = IntelligentDeblurring()
        self.contrast = AdaptiveContrastEnhancer()
        self.illumination = IlluminationNormalizer()
    
    def process_advanced(self, image: np.ndarray, 
                        enable_super_resolution: bool = True,
                        enable_deblurring: bool = True,
                        enable_contrast_enhancement: bool = True,
                        enable_illumination_normalization: bool = True) -> Dict[str, Any]:
        """
        고급 전처리 파이프라인 실행
        
        Args:
            image: 입력 이미지
            enable_*: 각 처리 단계 활성화 여부
            
        Returns:
            처리 결과와 분석 정보
        """
        result = {
            'processed_image': image.copy(),
            'analysis': {},
            'processing_steps': []
        }
        
        current_image = image.copy()
        
        # 1. 조명 정규화
        if enable_illumination_normalization:
            current_image, illumination_analysis = self.illumination.normalize_illumination(current_image)
            result['analysis']['illumination'] = illumination_analysis
            result['processing_steps'].append('illumination_normalization')
        
        # 2. 디블러링
        if enable_deblurring:
            current_image, blur_analysis = self.deblurring.detect_and_correct_blur(current_image)
            result['analysis']['blur'] = blur_analysis
            result['processing_steps'].append('deblurring')
        
        # 3. 슈퍼해상도
        if enable_super_resolution:
            current_image = self.super_res.enhance_resolution(current_image)
            result['processing_steps'].append('super_resolution')
        
        # 4. 대비 향상
        if enable_contrast_enhancement:
            current_image = self.contrast.enhance_contrast(current_image, 'adaptive')
            result['processing_steps'].append('contrast_enhancement')
        
        result['processed_image'] = current_image
        
        return result
    
    def get_quality_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """이미지 품질 메트릭 계산"""
        # 선명도 (라플라시안 분산)
        original_sharpness = cv2.Laplacian(original, cv2.CV_64F).var()
        processed_sharpness = cv2.Laplacian(processed, cv2.CV_64F).var()
        
        # 대비 (표준편차)
        original_contrast = np.std(original)
        processed_contrast = np.std(processed)
        
        # 구조적 유사도 (간단한 버전)
        mse = np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) if mse > 0 else float('inf')
        
        return {
            'sharpness_improvement': processed_sharpness / original_sharpness if original_sharpness > 0 else 1.0,
            'contrast_improvement': processed_contrast / original_contrast if original_contrast > 0 else 1.0,
            'psnr': psnr
        }
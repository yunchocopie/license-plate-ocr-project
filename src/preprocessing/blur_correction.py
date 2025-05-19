import cv2
import numpy as np
import config

"""
흐림 보정 모듈

이 모듈은 번호판 이미지의 흐림(블러)을 개선하는 클래스를 제공합니다.
역컨볼루션 기법과 샤프닝 알고리즘을 사용하여 흐린 이미지를 선명하게 만듭니다.
"""
class BlurCorrection:
    """흐림 보정을 위한 클래스"""
    
    def __init__(self, kernel_size=None, sigma=None):
        """
        BlurCorrection 클래스 초기화
        
        Args:
            kernel_size (tuple, optional): 가우시안 커널 크기. 기본값은 config에서 가져옴
            sigma (float, optional): 가우시안 시그마 값. 기본값은 config에서 가져옴
        """
        self.kernel_size = kernel_size or config.BLUR_KERNEL_SIZE
        self.sigma = sigma or config.BLUR_SIGMA
    
    def correct(self, image):
        """
        흐림 보정 수행
        
        Args:
            image (numpy.ndarray): 그레이스케일 이미지
            
        Returns:
            numpy.ndarray: 흐림이 보정된 이미지
        """
        # 1. 언샤프 마스킹 (Unsharp Masking) 적용
        # 가우시안 블러 적용
        gaussian = cv2.GaussianBlur(image, self.kernel_size, self.sigma)
        # 원본 이미지와의 차이 계산
        unsharp_mask = cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)
        
        # 2. 윈도우 기반 선명화 (Window-based sharpening)
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]], dtype=np.float32)
        sharpened = cv2.filter2D(unsharp_mask, -1, kernel)
        
        # 3. 라플라시안 선명화 (Laplacian sharpening)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        sharpened2 = cv2.addWeighted(sharpened, 0.7, laplacian, 0.3, 0)
        
        # 4. 경계선(Edge) 개선
        # 소벨 필터 적용
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = np.uint8(np.absolute(sobelx))
        sobely = np.uint8(np.absolute(sobely))
        sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        
        # 5. 최종 이미지: 선명화된 이미지와 경계선 가중 합성
        result = cv2.addWeighted(sharpened2, 0.8, sobel, 0.2, 0)
        
        # 6. 대비 조정
        result = self.adjust_contrast(result)
        
        return result
    
    def adjust_contrast(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        이미지 대비 조정
        
        Args:
            image (numpy.ndarray): 입력 이미지
            clip_limit (float, optional): CLAHE 클립 한계. 기본값은 2.0
            tile_grid_size (tuple, optional): CLAHE 타일 그리드 크기. 기본값은 (8, 8)
            
        Returns:
            numpy.ndarray: 대비가 조정된 이미지
        """
        # CLAHE(Contrast Limited Adaptive Histogram Equalization) 적용
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    def blind_deconvolution(self, image, iterations=10):
        """
        블라인드 디컨볼루션(Blind Deconvolution)을 통한 흐림 보정 (추가적인 방법)
        참고: 계산 비용이 높음
        
        Args:
            image (numpy.ndarray): 입력 이미지
            iterations (int, optional): 디컨볼루션 반복 횟수. 기본값은 10
            
        Returns:
            numpy.ndarray: 디컨볼루션이 적용된 이미지
        """
        # 초기 PSF(Point Spread Function) 추정
        psf = np.ones((5, 5)) / 25
        
        # Wiener 필터를 사용한 디컨볼루션
        deconvolved = image.copy().astype(np.float32)
        
        for _ in range(iterations):
            # Richardson-Lucy 알고리즘의 간소화된 버전
            blurred = cv2.filter2D(deconvolved, -1, psf)
            relative_blur = cv2.divide(image.astype(np.float32), blurred + 1e-10)
            deconvolved = cv2.multiply(deconvolved, cv2.filter2D(relative_blur, -1, psf[::-1, ::-1]))
        
        # 범위 정규화 및 8비트 변환
        deconvolved = np.clip(deconvolved, 0, 255).astype(np.uint8)
        
        return deconvolved
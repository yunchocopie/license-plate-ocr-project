import cv2
import numpy as np
import config

class BlurCorrection:
    def __init__(self, kernel_size=None, sigma=None):
        self.kernel_size = kernel_size or config.BLUR_KERNEL_SIZE
        self.sigma = sigma or config.BLUR_SIGMA

    def correct(self, image): # 입력은 그레이스케일 이미지로 가정
        if image is None or image.size == 0:
            return image # 또는 적절한 빈 이미지 반환

        # 1. 언샤프 마스킹 (비교적 안전하고 효과적인 방법)
        # 가우시안 블러의 sigma 값을 0으로 하면 커널 크기에 맞게 자동 계산됨
        gaussian_blurred = cv2.GaussianBlur(image, self.kernel_size, self.sigma)
        # addWeighted의 가중치를 조절하여 샤프닝 강도 조절 (예: 1.5, -0.5)
        # image * (1 + alpha) - gaussian_blurred * alpha
        # 여기서 alpha는 0.5, 1+alpha = 1.5
        unsharp_masked = cv2.addWeighted(image, 1.5, gaussian_blurred, -0.5, 0)

        # 추가적인 강한 샤프닝은 주석 처리하거나 매우 약하게 적용.
        # 현재 이미지에는 이정도면 충분하거나 오히려 과할 수 있음.
        # kernel = np.array([[-1, -1, -1],
        #                   [-1,  9, -1],
        #                   [-1, -1, -1]], dtype=np.float32)
        # sharpened = cv2.filter2D(unsharp_masked, -1, kernel)
        # laplacian = cv2.Laplacian(image, cv2.CV_64F) # 원본 이미지에 라플라시안
        # laplacian_uint8 = np.uint8(np.absolute(laplacian))
        # sharpened2 = cv2.addWeighted(sharpened, 0.7, laplacian_uint8, 0.3, 0)

        # 경계선 개선 등은 노이즈를 증폭시킬 수 있으므로 주의
        # sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        # ...

        # 최종적으로 대비 조정된 이미지 반환
        # 대비 조정은 ImageProcessor의 마지막 단계에서 수행하거나, 여기서도 가능
        # clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)) # clipLimit 낮춤
        # result = clahe.apply(unsharp_masked)
        result = unsharp_masked # 대비조정은 ImageProcessor에서

        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def adjust_contrast(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        # 이 함수는 ImageProcessor에서도 사용 가능하므로 여기에 둬도 괜찮음
        if image.dtype != np.uint8:
             # CLAHE는 uint8 타입 이미지에 적용되어야 함
             img_for_clahe = np.clip(image, 0, 255).astype(np.uint8)
        else:
             img_for_clahe = image

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img_for_clahe)

    # blind_deconvolution은 계산 비용이 매우 높으므로, 꼭 필요한 경우가 아니면 사용하지 않는 것이 좋음
    # def blind_deconvolution(...)
import cv2
import numpy as np
from typing import Tuple, Dict, List
from sklearn.cluster import KMeans
import webcolors

"""
번호판 색상 분석 전용 모듈

더욱 정확한 색상 분석을 위해 다양한 알고리즘을 조합하여 사용합니다.
- K-means 클러스터링을 통한 주요 색상 추출
- HSV 색공간에서의 정밀한 색상 분류
- 조명 조건에 강인한 색상 정규화
"""

class ColorAnalyzer:
    """번호판 색상 분석기"""
    
    def __init__(self):
        # 한국 번호판 색상 정의 (HSV 색공간)
        self.plate_colors = {
            'white': {
                'hsv_ranges': [
                    ([0, 0, 200], [180, 30, 255]),      # 기본 흰색
                    ([0, 0, 180], [180, 25, 255])       # 약간 어두운 흰색
                ],
                'rgb_center': (255, 255, 255),
                'tolerance': 40
            },
            'yellow': {
                'hsv_ranges': [
                    ([20, 100, 100], [30, 255, 255]),   # 표준 노란색
                    ([15, 80, 120], [35, 255, 255])     # 넓은 범위 노란색
                ],
                'rgb_center': (255, 255, 0),
                'tolerance': 50
            },
            'light_blue': {
                'hsv_ranges': [
                    ([90, 50, 150], [110, 255, 255]),   # 하늘색
                    ([85, 40, 140], [115, 255, 255])    # 넓은 범위 하늘색
                ],
                'rgb_center': (135, 206, 235),
                'tolerance': 45
            },
            'dark_blue': {
                'hsv_ranges': [
                    ([100, 150, 50], [130, 255, 200]),  # 남색
                    ([95, 120, 40], [135, 255, 220])    # 넓은 범위 남색
                ],
                'rgb_center': (0, 0, 139),
                'tolerance': 40
            },
            'orange': {
                'hsv_ranges': [
                    ([10, 150, 150], [20, 255, 255]),   # 주황색
                    ([8, 120, 140], [22, 255, 255])     # 넓은 범위 주황색
                ],
                'rgb_center': (255, 165, 0),
                'tolerance': 45
            },
            'red': {
                'hsv_ranges': [
                    ([0, 150, 150], [10, 255, 255]),    # 빨간색 (낮은 H)
                    ([170, 150, 150], [180, 255, 255])  # 빨간색 (높은 H)
                ],
                'rgb_center': (255, 0, 0),
                'tolerance': 40
            },
            'green': {
                'hsv_ranges': [
                    ([60, 100, 100], [80, 255, 255]),   # 연두색
                    ([55, 80, 120], [85, 255, 255])     # 넓은 범위 연두색
                ],
                'rgb_center': (124, 252, 0),
                'tolerance': 45
            },
            'blue': {  # 이륜차용 파란색
                'hsv_ranges': [
                    ([100, 100, 100], [120, 255, 255]), # 파란색
                    ([95, 80, 120], [125, 255, 255])    # 넓은 범위 파란색
                ],
                'rgb_center': (0, 0, 255),
                'tolerance': 40
            }
        }
        
        # 텍스트 색상 정의
        self.text_colors = {
            'black': {
                'hsv_ranges': [([0, 0, 0], [180, 255, 50])],
                'rgb_center': (0, 0, 0)
            },
            'white': {
                'hsv_ranges': [([0, 0, 200], [180, 30, 255])],
                'rgb_center': (255, 255, 255)
            },
            'blue': {
                'hsv_ranges': [([100, 100, 100], [120, 255, 255])],
                'rgb_center': (0, 0, 255)
            }
        }

    def analyze_plate_colors(self, image: np.ndarray) -> Dict[str, any]:
        """
        번호판 이미지의 배경색과 텍스트 색상을 종합적으로 분석
        
        Args:
            image: BGR 형식의 번호판 이미지
            
        Returns:
            Dict: 색상 분석 결과
        """
        # 이미지 전처리
        processed_image = self._preprocess_image(image)
        
        # 배경 영역과 텍스트 영역 분리
        bg_regions, text_regions = self._segment_regions(processed_image)
        
        # 각 영역의 주요 색상 추출
        bg_colors = self._extract_dominant_colors(bg_regions)
        text_colors = self._extract_dominant_colors(text_regions)
        
        # 색상 분류
        bg_classification = self._classify_background_color(bg_colors)
        text_classification = self._classify_text_color(text_colors)
        
        # 조명 조건 분석
        lighting_condition = self._analyze_lighting(processed_image)
        
        # 신뢰도 계산
        confidence = self._calculate_color_confidence(
            processed_image, bg_classification, text_classification
        )
        
        return {
            'background_color': bg_classification['color'],
            'background_confidence': bg_classification['confidence'],
            'text_color': text_classification['color'],
            'text_confidence': text_classification['confidence'],
            'lighting_condition': lighting_condition,
            'overall_confidence': confidence,
            'dominant_colors': {
                'background': bg_colors,
                'text': text_colors
            }
        }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리 (노이즈 제거, 대비 향상)"""
        # 가우시안 블러로 노이즈 제거
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        
        # CLAHE를 이용한 대비 향상
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _segment_regions(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """배경 영역과 텍스트 영역 분리"""
        h, w = image.shape[:2]
        
        # 배경 영역: 이미지 가장자리 (20% 테두리)
        border_width = int(w * 0.2)
        border_height = int(h * 0.2)
        
        # 상하좌우 테두리 영역
        top_border = image[0:border_height, :]
        bottom_border = image[h-border_height:h, :]
        left_border = image[:, 0:border_width]
        right_border = image[:, w-border_width:w]
        
        bg_regions = np.vstack([
            top_border.reshape(-1, 3),
            bottom_border.reshape(-1, 3),
            left_border.reshape(-1, 3),
            right_border.reshape(-1, 3)
        ])
        
        # 텍스트 영역: 중앙 영역
        center_y1, center_y2 = h // 3, 2 * h // 3
        center_x1, center_x2 = w // 4, 3 * w // 4
        
        text_regions = image[center_y1:center_y2, center_x1:center_x2].reshape(-1, 3)
        
        return bg_regions, text_regions
    
    def _extract_dominant_colors(self, pixels: np.ndarray, n_colors: int = 3) -> List[Tuple]:
        """K-means를 이용한 주요 색상 추출"""
        if len(pixels) == 0:
            return []
        
        # RGB로 변환 (OpenCV는 BGR)
        rgb_pixels = pixels[:, ::-1]
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), 
                       random_state=42, n_init=10)
        kmeans.fit(rgb_pixels)
        
        # 클러스터 중심과 비율 계산
        colors = []
        labels = kmeans.labels_
        
        for i in range(kmeans.n_clusters):
            cluster_size = np.sum(labels == i)
            percentage = cluster_size / len(labels)
            color_rgb = kmeans.cluster_centers_[i].astype(int)
            
            colors.append({
                'rgb': tuple(color_rgb),
                'percentage': percentage,
                'cluster_size': cluster_size
            })
        
        # 비율 순으로 정렬
        colors.sort(key=lambda x: x['percentage'], reverse=True)
        
        return colors
    
    def _classify_background_color(self, dominant_colors: List[Dict]) -> Dict[str, any]:
        """배경색 분류"""
        if not dominant_colors:
            return {'color': 'unknown', 'confidence': 0.0}
        
        # 가장 큰 비율의 색상을 배경색으로 간주
        main_color = dominant_colors[0]
        rgb = main_color['rgb']
        
        # BGR로 변환하여 HSV 분석
        bgr = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        
        # 각 번호판 색상과 비교
        best_match = 'unknown'
        best_score = 0.0
        
        for color_name, color_info in self.plate_colors.items():
            score = self._calculate_color_match_score(hsv, rgb, color_info)
            if score > best_score:
                best_score = score
                best_match = color_name
        
        # 신뢰도 조정 (주요 색상의 비율 고려)
        confidence = best_score * main_color['percentage']
        
        return {
            'color': best_match,
            'confidence': confidence,
            'rgb': rgb,
            'hsv': hsv.tolist()
        }
    
    def _classify_text_color(self, dominant_colors: List[Dict]) -> Dict[str, any]:
        """텍스트 색상 분류"""
        if not dominant_colors:
            return {'color': 'unknown', 'confidence': 0.0}
        
        # 두 번째로 큰 비율의 색상을 텍스트 색상으로 간주
        # (첫 번째는 배경색과 겹칠 가능성이 높음)
        text_color = dominant_colors[1] if len(dominant_colors) > 1 else dominant_colors[0]
        rgb = text_color['rgb']
        
        # BGR로 변환하여 HSV 분석
        bgr = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        
        # 명도 기반 간단 분류
        if hsv[2] < 100:  # 어두운 색상
            return {
                'color': 'black',
                'confidence': 0.8,
                'rgb': rgb,
                'hsv': hsv.tolist()
            }
        elif hsv[1] < 50 and hsv[2] > 200:  # 밝고 채도가 낮은 색상
            return {
                'color': 'white',
                'confidence': 0.8,
                'rgb': rgb,
                'hsv': hsv.tolist()
            }
        elif 100 <= hsv[0] <= 120:  # 파란색 계열
            return {
                'color': 'blue',
                'confidence': 0.7,
                'rgb': rgb,
                'hsv': hsv.tolist()
            }
        else:
            return {
                'color': 'unknown',
                'confidence': 0.3,
                'rgb': rgb,
                'hsv': hsv.tolist()
            }
    
    def _calculate_color_match_score(self, hsv: np.ndarray, rgb: tuple, color_info: Dict) -> float:
        """색상 매칭 점수 계산"""
        # HSV 범위 매칭
        hsv_score = 0.0
        for hsv_range in color_info['hsv_ranges']:
            lower, upper = hsv_range
            if (lower[0] <= hsv[0] <= upper[0] and
                lower[1] <= hsv[1] <= upper[1] and
                lower[2] <= hsv[2] <= upper[2]):
                hsv_score = 1.0
                break
            else:
                # 부분 매칭 점수 계산
                h_score = 1.0 - abs(hsv[0] - (lower[0] + upper[0]) / 2) / 90
                s_score = 1.0 - abs(hsv[1] - (lower[1] + upper[1]) / 2) / 127.5
                v_score = 1.0 - abs(hsv[2] - (lower[2] + upper[2]) / 2) / 127.5
                partial_score = max(0, (h_score + s_score + v_score) / 3 - 0.5) * 2
                hsv_score = max(hsv_score, partial_score)
        
        # RGB 거리 기반 점수
        rgb_center = color_info['rgb_center']
        rgb_distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, rgb_center)))
        max_distance = np.sqrt(3 * 255 ** 2)
        rgb_score = max(0, 1 - rgb_distance / max_distance)
        
        # 가중 평균
        return hsv_score * 0.7 + rgb_score * 0.3
    
    def _analyze_lighting(self, image: np.ndarray) -> str:
        """조명 조건 분석"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        if mean_brightness < 80:
            return 'dark'
        elif mean_brightness > 180:
            return 'bright'
        elif std_brightness < 30:
            return 'uniform'
        else:
            return 'normal'
    
    def _calculate_color_confidence(self, image: np.ndarray, 
                                  bg_classification: Dict, 
                                  text_classification: Dict) -> float:
        """전체 색상 분석 신뢰도 계산"""
        bg_conf = bg_classification['confidence']
        text_conf = text_classification['confidence']
        
        # 이미지 품질 요소
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 선명도 (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, sharpness / 500)
        
        # 대비도
        contrast = gray.std()
        contrast_score = min(1.0, contrast / 50)
        
        # 종합 신뢰도
        quality_score = (sharpness_score + contrast_score) / 2
        color_score = (bg_conf + text_conf) / 2
        
        return (quality_score * 0.3 + color_score * 0.7)
    
    def get_color_name(self, rgb: tuple) -> str:
        """RGB 값을 일반적인 색상 이름으로 변환"""
        try:
            return webcolors.rgb_to_name(rgb)
        except ValueError:
            # 가장 가까운 색상 찾기
            min_colors = {}
            for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(key)
                rd = (r_c - rgb[0]) ** 2
                gd = (g_c - rgb[1]) ** 2
                bd = (b_c - rgb[2]) ** 2
                min_colors[(rd + gd + bd)] = name
            return min_colors[min(min_colors.keys())]
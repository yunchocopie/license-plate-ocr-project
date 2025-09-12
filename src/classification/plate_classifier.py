import cv2
import numpy as np
from enum import Enum
from typing import Tuple, Dict, Any, Optional
import re
from .color_analyzer import ColorAnalyzer

"""
한국 번호판 타입 분류 시스템

PRD 요구사항에 따른 9가지 번호판 타입을 색상, 형식, 글자 패턴을 통해 분류합니다.
"""

class PlateType(Enum):
    """번호판 타입 열거형"""
    GENERAL = "일반자가용"           # 흰색 바탕, 검은 글자
    COMMERCIAL = "영업용"           # 노란색 바탕, 검은 글자  
    ELECTRIC = "전기차"             # 하늘색 바탕, 검은 글자, EV 표기
    DIPLOMATIC = "외교관용"         # 남색 바탕, 흰 글자
    MILITARY = "군용"               # 빨강/남색/하늘색 바탕, 흰 글자
    CONSTRUCTION = "건설기계"       # 주황색 바탕, 흰 글자
    MOTORCYCLE = "이륜차"           # 흰색 바탕, 파란 글자
    TEMPORARY = "임시운행"          # 흰색 바탕, 대각선 표시
    SPECIAL = "특수용도"            # 연두색 바탕 등
    UNKNOWN = "미분류"              # 분류 불가

class PlateClassifier:
    """번호판 타입 분류기"""
    
    def __init__(self):
        # 향상된 색상 분석기 초기화
        self.color_analyzer = ColorAnalyzer()
        
        # 번호판 형식 패턴 정의
        self.patterns = {
            'general': r'^\d{2}[가-힣]\d{4}$',                    # 12가3456
            'commercial_with_region': r'^[가-힣]{2}\s\d{2}[가-힣]\d{4}$',  # 서울 12자3456
            'rental': r'^\d{2}[하허호]\d{4}$',                     # 15허6789
            'diplomatic': r'^[가-힣]{2}\s\d{6}$',                  # 외교 123456
            'military': r'^\d{2}[국육해공합]\d+$',                  # 12국1234
            'construction': r'^[가-힣]{2}\s\d{2}-\d{4}$',          # 서울 03-7123
            'motorcycle': r'^[가-힣]{2}\s[가-힣]+\d+[가-힣]\d+$'    # 서울 강남01가1234
        }
        
        # 한글 용도 기호 정의
        self.usage_chars = {
            'general': ['가', '나', '다', '라', '마', '바', '사', '아', '자', '차', '카', '타', '파', '하',
                       '거', '너', '더', '러', '머', '버', '서', '어', '저', '처', '커', '터', '퍼', '허',
                       '고', '노', '도', '로', '모', '보', '소', '오', '조', '초', '코', '토', '포', '호',
                       '구', '누', '두', '루', '무', '부', '수', '우', '주', '추', '쿠', '투', '푸', '후',
                       '그', '느', '드', '르', '므', '브', '스', '으', '즉', '츠', '크', '트', '프', '흐',
                       '기', '니', '디', '리', '미', '비', '시', '이', '지', '치', '키', '티', '피', '히'],
            'commercial': ['아', '바', '사', '자'],
            'rental': ['하', '허', '호'],
            'military': ['국', '육', '해', '공', '합']
        }

    def classify_plate(self, plate_image: np.ndarray, text: str = "") -> Dict[str, Any]:
        """
        번호판 이미지와 텍스트를 분석하여 타입을 분류
        
        Args:
            plate_image: 번호판 이미지 (BGR)
            text: OCR로 추출된 텍스트 (선택사항)
            
        Returns:
            Dict: 분류 결과 정보
        """
        result = {
            'type': PlateType.UNKNOWN,
            'confidence': 0.0,
            'background_color': 'unknown',
            'text_color': 'unknown',
            'features': {},
            'analysis': {}
        }
        
        # 1. 향상된 색상 기반 분석
        color_analysis = self.color_analyzer.analyze_plate_colors(plate_image)
        result['background_color'] = color_analysis['background_color']
        result['text_color'] = color_analysis['text_color']
        result['analysis']['color_analysis'] = color_analysis
        
        # 2. 텍스트 패턴 분석 (OCR 텍스트가 있는 경우)
        text_analysis = {}
        if text:
            text_analysis = self._analyze_text_pattern(text)
            result['analysis']['text_pattern'] = text_analysis
        
        # 3. 특수 표시 탐지
        special_features = self._detect_special_features(plate_image)
        result['features'] = special_features
        
        # 4. 종합 분류 결정
        plate_type, confidence = self._determine_plate_type(
            color_analysis['background_color'], color_analysis['text_color'], 
            text_analysis, special_features, color_analysis
        )
        
        result['type'] = plate_type
        result['confidence'] = confidence
        
        return result
    
    
    def _analyze_text_pattern(self, text: str) -> Dict[str, Any]:
        """텍스트 패턴 분석"""
        analysis = {
            'original_text': text,
            'cleaned_text': re.sub(r'\s+', '', text),  # 공백 제거
            'has_region': False,
            'usage_char': None,
            'pattern_match': None,
            'format_confidence': 0.0
        }
        
        cleaned_text = analysis['cleaned_text']
        
        # 지역명 포함 여부 확인
        for region in ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
                      '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']:
            if region in text:
                analysis['has_region'] = True
                break
        
        # 용도 문자 추출
        korean_chars = re.findall(r'[가-힣]', cleaned_text)
        if korean_chars:
            # 지역명이 아닌 한글 문자 (용도 기호)
            for char in korean_chars:
                if char not in ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
                               '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']:
                    analysis['usage_char'] = char
                    break
        
        # 패턴 매칭
        for pattern_name, pattern in self.patterns.items():
            if re.match(pattern, cleaned_text):
                analysis['pattern_match'] = pattern_name
                analysis['format_confidence'] = 0.9
                break
        
        return analysis
    
    def _detect_special_features(self, image: np.ndarray) -> Dict[str, bool]:
        """특수 표시 탐지 (EV 표기, 대각선, 홀로그램 등)"""
        features = {
            'has_ev_marking': False,
            'has_diagonal_lines': False,
            'has_hologram': False,
            'has_special_symbols': False
        }
        
        # 이미지가 이미 그레이스케일인지 확인
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # EV 표기 탐지 (간단한 템플릿 매칭 또는 텍스트 영역 분석)
        # 실제 구현에서는 더 정교한 알고리즘 필요
        
        # 대각선 탐지 (Hough 변환 사용)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None:
            # 대각선 각도 확인 (45도 근처)
            for rho, theta in lines[0]:
                angle = np.degrees(theta)
                if 30 <= angle <= 60 or 120 <= angle <= 150:
                    features['has_diagonal_lines'] = True
                    break
        
        return features
    
    def _determine_plate_type(self, bg_color: str, text_color: str, 
                            text_analysis: Dict, features: Dict, color_analysis: Dict) -> Tuple[PlateType, float]:
        """종합적인 번호판 타입 결정"""
        
        # 색상 신뢰도 기반 가중치
        color_confidence = color_analysis.get('overall_confidence', 0.5)
        base_confidence = 0.6 + (color_confidence * 0.3)
        
        # 색상 기반 1차 분류 (향상된 색상 분석 결과 사용)
        if bg_color == 'white' and text_color == 'blue':
            return PlateType.MOTORCYCLE, base_confidence + 0.2
        elif bg_color == 'yellow' and text_color in ['black']:
            return PlateType.COMMERCIAL, base_confidence + 0.2
        elif bg_color == 'light_blue':
            # 전기차는 EV 표기나 특수 문양으로 추가 확인
            ev_confidence = 0.3 if features.get('has_ev_marking', False) else 0.1
            return PlateType.ELECTRIC, base_confidence + ev_confidence
        elif bg_color == 'dark_blue' and text_color == 'white':
            return PlateType.DIPLOMATIC, base_confidence + 0.2
        elif bg_color in ['red'] and text_color == 'white':
            return PlateType.MILITARY, base_confidence + 0.2
        elif bg_color == 'orange' and text_color == 'white':
            return PlateType.CONSTRUCTION, base_confidence + 0.2
        elif bg_color == 'green':
            return PlateType.SPECIAL, base_confidence + 0.2
        elif bg_color == 'white' and features.get('has_diagonal_lines', False):
            return PlateType.TEMPORARY, base_confidence + 0.2
        
        # 텍스트 패턴 기반 2차 분류
        if text_analysis:
            usage_char = text_analysis.get('usage_char')
            
            if usage_char in self.usage_chars['rental']:
                return PlateType.GENERAL, 0.7  # 렌터카는 일반 분류에 포함
            elif usage_char in self.usage_chars['commercial']:
                return PlateType.COMMERCIAL, 0.7
            elif usage_char in self.usage_chars['military']:
                return PlateType.MILITARY, 0.7
            elif text_analysis.get('pattern_match') == 'diplomatic':
                return PlateType.DIPLOMATIC, 0.7
        
        # 기본값: 일반 자가용 (가장 흔한 케이스)
        if bg_color == 'white':
            return PlateType.GENERAL, 0.6
        
        return PlateType.UNKNOWN, 0.0
    
    def get_plate_info(self, plate_type: PlateType) -> Dict[str, str]:
        """번호판 타입별 상세 정보 반환"""
        info_map = {
            PlateType.GENERAL: {
                'name': '일반자가용',
                'description': '가장 흔한 승용차·승합차·화물차 등의 비사업용 차량',
                'background': '흰색',
                'text_color': '검은색',
                'format': 'XX가XXXX',
                'example': '12가3456'
            },
            PlateType.COMMERCIAL: {
                'name': '영업용',
                'description': '택시, 버스 등 상업 용도 운행 차량',
                'background': '노란색',
                'text_color': '검은색',
                'format': '지역명 XX자XXXX',
                'example': '서울 12자3456'
            },
            PlateType.ELECTRIC: {
                'name': '전기차',
                'description': '2017년부터 도입된 전기자동차 전용 번호판',
                'background': '하늘색',
                'text_color': '검은색',
                'format': 'XX가XXXX + EV표기',
                'example': '32가1234'
            },
            PlateType.DIPLOMATIC: {
                'name': '외교관용',
                'description': '대사관 및 외교 면책 특권 차량',
                'background': '남색',
                'text_color': '흰색',
                'format': '외교/영사 XXXXXX',
                'example': '외교 123456'
            },
            PlateType.MILITARY: {
                'name': '군용',
                'description': '군 등록 차량 (육군:빨강, 해군:남색, 공군:하늘색)',
                'background': '빨강/남색/하늘색',
                'text_color': '흰색',
                'format': 'XX국XXXX',
                'example': '12육1234'
            },
            PlateType.CONSTRUCTION: {
                'name': '건설기계',
                'description': '굴착기, 덤프트럭 등 건설기계 장비',
                'background': '주황색',
                'text_color': '흰색',
                'format': '지역명 XX-XXXX',
                'example': '서울 03-7123'
            },
            PlateType.MOTORCYCLE: {
                'name': '이륜차',
                'description': '오토바이 및 원동기장치자전거',
                'background': '흰색',
                'text_color': '파란색',
                'format': '지역명 XX가XXXX',
                'example': '서울 강남01가1234'
            },
            PlateType.TEMPORARY: {
                'name': '임시운행',
                'description': '정식 등록 전 임시번호판',
                'background': '흰색 + 대각선',
                'text_color': '검은색',
                'format': '임시XXXXXX',
                'example': '임시123456'
            },
            PlateType.SPECIAL: {
                'name': '특수용도',
                'description': '고가 법인 차량 등 특정 용도',
                'background': '연두색',
                'text_color': '검은색',
                'format': 'XX가XXXX',
                'example': '88가8888'
            }
        }
        
        return info_map.get(plate_type, {
            'name': '미분류',
            'description': '분류할 수 없는 번호판',
            'background': '알 수 없음',
            'text_color': '알 수 없음',
            'format': '알 수 없음',
            'example': '알 수 없음'
        })
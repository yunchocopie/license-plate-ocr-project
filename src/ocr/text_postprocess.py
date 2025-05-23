import re
import string
import config

"""
OCR 결과 후처리 모듈

이 모듈은 OCR 엔진이 인식한 텍스트를 정제하고 교정하는 클래스를 제공합니다.
한국 차량 번호판 형식에 맞게 텍스트를 구조화합니다.
"""
class TextPostProcessor:
    """OCR 인식 결과 텍스트 후처리 클래스"""
    
    def __init__(self, allowed_chars=None):
        """
        TextPostProcessor 클래스 초기화
        
        Args:
            allowed_chars (str, optional): 허용된 문자 집합. 기본값은 config에서 가져옴
        """
        self.allowed_chars = allowed_chars or config.OCR_ALLOWED_CHARS
        
        # 한국어 자동차 번호판 지역명 (시/도)
        self.korean_regions = {
            '서울': '서울', '부산': '부산', '대구': '대구', '인천': '인천',
            '광주': '광주', '대전': '대전', '울산': '울산', '세종': '세종',
            '경기': '경기', '강원': '강원', '충북': '충북', '충남': '충남',
            '전북': '전북', '전남': '전남', '경북': '경북', '경남': '경남', '제주': '제주'
        }
        
        # 비슷한 문자 매핑 (OCR 오인식 교정)
        self.char_correction = {
            '0': 'O', 'O': '0',  # 숫자 0과 문자 O
            '1': 'I', 'I': '1',  # 숫자 1과 문자 I
            '2': 'Z', 'Z': '2',  # 숫자 2와 문자 Z
            '5': 'S', 'S': '5',  # 숫자 5와 문자 S
            '6': 'G', 'G': '6',  # 숫자 6과 문자 G
            '8': 'B', 'B': '8'   # 숫자 8과 문자 B
        }
    
    def process(self, text):
        """
        텍스트 후처리 수행
        
        Args:
            text (str): OCR 인식 결과 텍스트
            
        Returns:
            str: 후처리된 텍스트
        """
        if not text:
            return ""
        
        # 공백 및 특수문자 제거
        processed = self._remove_unwanted_chars(text)
        
        # 허용된 문자만 필터링
        processed = self._filter_allowed_chars(processed)
        
        # 유사 문자 교정 (번호판 형식에 맞게)
        processed = self._correct_similar_chars(processed)
        
        return processed
    
    def _remove_unwanted_chars(self, text):
        """
        원치 않는 문자 제거
        
        Args:
            text (str): 입력 텍스트
            
        Returns:
            str: 정제된 텍스트
        """
        # 공백 제거
        text = text.replace(" ", "")
        
        # 특수문자 제거 (알파벳, 숫자, 한글만 유지)
        pattern = r'[^가-힣0-9a-zA-Z]'
        return re.sub(pattern, '', text)
    
    def _filter_allowed_chars(self, text):
        """
        허용된 문자만 필터링
        
        Args:
            text (str): 입력 텍스트
            
        Returns:
            str: 필터링된 텍스트
        """
        return ''.join(c for c in text if c in self.allowed_chars)
    
    def _correct_similar_chars(self, text):
        """
        비슷한 문자 교정
        
        Args:
            text (str): 입력 텍스트
            
        Returns:
            str: 교정된 텍스트
        """
        # 한국 번호판 패턴 분석
        if len(text) >= 7:  # 최소한 7자 이상 (예: 12가3456)
            # 번호판의 패턴에 따라 교정
            # 새 번호판 형식: 12가3456 (2자리 숫자 + 1자리 한글 + 4자리 숫자)
            # 구 번호판 형식: 서울12가3456 (지역명 + 2자리 숫자 + 1자리 한글 + 4자리 숫자)
            
            # 한글 위치 찾기
            hangul_idx = -1
            for i, c in enumerate(text):
                if '가' <= c <= '힣':
                    hangul_idx = i
                    break
            
            if hangul_idx > 0:
                # 한글 앞은 숫자여야 함
                for i in range(hangul_idx):
                    if text[i] in self.char_correction and text[i].isalpha():
                        text = text[:i] + self.char_correction[text[i]] + text[i+1:]
                
                # 한글 뒤는 숫자여야 함
                for i in range(hangul_idx+1, len(text)):
                    if text[i] in self.char_correction and text[i].isalpha():
                        text = text[:i] + self.char_correction[text[i]] + text[i+1:]
        
        return text
    
    def format_korean_license_plate(self, text):
        """
        한국 차량 번호판 형식에 맞게 포맷팅
        
        Args:
            text (str): 입력 텍스트
            
        Returns:
            str: 번호판 형식으로 포맷팅된 텍스트
        """
        if not text:
            return ""
        
        # 기본 후처리 적용
        processed = self.process(text)
        
        # 번호판 패턴 매칭
        # 1. 신형 번호판 (12가3456 / 123가4567) 패턴 확인
        new_pattern1 = r'(\d{2})([가-힣]{1})(\d{4})'  # 12가3456
        new_pattern2 = r'(\d{3})([가-힣]{1})(\d{4})'  # 123가4567
        
        # 2. 구형 번호판 (서울12가3456) 패턴 확인
        old_pattern = r'([가-힣]{2})(\d{2})([가-힣]{1})(\d{4})'
        
        # 패턴 매칭 및 포맷팅
        new_match1 = re.search(new_pattern1, processed)
        new_match2 = re.search(new_pattern2, processed)
        old_match = re.search(old_pattern, processed)
        
        if new_match1:
            # 12가3456 형식
            area, type_char, number = new_match1.groups()
            return f"{area}{type_char}{number}"
        elif new_match2:
            # 123가4567 형식
            area, type_char, number = new_match2.groups()
            return f"{area}{type_char}{number}"
        elif old_match:
            # 서울12가3456 형식
            region, area, type_char, number = old_match.groups()
            # 지역명이 올바른지 확인
            if region in self.korean_regions:
                return f"{region}{area}{type_char}{number}"
            else:
                # 지역명이 잘못 인식됐을 가능성이 있으므로 제외
                return f"{area}{type_char}{number}"
        else:
            # 패턴 매칭 실패 시 그대로 반환
            return processed
    
    def correct_with_rules(self, text):
        """
        특정 규칙에 기반한 텍스트 교정
        
        Args:
            text (str): 입력 텍스트
            
        Returns:
            str: 규칙에 따라 교정된 텍스트
        """
        if not text:
            return ""
        
        # 번호판 텍스트에서 일반적으로 사용되지 않는 문자 교체
        replacements = {
            'Q': '0',  # Q는 번호판에 거의 사용되지 않음
            'D': '0',  # D와 0 혼동
            'U': '0',  # U와 0 혼동
            'L': '1',  # L과 1 혼동
            'J': '1',  # J와 1 혼동
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # 한글 부분 (가, 나, 다, ...) 교정
        # 일반적으로 번호판의 한글은 특정 위치에 있음
        hangul_positions = []
        for i, c in enumerate(text):
            if '가' <= c <= '힣':
                hangul_positions.append(i)
        
        # 한글이 여러 개 인식된 경우, 가장 신뢰할 수 있는 위치 선택
        if len(hangul_positions) > 1:
            # 번호판 형식 분석으로 가장 적절한 한글 위치 선택
            # (예: 12X3456 형식이면 X 위치의 한글이 맞을 가능성이 높음)
            pass
        
        return text
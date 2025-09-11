#!/usr/bin/env python3
"""
단일 이미지 처리 결과를 위한 Excel 출력 헬퍼

기존 ui/output/output.py의 기능을 확장하여 고급 Excel 출력을 지원합니다.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

try:
    from .excel_exporter import ExcelExporter, OPENPYXL_AVAILABLE
except ImportError:
    try:
        from excel_exporter import ExcelExporter, OPENPYXL_AVAILABLE
    except ImportError:
        OPENPYXL_AVAILABLE = False


def create_single_result_excel(
    image_path: str,
    plate_results: List[Dict[str, Any]], 
    output_path: str,
    include_image: bool = True,
    processing_time: float = 0.0,
    detection_mode: str = "auto"
) -> Optional[str]:
    """단일 이미지 처리 결과를 고급 Excel로 출력
    
    Args:
        image_path: 처리된 이미지 파일 경로
        plate_results: 번호판 인식 결과 리스트
        output_path: 출력 Excel 파일 경로
        include_image: 이미지 포함 여부
        processing_time: 처리 시간 (초)
        detection_mode: 감지 모드
        
    Returns:
        생성된 파일 경로 또는 None (실패 시)
    """
    
    if not OPENPYXL_AVAILABLE:
        print("Excel 출력을 위해서는 openpyxl이 필요합니다: pip install openpyxl")
        return None
    
    try:
        # 단일 결과를 배치 형태로 변환
        excel_result = {
            'file_name': Path(image_path).name,
            'file_path': image_path,
            'success': len(plate_results) > 0,
            'processing_time': processing_time,
            'file_size': Path(image_path).stat().st_size / (1024 * 1024) if os.path.exists(image_path) else 0,
            'plates': [],
            'detection_mode': detection_mode,
            'processed_at': datetime.now().isoformat()
        }
        
        # 번호판 데이터 변환
        for result in plate_results:
            plate_info = {
                'text': result.get('text', result.get('plate_text', '')),
                'confidence': result.get('confidence', result.get('ocr_confidence', 0.0)),
                'type': result.get('type', result.get('plate_type', 'Unknown')),
                'color': result.get('color', result.get('plate_color', 'Unknown')),
                'is_valid': result.get('is_valid', result.get('valid', True))
            }
            excel_result['plates'].append(plate_info)
        
        # 가짜 배치 요약 생성 (단일 파일용)
        class SingleBatchSummary:
            def __init__(self, result):
                self.total_files = 1
                self.processed_files = 1 if result['success'] else 0
                self.success_rate_percent = 100.0 if result['success'] else 0.0
                self.total_plates_detected = len(result['plates'])
                self.total_valid_plates = len([p for p in result['plates'] if p['is_valid']])
                self.throughput_files_per_min = 60.0 / max(result['processing_time'], 0.1)
                self.failed_files = 0 if result['success'] else 1
                
                # 번호판 타입 분포
                self.plate_type_distribution = {}
                for plate in result['plates']:
                    plate_type = plate['type']
                    self.plate_type_distribution[plate_type] = self.plate_type_distribution.get(plate_type, 0) + 1
        
        summary = SingleBatchSummary(excel_result)
        
        # Excel 출력
        exporter = ExcelExporter()
        return exporter.export_batch_results(
            summary,
            [excel_result],
            output_path,
            include_images=include_image,
            include_statistics=True,
            include_charts=len(summary.plate_type_distribution) > 1,  # 타입이 2개 이상일 때만 차트
            image_max_size=(150, 112)  # 단일 이미지이므로 좀 더 크게
        )
        
    except Exception as e:
        print(f"Excel 출력 중 오류 발생: {e}")
        return None


def append_to_enhanced_excel(
    data_list: List[Tuple[str, str, str]],
    excel_path: str,
    include_images: bool = True,
    processing_times: Optional[List[float]] = None
) -> Optional[str]:
    """기존 append_to_excel의 향상된 버전
    
    Args:
        data_list: [(이미지경로, 번호판1, 번호판2), ...] 리스트
        excel_path: 출력 Excel 파일 경로
        include_images: 이미지 포함 여부
        processing_times: 각 이미지의 처리 시간 리스트 (옵션)
        
    Returns:
        생성된 파일 경로 또는 None (실패 시)
    """
    
    if not OPENPYXL_AVAILABLE:
        print("Excel 출력을 위해서는 openpyxl이 필요합니다: pip install openpyxl")
        return None
    
    try:
        # 데이터를 Excel 형태로 변환
        excel_results = []
        
        for idx, item in enumerate(data_list):
            image_path, plate1, plate2 = item
            processing_time = processing_times[idx] if processing_times and idx < len(processing_times) else 0.0
            
            # 번호판 결과 생성
            plates = []
            for plate_text in [plate1, plate2]:
                if plate_text and plate_text.strip():
                    plates.append({
                        'text': plate_text.strip(),
                        'confidence': 0.9,  # 기본값
                        'type': 'Unknown',
                        'color': 'Unknown', 
                        'is_valid': True
                    })
            
            excel_result = {
                'file_name': Path(image_path).name if image_path else f'image_{idx+1}',
                'file_path': image_path,
                'success': len(plates) > 0,
                'processing_time': processing_time,
                'file_size': Path(image_path).stat().st_size / (1024 * 1024) if image_path and os.path.exists(image_path) else 0,
                'plates': plates
            }
            excel_results.append(excel_result)
        
        # 배치 요약 생성
        class MultiBatchSummary:
            def __init__(self, results):
                self.total_files = len(results)
                self.processed_files = len([r for r in results if r['success']])
                self.success_rate_percent = (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0
                self.total_plates_detected = sum(len(r['plates']) for r in results)
                self.total_valid_plates = sum(len([p for p in r['plates'] if p['is_valid']]) for r in results)
                self.failed_files = self.total_files - self.processed_files
                
                # 처리 시간 계산
                processing_times = [r['processing_time'] for r in results if r['processing_time'] > 0]
                avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
                self.throughput_files_per_min = 60.0 / max(avg_time, 0.1)
                
                # 번호판 타입 분포 (기본값으로 설정)
                self.plate_type_distribution = {'일반': self.total_plates_detected}
        
        summary = MultiBatchSummary(excel_results)
        
        # Excel 출력
        exporter = ExcelExporter()
        return exporter.export_batch_results(
            summary,
            excel_results,
            excel_path,
            include_images=include_images,
            include_statistics=True,
            include_charts=False,  # 단순한 데이터이므로 차트 제외
            image_max_size=(120, 90)
        )
        
    except Exception as e:
        print(f"향상된 Excel 출력 중 오류 발생: {e}")
        return None


if __name__ == "__main__":
    # 테스트 코드
    test_results = [
        {
            'text': '12가3456',
            'confidence': 0.95,
            'type': '자가용',
            'color': '흰색',
            'is_valid': True
        }
    ]
    
    output_file = create_single_result_excel(
        "/path/to/test_image.jpg",
        test_results,
        "test_single_result.xlsx",
        include_image=False,
        processing_time=1.23,
        detection_mode="auto"
    )
    
    if output_file:
        print(f"테스트 Excel 파일 생성됨: {output_file}")
    else:
        print("테스트 실패")
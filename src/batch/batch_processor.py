import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import csv
import numpy as np
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import uuid

from ..detection.vehicle_detector import VehicleDetector
from ..detection.plate_detector import create_optimized_plate_detector
from ..preprocessing.image_processor import ImageProcessor
from ..ocr.ocr_engine import OCREngine
from ..utils.system_optimizer import SystemOptimizer
import config

try:
    from ..utils.excel_exporter import ExcelExporter
    EXCEL_EXPORT_AVAILABLE = True
except ImportError:
    EXCEL_EXPORT_AVAILABLE = False

"""
배치 이미지 처리 및 일괄 분석 시스템

대량의 이미지를 효율적으로 처리하고 종합 분석 보고서를 생성:
1. 멀티스레딩/멀티프로세싱 기반 병렬 처리
2. 실시간 진행률 추적 및 결과 수집
3. 오류 처리 및 재시도 메커니즘
4. 종합 통계 분석 및 보고서 생성
5. 메모리 효율적 대용량 파일 처리
"""

@dataclass
class BatchProcessResult:
    """배치 처리 개별 결과"""
    file_path: str
    file_name: str
    file_size_mb: float
    processing_time_sec: float
    success: bool
    error_message: Optional[str] = None
    
    # 검출 결과
    vehicles_detected: int = 0
    plates_detected: int = 0
    
    # OCR 결과
    recognized_texts: List[str] = None
    ocr_confidences: List[float] = None
    
    # 분류 결과
    plate_types: List[str] = None
    plate_colors: List[str] = None
    
    # 유효성 검사
    valid_plates: int = 0
    invalid_plates: int = 0
    
    def __post_init__(self):
        if self.recognized_texts is None:
            self.recognized_texts = []
        if self.ocr_confidences is None:
            self.ocr_confidences = []
        if self.plate_types is None:
            self.plate_types = []
        if self.plate_colors is None:
            self.plate_colors = []

@dataclass 
class BatchProcessSummary:
    """배치 처리 요약"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    total_files: int = 0
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    
    total_processing_time_sec: float = 0.0
    average_time_per_file_sec: float = 0.0
    
    total_vehicles_detected: int = 0
    total_plates_detected: int = 0
    total_valid_plates: int = 0
    
    # 통계
    plate_type_distribution: Dict[str, int] = None
    plate_color_distribution: Dict[str, int] = None
    confidence_statistics: Dict[str, float] = None
    
    # 성능 통계
    throughput_files_per_min: float = 0.0
    success_rate_percent: float = 0.0
    
    def __post_init__(self):
        if self.plate_type_distribution is None:
            self.plate_type_distribution = {}
        if self.plate_color_distribution is None:
            self.plate_color_distribution = {}
        if self.confidence_statistics is None:
            self.confidence_statistics = {}

class BatchProcessor:
    """배치 이미지 처리기"""
    
    def __init__(self, max_workers: Optional[int] = None, use_multiprocessing: bool = False):
        """
        배치 처리기 초기화
        
        Args:
            max_workers: 최대 워커 스레드/프로세스 수 (None이면 자동 설정)
            use_multiprocessing: 멀티프로세싱 사용 여부 (False면 멀티스레딩)
        """
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers or self._get_optimal_workers()
        
        # 시스템 최적화기 연동 (선택사항)
        self.system_optimizer = None
        try:
            self.system_optimizer = SystemOptimizer()
        except Exception as e:
            print(f"시스템 최적화기 초기화 실패 (기본 설정 사용): {e}")
        
        # 모델 초기화 (메인 스레드에서)
        self.vehicle_detector = VehicleDetector()
        self.plate_detector = create_optimized_plate_detector()
        self.image_processor = ImageProcessor()
        self.ocr_engine = OCREngine()
        
        # 처리 상태
        self.is_processing = False
        self.current_session_id = None
        self.progress_callback: Optional[Callable] = None
        self.cancel_requested = False
        
        # 결과 저장
        self.results: List[BatchProcessResult] = []
        self.summary: Optional[BatchProcessSummary] = None
        
        # 지원되는 이미지 형식
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 처리 설정
        self.processing_config = {
            'detection_mode': 'auto',  # 'vehicle_first', 'plate_direct', 'auto'
            'min_confidence': 0.3,
            'max_retry_attempts': 3,
            'chunk_size': 10  # 메모리 관리를 위한 청크 크기
        }
    
    def _get_optimal_workers(self) -> int:
        """시스템에 최적인 워커 수 계산"""
        if self.use_multiprocessing:
            # CPU 코어 수 기준 (I/O 바운드가 아닌 CPU 바운드 작업)
            return min(os.cpu_count() or 4, 8)
        else:
            # I/O 바운드 작업을 고려한 스레드 수
            return min((os.cpu_count() or 4) * 2, 16)
    
    def set_progress_callback(self, callback: Callable[[int, int, Dict], None]):
        """
        진행률 콜백 설정
        
        Args:
            callback: 콜백 함수 (processed_count, total_count, current_result)
        """
        self.progress_callback = callback
    
    def configure_processing(self, **kwargs):
        """처리 설정 변경"""
        self.processing_config.update(kwargs)
    
    def process_directory(self, directory_path: str, recursive: bool = True,
                         output_dir: Optional[str] = None) -> BatchProcessSummary:
        """
        디렉토리 내 모든 이미지 파일을 배치 처리
        
        Args:
            directory_path: 처리할 디렉토리 경로
            recursive: 하위 디렉토리 포함 여부
            output_dir: 결과 저장 디렉토리
            
        Returns:
            처리 요약 결과
        """
        # 이미지 파일 목록 수집
        image_files = self._collect_image_files(directory_path, recursive)
        
        if not image_files:
            raise ValueError(f"처리할 이미지 파일이 없습니다: {directory_path}")
        
        return self.process_files(image_files, output_dir)
    
    def process_files(self, file_paths: List[str], 
                     output_dir: Optional[str] = None) -> BatchProcessSummary:
        """
        파일 목록을 배치 처리
        
        Args:
            file_paths: 처리할 파일 경로 목록
            output_dir: 결과 저장 디렉토리
            
        Returns:
            처리 요약 결과
        """
        if self.is_processing:
            raise RuntimeError("이미 배치 처리가 진행 중입니다.")
        
        # 세션 시작
        self.current_session_id = str(uuid.uuid4())
        self.is_processing = True
        self.cancel_requested = False
        self.results.clear()
        
        start_time = datetime.now()
        
        # 요약 초기화
        self.summary = BatchProcessSummary(
            session_id=self.current_session_id,
            start_time=start_time.isoformat(),
            total_files=len(file_paths)
        )
        
        try:
            print(f"배치 처리 시작: {len(file_paths)}개 파일")
            
            # 청크 단위로 처리 (메모리 효율성)
            chunk_size = self.processing_config['chunk_size']
            
            for i in range(0, len(file_paths), chunk_size):
                if self.cancel_requested:
                    print("배치 처리가 취소되었습니다.")
                    break
                
                chunk = file_paths[i:i + chunk_size]
                chunk_results = self._process_file_chunk(chunk)
                self.results.extend(chunk_results)
                
                # 진행률 업데이트
                if self.progress_callback:
                    progress_data = {
                        'processed': len(self.results),
                        'total': len(file_paths),
                        'current_chunk_size': len(chunk_results)
                    }
                    self.progress_callback(len(self.results), len(file_paths), progress_data)
                
                # 메모리 정리
                if self.system_optimizer:
                    self.system_optimizer._optimize_memory()
            
            # 요약 완성
            end_time = datetime.now()
            self._finalize_summary(end_time)
            
            # 결과 저장
            if output_dir:
                self._save_results(output_dir)
            
            print(f"배치 처리 완료: {self.summary.successful_files}/{self.summary.total_files} 성공")
            
        except Exception as e:
            print(f"배치 처리 오류: {e}")
            if self.summary:
                self.summary.end_time = datetime.now().isoformat()
            raise
        finally:
            self.is_processing = False
        
        return self.summary
    
    def _collect_image_files(self, directory_path: str, recursive: bool) -> List[str]:
        """이미지 파일 목록 수집"""
        image_files = []
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"디렉토리가 존재하지 않습니다: {directory_path}")
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if (file_path.is_file() and 
                file_path.suffix.lower() in self.supported_extensions):
                image_files.append(str(file_path))
        
        # 파일명 순서로 정렬
        image_files.sort()
        return image_files
    
    def _process_file_chunk(self, file_paths: List[str]) -> List[BatchProcessResult]:
        """파일 청크를 병렬 처리"""
        results = []
        
        if self.use_multiprocessing:
            # 멀티프로세싱 (CPU 집약적 작업)
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file_static, file_path, self.processing_config): file_path
                    for file_path in file_paths
                }
                
                for future in as_completed(future_to_file):
                    if self.cancel_requested:
                        break
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        file_path = future_to_file[future]
                        error_result = BatchProcessResult(
                            file_path=file_path,
                            file_name=Path(file_path).name,
                            file_size_mb=0.0,
                            processing_time_sec=0.0,
                            success=False,
                            error_message=str(e)
                        )
                        results.append(error_result)
        else:
            # 멀티스레딩 (I/O 바운드)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file, file_path): file_path
                    for file_path in file_paths
                }
                
                for future in as_completed(future_to_file):
                    if self.cancel_requested:
                        break
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        file_path = future_to_file[future]
                        error_result = BatchProcessResult(
                            file_path=file_path,
                            file_name=Path(file_path).name,
                            file_size_mb=0.0,
                            processing_time_sec=0.0,
                            success=False,
                            error_message=str(e)
                        )
                        results.append(error_result)
        
        return results
    
    def _process_single_file(self, file_path: str) -> BatchProcessResult:
        """단일 파일 처리 (스레딩용)"""
        return self._process_file_core(
            file_path, 
            self.vehicle_detector,
            self.plate_detector, 
            self.image_processor,
            self.ocr_engine,
            self.processing_config
        )
    
    @staticmethod
    def _process_single_file_static(file_path: str, config: Dict) -> BatchProcessResult:
        """단일 파일 처리 (프로세싱용 - static 메서드)"""
        # 각 프로세스에서 모델 새로 초기화
        vehicle_detector = VehicleDetector()
        plate_detector = create_optimized_plate_detector()
        image_processor = ImageProcessor()
        ocr_engine = OCREngine()
        
        return BatchProcessor._process_file_core(
            file_path, vehicle_detector, plate_detector, 
            image_processor, ocr_engine, config
        )
    
    @staticmethod
    def _process_file_core(file_path: str, vehicle_detector, plate_detector, 
                          image_processor, ocr_engine, config: Dict) -> BatchProcessResult:
        """파일 처리 핵심 로직"""
        start_time = time.time()
        file_path_obj = Path(file_path)
        
        try:
            # 파일 정보
            file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
            
            # 이미지 로드
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError("이미지를 읽을 수 없습니다")

            results = []
            vehicle_boxes = []  # 변수 초기화
            
            # 감지 모드에 따른 처리
            if config['detection_mode'] == 'vehicle_first':
                # 차량 먼저 감지
                vehicle_boxes = vehicle_detector.detect(image)
                for vehicle_box in vehicle_boxes:
                    vehicle_image = image[vehicle_box[1]:vehicle_box[3], vehicle_box[0]:vehicle_box[2]]
                    plate_boxes = plate_detector.detect(vehicle_image)
                    
                    for plate_box in plate_boxes:
                        plate_image = vehicle_image[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]
                        result = BatchProcessor._process_plate(
                            plate_image, image_processor, ocr_engine, config
                        )
                        results.append(result)
            
            elif config['detection_mode'] == 'plate_direct':
                # 번호판 직접 감지
                plate_boxes = plate_detector.detect(image)
                for plate_box in plate_boxes:
                    plate_image = image[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]
                    result = BatchProcessor._process_plate(
                        plate_image, image_processor, ocr_engine, config
                    )
                    results.append(result)
            
            else:  # auto 모드
                # 자동 감지 (번호판 우선, 실패시 차량 경유)
                plate_boxes = plate_detector.detect(image)
                if plate_boxes:
                    for plate_box in plate_boxes:
                        plate_image = image[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]
                        result = BatchProcessor._process_plate(
                            plate_image, image_processor, ocr_engine, config
                        )
                        results.append(result)
                else:
                    # 번호판이 직접 감지되지 않으면 차량 경유
                    vehicle_boxes = vehicle_detector.detect(image)
                    for vehicle_box in vehicle_boxes:
                        vehicle_image = image[vehicle_box[1]:vehicle_box[3], vehicle_box[0]:vehicle_box[2]]
                        plate_boxes = plate_detector.detect(vehicle_image)
                        
                        for plate_box in plate_boxes:
                            plate_image = vehicle_image[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]
                            result = BatchProcessor._process_plate(
                                plate_image, image_processor, ocr_engine, config
                            )
                            results.append(result)
            
            # 결과 집계
            recognized_texts = []
            ocr_confidences = []
            plate_types = []
            plate_colors = []
            valid_plates = 0
            
            for result in results:
                if result.get('success', False) and result.get('text'):
                    recognized_texts.append(result['text'])
                    ocr_confidences.append(result.get('confidence', 0.0))

                    # 안전한 딕셔너리 접근
                    classification = result.get('classification', {})
                    plate_type = classification.get('type')
                    if plate_type:
                        plate_types.append(plate_type.value)
                    plate_colors.append(classification.get('background_color', 'unknown'))

                    validation = result.get('validation', {})
                    if validation.get('is_valid', False):
                        valid_plates += 1
            
            processing_time = time.time() - start_time
            
            return BatchProcessResult(
                file_path=file_path,
                file_name=file_path_obj.name,
                file_size_mb=file_size_mb,
                processing_time_sec=processing_time,
                success=True,
                vehicles_detected=len(vehicle_boxes),
                plates_detected=len(results),
                recognized_texts=recognized_texts,
                ocr_confidences=ocr_confidences,
                plate_types=plate_types,
                plate_colors=plate_colors,
                valid_plates=valid_plates,
                invalid_plates=len(results) - valid_plates
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return BatchProcessResult(
                file_path=file_path,
                file_name=file_path_obj.name,
                file_size_mb=file_path_obj.stat().st_size / (1024 * 1024) if file_path_obj.exists() else 0.0,
                processing_time_sec=processing_time,
                success=False,
                error_message=str(e)
            )
    
    @staticmethod
    def _process_plate(plate_image, image_processor, ocr_engine, config: Dict) -> Dict:
        """번호판 이미지 처리"""
        try:
            # 전처리
            processed_plate = image_processor.process(plate_image)
            
            # OCR 및 분류
            result = ocr_engine.recognize_with_classification(
                processed_plate,
                min_confidence=config.get('min_confidence', 0.3)
            )
            
            return {
                'success': True,
                'text': result['text'],
                'confidence': result['confidence'],
                'classification': result['classification'],
                'validation': result['validation']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _finalize_summary(self, end_time: datetime):
        """요약 정보 완성"""
        if not self.summary:
            return
        
        self.summary.end_time = end_time.isoformat()
        self.summary.processed_files = len(self.results)
        self.summary.successful_files = sum(1 for r in self.results if r.success)
        self.summary.failed_files = self.summary.processed_files - self.summary.successful_files
        
        # 시간 통계
        total_time = sum(r.processing_time_sec for r in self.results)
        self.summary.total_processing_time_sec = total_time
        
        if self.summary.processed_files > 0:
            self.summary.average_time_per_file_sec = total_time / self.summary.processed_files
            
        # 처리량 계산
        duration_minutes = (end_time - datetime.fromisoformat(self.summary.start_time)).total_seconds() / 60
        if duration_minutes > 0:
            self.summary.throughput_files_per_min = self.summary.processed_files / duration_minutes
        
        # 성공률
        if self.summary.total_files > 0:
            self.summary.success_rate_percent = (self.summary.successful_files / self.summary.total_files) * 100
        
        # 검출 통계
        self.summary.total_vehicles_detected = sum(r.vehicles_detected for r in self.results if r.success)
        self.summary.total_plates_detected = sum(r.plates_detected for r in self.results if r.success)
        self.summary.total_valid_plates = sum(r.valid_plates for r in self.results if r.success)
        
        # 분포 통계
        plate_types = {}
        plate_colors = {}
        all_confidences = []
        
        for result in self.results:
            if result.success:
                for plate_type in result.plate_types:
                    plate_types[plate_type] = plate_types.get(plate_type, 0) + 1
                
                for plate_color in result.plate_colors:
                    plate_colors[plate_color] = plate_colors.get(plate_color, 0) + 1
                
                all_confidences.extend(result.ocr_confidences)
        
        self.summary.plate_type_distribution = plate_types
        self.summary.plate_color_distribution = plate_colors
        
        # 신뢰도 통계
        if all_confidences:
            self.summary.confidence_statistics = {
                'mean': float(np.mean(all_confidences)),
                'median': float(np.median(all_confidences)),
                'std': float(np.std(all_confidences)),
                'min': float(np.min(all_confidences)),
                'max': float(np.max(all_confidences))
            }
    
    def _save_results(self, output_dir: str):
        """결과를 파일로 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id_short = self.current_session_id[:8]
        
        # JSON 상세 결과
        json_file = output_path / f"batch_results_{timestamp}_{session_id_short}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            data = {
                'summary': asdict(self.summary),
                'detailed_results': [asdict(r) for r in self.results]
            }
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # CSV 요약
        csv_file = output_path / f"batch_summary_{timestamp}_{session_id_short}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                '파일명', '파일크기(MB)', '처리시간(초)', '성공여부', 
                '차량수', '번호판수', '인식텍스트', 'OCR신뢰도', 
                '번호판타입', '번호판색상', '유효번호판수', '오류메시지'
            ])
            
            for result in self.results:
                writer.writerow([
                    result.file_name,
                    f"{result.file_size_mb:.2f}",
                    f"{result.processing_time_sec:.3f}",
                    '성공' if result.success else '실패',
                    result.vehicles_detected,
                    result.plates_detected,
                    '; '.join(result.recognized_texts),
                    '; '.join([f"{c:.3f}" for c in result.ocr_confidences]),
                    '; '.join(result.plate_types),
                    '; '.join(result.plate_colors),
                    result.valid_plates,
                    result.error_message or ''
                ])
        
        print(f"결과 저장 완료:")
        print(f"  - 상세 결과: {json_file}")
        print(f"  - 요약 CSV: {csv_file}")
        
        # Excel 출력 기능이 활성화되어 있는 경우 Excel 파일도 생성
        if EXCEL_EXPORT_AVAILABLE:
            self._save_excel_results(output_dir)
    
    def _save_excel_results(self, output_dir: str):
        """Excel 형태로 결과 저장"""
        try:
            excel_file = os.path.join(output_dir, f"batch_results_comprehensive_{self.current_session_id}.xlsx")
            
            # BatchProcessResult를 ExcelExporter가 예상하는 형태로 변환
            excel_results = []
            for result in self.results:
                plates_data = []
                
                # 번호판 정보를 딕셔너리 형태로 변환
                for i, text in enumerate(result.recognized_texts):
                    plate_info = {
                        'text': text,
                        'confidence': result.ocr_confidences[i] if i < len(result.ocr_confidences) else 0.0,
                        'type': result.plate_types[i] if i < len(result.plate_types) else 'Unknown',
                        'color': result.plate_colors[i] if i < len(result.plate_colors) else 'Unknown',
                        'is_valid': i < result.valid_plates
                    }
                    plates_data.append(plate_info)
                
                excel_result = {
                    'file_name': result.file_name,
                    'file_path': result.file_path,
                    'success': result.success,
                    'processing_time': result.processing_time_sec,
                    'file_size': result.file_size_mb,
                    'plates': plates_data,
                    'error': result.error_message
                }
                excel_results.append(excel_result)
            
            # Excel 출력기 생성 및 보고서 생성
            exporter = ExcelExporter()
            exporter.export_batch_results(
                self.summary,
                excel_results,
                excel_file,
                include_images=True,
                include_statistics=True,
                include_charts=True,
                image_max_size=(100, 75)
            )
            
            print(f"  - 종합 Excel: {excel_file}")
            
        except Exception as e:
            print(f"Excel 출력 중 오류 발생: {e}")
    
    def export_to_excel(
        self,
        output_path: str,
        include_images: bool = True,
        include_statistics: bool = True,
        include_charts: bool = True,
        **kwargs
    ) -> Optional[str]:
        """처리 결과를 고급 Excel 파일로 출력
        
        Args:
            output_path: 출력 파일 경로
            include_images: 이미지 포함 여부
            include_statistics: 통계 시트 포함 여부
            include_charts: 차트 포함 여부
            **kwargs: 추가 옵션들
            
        Returns:
            생성된 파일 경로 또는 None (실패 시)
        """
        
        if not EXCEL_EXPORT_AVAILABLE:
            print("Excel 출력 기능을 사용할 수 없습니다. openpyxl을 설치해주세요.")
            return None
        
        if not self.results or not self.summary:
            print("내보낼 결과가 없습니다. 먼저 배치 처리를 실행해주세요.")
            return None
        
        try:
            # 결과 데이터 변환
            excel_results = []
            for result in self.results:
                plates_data = []
                
                for i, text in enumerate(result.recognized_texts):
                    plate_info = {
                        'text': text,
                        'confidence': result.ocr_confidences[i] if i < len(result.ocr_confidences) else 0.0,
                        'type': result.plate_types[i] if i < len(result.plate_types) else 'Unknown',
                        'color': result.plate_colors[i] if i < len(result.plate_colors) else 'Unknown',
                        'is_valid': i < result.valid_plates
                    }
                    plates_data.append(plate_info)
                
                excel_result = {
                    'file_name': result.file_name,
                    'file_path': result.file_path,
                    'success': result.success,
                    'processing_time': result.processing_time_sec,
                    'file_size': result.file_size_mb,
                    'plates': plates_data,
                    'error': result.error_message
                }
                excel_results.append(excel_result)
            
            # Excel 출력
            exporter = ExcelExporter()
            return exporter.export_batch_results(
                self.summary,
                excel_results,
                output_path,
                include_images=include_images,
                include_statistics=include_statistics,
                include_charts=include_charts,
                **kwargs
            )
            
        except Exception as e:
            print(f"Excel 출력 중 오류 발생: {e}")
            return None
    
    def cancel_processing(self):
        """배치 처리 취소"""
        if self.is_processing:
            self.cancel_requested = True
            print("배치 처리 취소 요청됨")
    
    def get_progress(self) -> Dict:
        """현재 진행 상황 조회"""
        if not self.is_processing:
            return {"status": "idle"}
        
        return {
            "status": "processing",
            "session_id": self.current_session_id,
            "processed": len(self.results),
            "total": self.summary.total_files if self.summary else 0,
            "success_rate": (sum(1 for r in self.results if r.success) / len(self.results) * 100) if self.results else 0
        }
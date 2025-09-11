#!/usr/bin/env python3
"""
배치 번호판 OCR 처리 도구

대량의 이미지 파일을 효율적으로 처리하고 결과를 생성하는 커맨드라인 도구입니다.

사용법:
    python batch_ocr.py --input /path/to/images --output /path/to/results
    python batch_ocr.py --input /path/to/images --recursive --workers 8
    python batch_ocr.py --config batch_config.json
"""

import argparse
import json
import time
from pathlib import Path
import signal
import sys

from src.batch.batch_processor import BatchProcessor, BatchProcessSummary
from src.utils.system_optimizer import SystemOptimizer

def create_progress_callback(show_details=False):
    """진행률 콜백 생성"""
    last_update_time = [0]  # 리스트로 감싸서 closure에서 수정 가능하게
    
    def progress_callback(processed: int, total: int, data: dict):
        current_time = time.time()
        
        # 1초마다 또는 완료시에만 출력
        if current_time - last_update_time[0] >= 1.0 or processed == total:
            progress_percent = (processed / total * 100) if total > 0 else 0
            
            if show_details:
                print(f"\r진행률: {processed}/{total} ({progress_percent:.1f}%) | "
                      f"청크: {data.get('current_chunk_size', 0)}개", end='', flush=True)
            else:
                print(f"\r진행률: {processed}/{total} ({progress_percent:.1f}%)", end='', flush=True)
            
            last_update_time[0] = current_time
            
            # 완료시 줄바꿈
            if processed == total:
                print()
    
    return progress_callback

def load_config_file(config_path: str) -> dict:
    """설정 파일 로드"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"설정 파일 로드 오류: {e}")
        return {}

def create_sample_config(output_path: str):
    """샘플 설정 파일 생성"""
    sample_config = {
        "input_directory": "/path/to/input/images",
        "output_directory": "/path/to/output/results",
        "recursive": True,
        "processing_options": {
            "detection_mode": "auto",
            "preprocessing_mode": "balanced",
            "min_confidence": 0.3,
            "max_retry_attempts": 3,
            "chunk_size": 10
        },
        "performance_options": {
            "max_workers": 4,
            "use_multiprocessing": False,
            "enable_system_optimization": True
        },
        "output_options": {
            "save_detailed_json": True,
            "save_summary_csv": True,
            "save_failed_list": True
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, ensure_ascii=False, indent=2)
    
    print(f"샘플 설정 파일 생성: {output_path}")

def print_system_info():
    """시스템 정보 출력"""
    try:
        optimizer = SystemOptimizer()
        report = optimizer.get_system_report()
        
        print("=== 시스템 정보 ===")
        resources = report['system_resources']
        print(f"CPU: {resources['cpu']['total_cores']}코어 (사용률: {resources['cpu']['current_usage']:.1f}%)")
        print(f"메모리: {resources['memory']['total_mb']:.0f}MB (사용률: {resources['memory']['current_usage_percent']:.1f}%)")
        
        if resources['gpu']['available']:
            print(f"GPU: 사용가능 ({resources['gpu']['count']}개)")
            print(f"GPU 메모리: {resources['gpu']['memory_total_mb']:.0f}MB")
        else:
            print("GPU: 사용불가")
        
        current_profile = report['current_profile']
        print(f"성능 모드: {current_profile['name']}")
        print()
        
    except Exception as e:
        print(f"시스템 정보 조회 실패: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="배치 번호판 OCR 처리 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python batch_ocr.py --input ./images --output ./results
  python batch_ocr.py --input ./images --recursive --workers 8
  python batch_ocr.py --config my_config.json
  python batch_ocr.py --create-config sample_config.json
        """
    )
    
    # 기본 옵션
    parser.add_argument('--input', '-i', type=str, help='입력 디렉토리 경로')
    parser.add_argument('--output', '-o', type=str, help='출력 디렉토리 경로')
    parser.add_argument('--recursive', '-r', action='store_true', help='하위 디렉토리 포함')
    
    # 성능 옵션
    parser.add_argument('--workers', '-w', type=int, help='워커 스레드/프로세스 수')
    parser.add_argument('--multiprocessing', '-mp', action='store_true', help='멀티프로세싱 사용')
    
    # 처리 옵션
    parser.add_argument('--detection-mode', choices=['auto', 'vehicle_first', 'plate_direct'], 
                       default='auto', help='감지 모드')
    parser.add_argument('--preprocessing-mode', choices=['fast', 'balanced', 'high_quality'], 
                       default='balanced', help='전처리 모드')
    parser.add_argument('--min-confidence', type=float, default=0.3, help='최소 OCR 신뢰도')
    
    # 설정 파일 옵션
    parser.add_argument('--config', '-c', type=str, help='설정 파일 경로')
    parser.add_argument('--create-config', type=str, help='샘플 설정 파일 생성')
    
    # 기타 옵션
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 출력')
    parser.add_argument('--system-info', action='store_true', help='시스템 정보 출력')
    parser.add_argument('--no-optimization', action='store_true', help='시스템 최적화 비활성화')
    
    # 출력 옵션
    parser.add_argument('--excel-output', action='store_true', help='Excel 종합 보고서 생성')
    parser.add_argument('--include-images', action='store_true', help='Excel에 이미지 포함 (용량 증가)')
    parser.add_argument('--no-charts', action='store_true', help='Excel 차트 생성 비활성화')
    
    args = parser.parse_args()
    
    # 시스템 정보 출력
    if args.system_info:
        print_system_info()
        if not args.input:  # 시스템 정보만 출력하고 종료
            return
    
    # 샘플 설정 파일 생성
    if args.create_config:
        create_sample_config(args.create_config)
        return
    
    # 설정 로드
    config = {}
    if args.config:
        config = load_config_file(args.config)
    
    # 커맨드라인 인자가 설정 파일보다 우선
    input_dir = args.input or config.get('input_directory')
    output_dir = args.output or config.get('output_directory')
    
    if not input_dir:
        parser.error("입력 디렉토리가 지정되지 않았습니다. --input 또는 --config를 사용하세요.")
    
    if not output_dir:
        output_dir = Path(input_dir).parent / "batch_results"
        print(f"출력 디렉토리가 지정되지 않아 기본값 사용: {output_dir}")
    
    # 처리 옵션 설정
    processing_config = config.get('processing_options', {})
    processing_config.update({
        'detection_mode': args.detection_mode,
        'preprocessing_mode': args.preprocessing_mode,
        'min_confidence': args.min_confidence
    })
    
    # 성능 옵션 설정
    performance_config = config.get('performance_options', {})
    max_workers = args.workers or performance_config.get('max_workers')
    use_multiprocessing = args.multiprocessing or performance_config.get('use_multiprocessing', False)
    
    recursive = args.recursive or config.get('recursive', True)
    
    print(f"배치 OCR 처리 시작")
    print(f"입력: {input_dir} (재귀: {recursive})")
    print(f"출력: {output_dir}")
    print(f"감지 모드: {processing_config['detection_mode']}")
    print(f"전처리 모드: {processing_config['preprocessing_mode']}")
    print(f"워커 수: {max_workers or '자동'}")
    print(f"처리 방식: {'멀티프로세싱' if use_multiprocessing else '멀티스레딩'}")
    print()
    
    # 배치 프로세서 초기화
    try:
        processor = BatchProcessor(
            max_workers=max_workers,
            use_multiprocessing=use_multiprocessing
        )
        
        # 처리 설정 적용
        processor.configure_processing(**processing_config)
        
        # 진행률 콜백 설정
        progress_callback = create_progress_callback(show_details=args.verbose)
        processor.set_progress_callback(progress_callback)
        
        # 중단 처리 (Ctrl+C)
        def signal_handler(sig, frame):
            print("\n\n중단 신호 감지, 처리를 중지합니다...")
            processor.cancel_processing()
            print("정리 작업 완료")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # 배치 처리 실행
        start_time = time.time()
        
        summary = processor.process_directory(
            input_dir, 
            recursive=recursive,
            output_dir=output_dir
        )
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 결과 출력
        print(f"\n=== 배치 처리 완료 ===")
        print(f"총 처리 시간: {total_duration:.2f}초")
        print(f"처리된 파일: {summary.processed_files}/{summary.total_files}")
        print(f"성공률: {summary.success_rate_percent:.1f}%")
        print(f"평균 처리 속도: {summary.throughput_files_per_min:.1f}파일/분")
        
        if summary.total_plates_detected > 0:
            print(f"총 감지된 번호판: {summary.total_plates_detected}개")
            print(f"유효한 번호판: {summary.total_valid_plates}개")
            
            if summary.plate_type_distribution:
                print(f"번호판 타입 분포:")
                for plate_type, count in sorted(summary.plate_type_distribution.items()):
                    print(f"  - {plate_type}: {count}개")
        
        if summary.failed_files > 0:
            print(f"\n⚠️ 실패한 파일: {summary.failed_files}개")
            
        print(f"\n결과가 저장되었습니다: {output_dir}")
        
        # Excel 출력 (요청시)
        if args.excel_output:
            try:
                print("\n📈 Excel 종합 보고서 생성 중...")
                excel_file = str(Path(output_dir) / f"comprehensive_report_{int(time.time())}.xlsx")
                
                result_file = processor.export_to_excel(
                    excel_file,
                    include_images=args.include_images,
                    include_statistics=True,
                    include_charts=not args.no_charts,
                    image_max_size=(120, 90) if args.include_images else None
                )
                
                if result_file:
                    print(f"✅ Excel 보고서 생성 완료: {result_file}")
                    file_size = Path(result_file).stat().st_size / (1024 * 1024)
                    print(f"   파일 크기: {file_size:.2f}MB")
                else:
                    print("❌ Excel 보고서 생성 실패")
                    
            except Exception as e:
                print(f"❌ Excel 보고서 생성 중 오류: {e}")
        
        # 성능 권장사항 (verbose 모드)
        if args.verbose and not args.no_optimization:
            try:
                optimizer = SystemOptimizer()
                recommendations = optimizer.get_optimization_recommendations()
                if recommendations:
                    print(f"\n💡 성능 최적화 권장사항:")
                    for rec in recommendations[:3]:
                        print(f"  {rec}")
            except:
                pass
        
    except KeyboardInterrupt:
        print("\n배치 처리가 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n배치 처리 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
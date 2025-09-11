#!/usr/bin/env python3
"""
한국 번호판 특화 YOLOv8 학습 스크립트

PRD 요구사항에 따른 한국 번호판 9가지 타입 학습:
- 시스템 자동 최적화
- 클래스 불균형 처리
- 한국 번호판 특성 반영
- 로컬 환경 성능 최적화
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

from src.detection.korean_plate_optimizer import KoreanPlateOptimizer
from src.training.dataset_manager import KoreanPlateDatasetManager

def main():
    parser = argparse.ArgumentParser(description='한국 번호판 특화 YOLO 모델 학습')
    parser.add_argument('--data', type=str, help='데이터셋 YAML 파일 경로')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='기본 모델 (yolov8n/s/m/l/x.pt)')
    parser.add_argument('--epochs', type=int, default=100, help='학습 에포크 수')
    parser.add_argument('--batch-size', type=int, help='배치 크기 (자동 설정시 생략)')
    parser.add_argument('--imgsz', type=int, help='이미지 크기 (자동 설정시 생략)')
    parser.add_argument('--project', type=str, default='korean_plate_training', help='프로젝트 이름')
    parser.add_argument('--name', type=str, default='korean_specialized', help='실험 이름')
    parser.add_argument('--device', type=str, help='디바이스 (자동 설정시 생략)')
    parser.add_argument('--resume', type=str, help='학습 재개할 체크포인트 경로')
    parser.add_argument('--pretrained', action='store_true', help='사전 학습된 가중치 사용')
    parser.add_argument('--analyze-only', action='store_true', help='데이터셋 분석만 실행')
    parser.add_argument('--dataset-root', type=str, help='데이터셋 루트 디렉토리')
    
    args = parser.parse_args()
    
    print("=== 한국 번호판 특화 YOLOv8 학습 시스템 ===")
    
    # 1. 시스템 최적화기 초기화
    print("\n1. 시스템 분석 및 최적화...")
    optimizer = KoreanPlateOptimizer()
    print(optimizer.get_system_recommendation())
    
    # 2. 데이터셋 매니저 초기화
    print("\n2. 데이터셋 분석...")
    dataset_manager = KoreanPlateDatasetManager(args.dataset_root)
    
    # 데이터셋 유효성 검사
    validation_results = dataset_manager.validate_dataset()
    if not validation_results['is_valid']:
        print("❌ 데이터셋 검증 실패:")
        for error in validation_results['errors']:
            print(f"   - {error}")
        return
    
    # 데이터셋 분석
    stats = dataset_manager.analyze_dataset()
    print(f"✅ 데이터셋 분석 완료:")
    print(f"   - 총 이미지: {stats['total_images']}")
    print(f"   - 총 라벨: {stats['total_labels']}")
    print(f"   - 학습/검증/테스트: {stats['split_distribution']}")
    print(f"   - 클래스 분포: {stats['class_distribution']}")
    
    if validation_results['warnings']:
        print("\n⚠️ 경고:")
        for warning in validation_results['warnings']:
            print(f"   - {warning}")
    
    # 학습 권장사항 표시
    print("\n📋 학습 권장사항:")
    recommendations = dataset_manager.get_training_recommendations()
    for rec in recommendations:
        print(f"   {rec}")
    
    # 분석만 실행하는 경우 여기서 종료
    if args.analyze_only:
        print("\n데이터셋 분석 완료. 학습을 실행하려면 --analyze-only 플래그를 제거하세요.")
        return
    
    # 3. 데이터셋 YAML 파일 설정
    if not args.data:
        print("\n3. 데이터셋 설정 파일 생성...")
        dataset_yaml = dataset_manager.create_yolo_dataset_config("korean_plates_v1")
        print(f"✅ 데이터셋 설정 생성: {dataset_yaml}")
    else:
        dataset_yaml = args.data
        if not Path(dataset_yaml).exists():
            print(f"❌ 데이터셋 파일이 없습니다: {dataset_yaml}")
            return
    
    # 4. 학습 설정 생성
    print("\n4. 학습 설정 최적화...")
    training_config = optimizer.create_training_config(
        data_yaml_path=dataset_yaml,
        epochs=args.epochs,
        patience=min(50, args.epochs // 2)
    )
    
    # 사용자 지정 파라미터 덮어쓰기
    if args.batch_size:
        training_config['batch'] = args.batch_size
    if args.imgsz:
        training_config['imgsz'] = args.imgsz
    if args.device:
        training_config['device'] = args.device
    if args.project:
        training_config['project'] = args.project
    if args.name:
        training_config['name'] = args.name
    
    print(f"✅ 학습 설정:")
    key_configs = ['batch', 'imgsz', 'device', 'lr0', 'epochs']
    for key in key_configs:
        if key in training_config:
            print(f"   - {key}: {training_config[key]}")
    
    # 5. 모델 초기화
    print(f"\n5. 모델 초기화...")
    if args.resume:
        print(f"학습 재개: {args.resume}")
        model = YOLO(args.resume)
    else:
        model_path = args.model if args.pretrained else args.model.replace('.pt', '.yaml')
        print(f"기본 모델 로드: {model_path} (pretrained: {args.pretrained})")
        model = YOLO(model_path if not args.pretrained else args.model)
    
    # 한국 번호판 특화 최적화 적용
    model = optimizer.optimize_for_inference(model)
    
    # 6. 학습 시작
    print(f"\n6. 학습 시작...")
    print(f"학습 설정이 저장될 위치: {training_config['project']}/{training_config['name']}")
    print(f"데이터셋: {dataset_yaml}")
    
    try:
        results = model.train(**training_config)
        print("\n✅ 학습 완료!")
        
        # 7. 학습 결과 요약
        print("\n7. 학습 결과 요약:")
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"   - 최종 mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"   - 최종 mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        
        # 최적 가중치 경로 출력
        best_weights = Path(training_config['project']) / training_config['name'] / 'weights' / 'best.pt'
        if best_weights.exists():
            print(f"   - 최적 가중치: {best_weights}")
        
        # 8. 모델 벤치마크 (선택사항)
        print("\n8. 학습된 모델 성능 테스트...")
        trained_model = YOLO(str(best_weights) if best_weights.exists() else model.ckpt_path)
        benchmark_results = optimizer.benchmark_model(trained_model)
        print(f"   - 평균 추론 시간: {benchmark_results['avg_inference_time']:.3f}초")
        print(f"   - 예상 FPS: {benchmark_results['fps']:.1f}")
        print(f"   - GPU 메모리 사용량: {benchmark_results['avg_memory_mb']:.1f}MB")
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        return
    
    print("\n🎉 한국 번호판 특화 모델 학습이 완료되었습니다!")
    print(f"모델 파일: {best_weights if best_weights.exists() else 'runs/train/exp/weights/best.pt'}")
    print("다음 단계:")
    print("1. 모델 성능을 검증하세요 (validation set)")
    print("2. 테스트 데이터로 최종 평가하세요")
    print("3. config.py에서 PLATE_DETECTION_MODEL 경로를 업데이트하세요")

if __name__ == "__main__":
    main()
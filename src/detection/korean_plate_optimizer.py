import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psutil
import time
import cv2
import config

"""
한국 번호판 특화 YOLOv8 모델 최적화 시스템

PRD 요구사항에 따른 한국 번호판 특성을 고려한 YOLO 모델 최적화:
1. 한국 번호판 크기 비율 (가로:세로 = 약 4:1) 최적화
2. 9가지 번호판 타입별 색상 특성 반영
3. 로컬 환경 성능 최적화 (CPU/GPU 자동 감지)
4. 실시간 추론 성능 향상
"""

class KoreanPlateOptimizer:
    """한국 번호판 특화 YOLO 모델 최적화기"""
    
    def __init__(self):
        self.system_info = self._analyze_system()
        self.optimized_configs = self._generate_optimized_configs()
        
    def _analyze_system(self) -> Dict:
        """시스템 사양 분석"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
        }
    
    def _generate_optimized_configs(self) -> Dict:
        """시스템 사양에 따른 최적화 설정 생성"""
        configs = {}
        
        # CPU 전용 환경
        if not self.system_info['cuda_available']:
            configs['cpu_optimized'] = {
                'model_size': 'yolov8n.pt',  # nano 모델로 경량화
                'imgsz': 416,  # 이미지 크기 축소
                'batch_size': 1,
                'workers': min(4, self.system_info['cpu_count']),
                'half': False,
                'device': 'cpu',
                'conf': 0.15,  # 낮은 confidence로 더 많은 검출
                'iou': 0.3
            }
        
        # GPU 환경 (메모리에 따른 차등 설정)
        else:
            gpu_memory = self.system_info['gpu_memory']
            
            if gpu_memory >= 8:  # 고성능 GPU (8GB+)
                configs['gpu_high'] = {
                    'model_size': 'yolov8s.pt',
                    'imgsz': 640,
                    'batch_size': 4,
                    'workers': 8,
                    'half': True,
                    'device': 0,
                    'conf': 0.01,
                    'iou': 0.45
                }
            elif gpu_memory >= 4:  # 중급 GPU (4-8GB)
                configs['gpu_medium'] = {
                    'model_size': 'yolov8s.pt',
                    'imgsz': 512,
                    'batch_size': 2,
                    'workers': 4,
                    'half': True,
                    'device': 0,
                    'conf': 0.05,
                    'iou': 0.4
                }
            else:  # 저사양 GPU (2-4GB)
                configs['gpu_low'] = {
                    'model_size': 'yolov8n.pt',
                    'imgsz': 416,
                    'batch_size': 1,
                    'workers': 2,
                    'half': True,
                    'device': 0,
                    'conf': 0.1,
                    'iou': 0.35
                }
        
        return configs
    
    def get_optimal_config(self) -> Dict:
        """현재 시스템에 최적화된 설정 반환"""
        if not self.system_info['cuda_available']:
            return self.optimized_configs['cpu_optimized']
        
        gpu_memory = self.system_info['gpu_memory']
        if gpu_memory >= 8:
            return self.optimized_configs.get('gpu_high', {})
        elif gpu_memory >= 4:
            return self.optimized_configs.get('gpu_medium', {})
        else:
            return self.optimized_configs.get('gpu_low', {})
    
    def create_korean_plate_model(self, base_model_path: Optional[str] = None) -> YOLO:
        """
        한국 번호판 특화 모델 생성
        
        Args:
            base_model_path: 기본 모델 경로 (없으면 최적화된 크기 자동 선택)
            
        Returns:
            최적화된 YOLO 모델
        """
        config = self.get_optimal_config()
        
        if base_model_path:
            model = YOLO(base_model_path)
        else:
            # 시스템 최적화된 모델 크기 선택
            model_size = config.get('model_size', 'yolov8s.pt')
            model = YOLO(model_size)
        
        # 한국 번호판 특화 설정 적용
        self._apply_korean_plate_optimizations(model, config)
        
        return model
    
    def _apply_korean_plate_optimizations(self, model: YOLO, config: Dict):
        """한국 번호판 특화 최적화 적용"""
        
        # 모델 설정 업데이트
        if hasattr(model.model, 'yaml'):
            # 한국 번호판 앵커 비율 최적화 (4:1 비율)
            korean_plate_ratios = [2.0, 4.0, 6.0]  # 가로가 긴 형태
            
            # NMS 설정 최적화 (겹치는 번호판 검출 방지)
            model.overrides.update({
                'iou': config.get('iou', 0.4),
                'conf': config.get('conf', 0.01),
                'max_det': 10,  # 최대 검출 수 제한 (한 이미지에서 번호판은 많지 않음)
                'agnostic_nms': False,
                'classes': None  # 모든 클래스 허용
            })
    
    def optimize_for_inference(self, model: YOLO) -> YOLO:
        """추론 성능 최적화"""
        config = self.get_optimal_config()
        
        # Half precision 최적화 (GPU만)
        if config.get('half', False) and torch.cuda.is_available():
            model.model.half()
        
        # 모델을 evaluation 모드로 전환
        model.model.eval()
        
        return model
    
    def benchmark_model(self, model: YOLO, test_images: List[np.ndarray] = None, 
                       iterations: int = 10) -> Dict:
        """
        모델 성능 벤치마크
        
        Args:
            model: 벤치마크할 모델
            test_images: 테스트 이미지 (없으면 더미 이미지 생성)
            iterations: 반복 횟수
            
        Returns:
            성능 메트릭
        """
        if test_images is None:
            # 한국 번호판 크기 비율로 더미 이미지 생성
            test_images = []
            for _ in range(5):
                dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                test_images.append(dummy_img)
        
        config = self.get_optimal_config()
        
        # Warmup
        for img in test_images[:2]:
            model(img, verbose=False)
        
        # 실제 벤치마크
        times = []
        memory_usage = []
        
        for i in range(iterations):
            start_time = time.time()
            
            # 메모리 사용량 측정 시작
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
            
            # 추론 실행
            for img in test_images:
                results = model(img, verbose=False, 
                              conf=config.get('conf', 0.01),
                              iou=config.get('iou', 0.4))
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # 메모리 사용량 측정
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage.append((peak_memory - start_memory) / (1024**2))  # MB
        
        return {
            'avg_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'fps': len(test_images) / np.mean(times),
            'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
            'system_config': config,
            'system_info': self.system_info
        }
    
    def create_training_config(self, data_yaml_path: str, 
                             epochs: int = 100,
                             patience: int = 50) -> Dict:
        """
        한국 번호판 특화 학습 설정 생성
        
        Args:
            data_yaml_path: 학습 데이터 YAML 경로
            epochs: 학습 에포크
            patience: Early stopping patience
            
        Returns:
            학습 설정 딕셔너리
        """
        base_config = self.get_optimal_config()
        
        training_config = {
            'data': data_yaml_path,
            'epochs': epochs,
            'patience': patience,
            'batch': base_config.get('batch_size', 2),
            'imgsz': base_config.get('imgsz', 640),
            'save': True,
            'save_period': 10,
            'cache': False,  # 메모리 부족 방지
            'device': base_config.get('device', 0),
            'workers': base_config.get('workers', 4),
            'project': 'korean_plate_training',
            'name': 'korean_specialized_v1',
            
            # 한국 번호판 특화 증강 설정
            'hsv_h': 0.015,      # 색상 변화 (번호판 색상 고려)
            'hsv_s': 0.7,        # 채도 변화
            'hsv_v': 0.4,        # 명도 변화
            'degrees': 5.0,      # 회전 각도 (번호판은 보통 수평)
            'translate': 0.1,    # 이동
            'scale': 0.2,        # 크기 변화
            'shear': 2.0,        # 전단 변형
            'perspective': 0.0,  # 원근 변형 (번호판에는 적용하지 않음)
            'flipud': 0.0,       # 상하 뒤집기 (번호판에는 부적절)
            'fliplr': 0.0,       # 좌우 뒤집기 (번호판 글자가 뒤집힘)
            'mosaic': 0.5,       # 모자이크 증강
            'mixup': 0.1,        # MixUp 증강
            'copy_paste': 0.1,   # Copy-paste 증강
            
            # 최적화 설정
            'optimizer': 'AdamW',
            'lr0': 0.001,        # 초기 학습률
            'lrf': 0.1,          # 최종 학습률 (lr0 * lrf)
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Loss 가중치 (한국 번호판 특성 반영)
            'box': 0.05,         # Box loss 가중치
            'cls': 0.3,          # Classification loss 가중치
            'dfl': 1.5,          # DFL loss 가중치 (객체 경계 정확도)
            
            # 검증 설정
            'val': True,
            'plots': True,
            'save_json': True
        }
        
        return training_config
    
    def get_system_recommendation(self) -> str:
        """현재 시스템에 대한 최적화 권장사항 반환"""
        info = self.system_info
        config = self.get_optimal_config()
        
        recommendations = []
        recommendations.append(f"=== 시스템 분석 결과 ===")
        recommendations.append(f"CPU 코어: {info['cpu_count']}개")
        recommendations.append(f"메모리: {info['memory_gb']:.1f}GB")
        
        if info['cuda_available']:
            recommendations.append(f"GPU: 사용 가능 ({info['gpu_count']}개)")
            recommendations.append(f"GPU 메모리: {info['gpu_memory']:.1f}GB")
            recommendations.append(f"권장 모델: {config['model_size']}")
            recommendations.append(f"권장 배치 크기: {config['batch_size']}")
            recommendations.append(f"권장 이미지 크기: {config['imgsz']}")
            
            if info['gpu_memory'] < 4:
                recommendations.append("⚠️ GPU 메모리가 부족합니다. CPU 모드를 고려해보세요.")
            elif info['gpu_memory'] >= 8:
                recommendations.append("✅ 고성능 GPU 환경입니다. 최고 품질 설정을 사용할 수 있습니다.")
        else:
            recommendations.append("GPU: 사용 불가 (CPU 모드)")
            recommendations.append(f"권장 모델: {config['model_size']} (경량화)")
            recommendations.append("💡 성능 향상을 위해 CUDA 호환 GPU 사용을 권장합니다.")
        
        recommendations.append(f"\n예상 처리 성능:")
        if info['cuda_available'] and info['gpu_memory'] >= 4:
            recommendations.append(f"  - 실시간 처리: 가능 (15-30 FPS)")
        elif info['cpu_count'] >= 4:
            recommendations.append(f"  - 실시간 처리: 제한적 (3-8 FPS)")
        else:
            recommendations.append(f"  - 실시간 처리: 어려움 (1-3 FPS)")
        
        return "\n".join(recommendations)
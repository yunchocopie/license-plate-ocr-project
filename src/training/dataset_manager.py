import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import json
from datetime import datetime
import config

"""
한국 번호판 데이터셋 관리자

PRD 요구사항에 따른 한국 번호판 특화 학습 데이터 구성:
1. 9가지 번호판 타입별 데이터 분류
2. 색상별 데이터 증강 및 밸런싱
3. YOLO 형식 라벨링 및 검증
4. 학습/검증/테스트 데이터 분할
"""

class KoreanPlateDatasetManager:
    """한국 번호판 데이터셋 관리 클래스"""
    
    def __init__(self, dataset_root: str = None):
        """
        데이터셋 매니저 초기화
        
        Args:
            dataset_root: 데이터셋 루트 디렉토리 경로
        """
        self.dataset_root = Path(dataset_root) if dataset_root else Path(config.DATA_DIR) / "korean_plates"
        self.setup_directories()
        
        # 한국 번호판 타입별 클래스 정의
        self.plate_classes = {
            0: {
                'name': 'general',
                'korean_name': '일반자가용',
                'colors': ['white'],
                'text_colors': ['black'],
                'priority': 1.0  # 학습 가중치
            },
            1: {
                'name': 'commercial',
                'korean_name': '영업용',
                'colors': ['yellow'],
                'text_colors': ['black'],
                'priority': 1.2  # 상업용은 중요도 높음
            },
            2: {
                'name': 'electric',
                'korean_name': '전기차',
                'colors': ['light_blue'],
                'text_colors': ['black'],
                'priority': 1.3  # 전기차 인식 중요
            },
            3: {
                'name': 'diplomatic',
                'korean_name': '외교관용',
                'colors': ['dark_blue'],
                'text_colors': ['white'],
                'priority': 1.1
            },
            4: {
                'name': 'military',
                'korean_name': '군용',
                'colors': ['red', 'dark_blue', 'light_blue'],
                'text_colors': ['white'],
                'priority': 1.0
            },
            5: {
                'name': 'construction',
                'korean_name': '건설기계',
                'colors': ['orange'],
                'text_colors': ['white'],
                'priority': 1.0
            },
            6: {
                'name': 'motorcycle',
                'korean_name': '이륜차',
                'colors': ['white'],
                'text_colors': ['blue'],
                'priority': 0.8  # 크기가 작아 검출 어려움
            },
            7: {
                'name': 'temporary',
                'korean_name': '임시운행',
                'colors': ['white'],
                'text_colors': ['black'],
                'priority': 0.7  # 임시번호판은 덜 중요
            },
            8: {
                'name': 'special',
                'korean_name': '특수용도',
                'colors': ['green'],
                'text_colors': ['black'],
                'priority': 0.9
            }
        }
    
    def setup_directories(self):
        """데이터셋 디렉토리 구조 생성"""
        directories = [
            self.dataset_root,
            self.dataset_root / "raw",
            self.dataset_root / "processed",
            self.dataset_root / "train" / "images",
            self.dataset_root / "train" / "labels",
            self.dataset_root / "val" / "images", 
            self.dataset_root / "val" / "labels",
            self.dataset_root / "test" / "images",
            self.dataset_root / "test" / "labels",
            self.dataset_root / "annotations",
            self.dataset_root / "statistics"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def create_yolo_dataset_config(self, dataset_name: str = "korean_plates") -> str:
        """
        YOLO 학습용 데이터셋 설정 파일 생성
        
        Args:
            dataset_name: 데이터셋 이름
            
        Returns:
            생성된 YAML 파일 경로
        """
        config_data = {
            'path': str(self.dataset_root.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.plate_classes),
            'names': [info['name'] for info in self.plate_classes.values()],
            'korean_names': [info['korean_name'] for info in self.plate_classes.values()],
            
            # 한국 번호판 특화 설정
            'plate_info': {
                'aspect_ratio_range': [2.0, 6.0],  # 한국 번호판 가로:세로 비율
                'min_size': {'width': 30, 'height': 8},
                'typical_ratios': {
                    'general': 4.0,
                    'commercial': 4.2,
                    'electric': 4.0,
                    'diplomatic': 4.5,
                    'military': 4.0,
                    'construction': 4.8,
                    'motorcycle': 3.5,
                    'temporary': 4.0,
                    'special': 4.0
                }
            },
            
            # 클래스별 가중치
            'class_weights': [info['priority'] for info in self.plate_classes.values()]
        }
        
        yaml_path = self.dataset_root / f"{dataset_name}.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False)
        
        return str(yaml_path)
    
    def analyze_dataset(self) -> Dict:
        """데이터셋 분석 및 통계 생성"""
        stats = {
            'total_images': 0,
            'total_labels': 0,
            'class_distribution': {info['name']: 0 for info in self.plate_classes.values()},
            'split_distribution': {'train': 0, 'val': 0, 'test': 0},
            'aspect_ratios': [],
            'image_sizes': [],
            'missing_labels': [],
            'invalid_annotations': [],
            'created_at': datetime.now().isoformat()
        }
        
        # 각 분할별로 분석
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_root / split / "images"
            labels_dir = self.dataset_root / split / "labels"
            
            if not images_dir.exists():
                continue
                
            image_files = list(images_dir.glob("*"))
            stats['split_distribution'][split] = len(image_files)
            stats['total_images'] += len(image_files)
            
            for image_file in image_files:
                # 해당 라벨 파일 확인
                label_file = labels_dir / f"{image_file.stem}.txt"
                
                if not label_file.exists():
                    stats['missing_labels'].append(str(image_file))
                    continue
                
                # 라벨 파일 분석
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    stats['total_labels'] += len(lines)
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            stats['invalid_annotations'].append(str(label_file))
                            continue
                        
                        class_id = int(parts[0])
                        if class_id in self.plate_classes:
                            class_name = self.plate_classes[class_id]['name']
                            stats['class_distribution'][class_name] += 1
                        
                        # 바운딩 박스에서 종횡비 계산
                        _, x_center, y_center, width, height = map(float, parts)
                        aspect_ratio = width / height if height > 0 else 0
                        if 1.0 < aspect_ratio < 10.0:  # 유효한 범위만
                            stats['aspect_ratios'].append(aspect_ratio)
                
                except Exception as e:
                    stats['invalid_annotations'].append(f"{label_file}: {str(e)}")
        
        # 통계 계산
        if stats['aspect_ratios']:
            import numpy as np
            stats['aspect_ratio_stats'] = {
                'mean': float(np.mean(stats['aspect_ratios'])),
                'std': float(np.std(stats['aspect_ratios'])),
                'min': float(np.min(stats['aspect_ratios'])),
                'max': float(np.max(stats['aspect_ratios']))
            }
        
        # 통계 저장
        stats_file = self.dataset_root / "statistics" / "dataset_analysis.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return stats
    
    def generate_augmentation_config(self, target_samples_per_class: int = 1000) -> Dict:
        """
        클래스 불균형 해결을 위한 데이터 증강 설정 생성
        
        Args:
            target_samples_per_class: 클래스별 목표 샘플 수
            
        Returns:
            증강 설정 딕셔너리
        """
        stats = self.analyze_dataset()
        class_dist = stats['class_distribution']
        
        augmentation_config = {
            'target_samples': target_samples_per_class,
            'augmentation_strategies': {},
            'priority_based_sampling': True
        }
        
        for class_name, current_count in class_dist.items():
            if current_count == 0:
                continue
                
            # 부족한 샘플 수 계산
            needed_samples = max(0, target_samples_per_class - current_count)
            augmentation_factor = needed_samples / current_count if current_count > 0 else 0
            
            # 클래스별 특화 증강 전략
            class_info = next(info for info in self.plate_classes.values() 
                            if info['name'] == class_name)
            
            aug_strategy = {
                'current_count': current_count,
                'target_count': target_samples_per_class,
                'augmentation_factor': augmentation_factor,
                'priority': class_info['priority'],
                'transformations': self._get_class_specific_augmentations(class_name, class_info)
            }
            
            augmentation_config['augmentation_strategies'][class_name] = aug_strategy
        
        # 증강 설정 저장
        config_file = self.dataset_root / "statistics" / "augmentation_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(augmentation_config, f, ensure_ascii=False, indent=2)
        
        return augmentation_config
    
    def _get_class_specific_augmentations(self, class_name: str, class_info: Dict) -> Dict:
        """클래스별 특화 증강 변환 설정"""
        
        base_transforms = {
            'brightness': {'min': 0.7, 'max': 1.3},
            'contrast': {'min': 0.8, 'max': 1.2},
            'saturation': {'min': 0.8, 'max': 1.2},
            'hue': {'min': -0.01, 'max': 0.01},
            'noise': {'std': 0.01},
            'blur': {'kernel_size': (1, 3), 'probability': 0.1}
        }
        
        # 클래스별 특화 설정
        class_specific = {
            'general': {
                # 일반 번호판: 다양한 조명 조건
                'brightness': {'min': 0.6, 'max': 1.4},
                'shadow': {'probability': 0.3}
            },
            'commercial': {
                # 영업용: 노란색 보존
                'hue': {'min': -0.005, 'max': 0.005},
                'saturation': {'min': 0.9, 'max': 1.1}
            },
            'electric': {
                # 전기차: 하늘색 보존
                'hue': {'min': -0.005, 'max': 0.005},
                'contrast': {'min': 0.9, 'max': 1.1}
            },
            'motorcycle': {
                # 이륜차: 작은 크기, 해상도 개선
                'upscaling': {'factor': 1.2},
                'sharpening': {'probability': 0.5}
            },
            'temporary': {
                # 임시번호판: 대각선 패턴 보존
                'rotation': {'max_angle': 2},
                'perspective': {'probability': 0.1}
            }
        }
        
        # 기본 설정과 클래스별 설정 병합
        transforms = base_transforms.copy()
        if class_name in class_specific:
            transforms.update(class_specific[class_name])
        
        return transforms
    
    def validate_dataset(self) -> Dict:
        """데이터셋 유효성 검사"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # 1. 기본 디렉토리 구조 확인
        required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
        for dir_path in required_dirs:
            full_path = self.dataset_root / dir_path
            if not full_path.exists():
                validation_results['errors'].append(f"Required directory missing: {dir_path}")
                validation_results['is_valid'] = False
        
        # 2. 데이터 불균형 확인
        stats = self.analyze_dataset()
        class_dist = stats['class_distribution']
        
        if class_dist:
            max_samples = max(class_dist.values())
            min_samples = min(class_dist.values())
            
            if max_samples > 0 and min_samples / max_samples < 0.1:
                validation_results['warnings'].append(
                    f"Severe class imbalance detected. Max: {max_samples}, Min: {min_samples}"
                )
                validation_results['recommendations'].append(
                    "Consider using data augmentation or class weights"
                )
        
        # 3. 종횡비 검증
        if 'aspect_ratio_stats' in stats:
            ar_stats = stats['aspect_ratio_stats']
            if ar_stats['mean'] < 2.0 or ar_stats['mean'] > 6.0:
                validation_results['warnings'].append(
                    f"Unusual aspect ratio distribution. Mean: {ar_stats['mean']:.2f}"
                )
        
        # 4. 누락된 라벨 확인
        if stats['missing_labels']:
            validation_results['errors'].append(
                f"{len(stats['missing_labels'])} images missing labels"
            )
            validation_results['is_valid'] = False
        
        # 5. 최소 샘플 수 확인
        min_samples_required = 50
        for class_name, count in class_dist.items():
            if count < min_samples_required:
                validation_results['warnings'].append(
                    f"Class '{class_name}' has only {count} samples (minimum recommended: {min_samples_required})"
                )
        
        # 검증 결과 저장
        validation_file = self.dataset_root / "statistics" / "validation_report.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)
        
        return validation_results
    
    def get_training_recommendations(self) -> List[str]:
        """학습 권장사항 생성"""
        stats = self.analyze_dataset()
        validation = self.validate_dataset()
        
        recommendations = []
        
        # 1. 데이터셋 크기 기반 권장사항
        total_images = stats['total_images']
        if total_images < 500:
            recommendations.append("⚠️ 데이터셋이 작습니다 (500장 미만). 더 많은 데이터 수집을 권장합니다.")
            recommendations.append("💡 전이학습(pretrained model) 사용을 권장합니다.")
        elif total_images < 2000:
            recommendations.append("📊 중간 규모 데이터셋입니다. 적절한 증강 기법을 사용하세요.")
        else:
            recommendations.append("✅ 충분한 데이터셋 크기입니다.")
        
        # 2. 클래스 불균형 권장사항
        class_dist = stats['class_distribution']
        if class_dist:
            max_samples = max(class_dist.values())
            min_samples = min(class_dist.values())
            
            if max_samples > 0 and min_samples / max_samples < 0.3:
                recommendations.append("⚖️ 클래스 불균형이 있습니다. 다음을 고려하세요:")
                recommendations.append("   - 클래스 가중치 (class_weight) 적용")
                recommendations.append("   - 부족한 클래스 데이터 증강")
                recommendations.append("   - Focal Loss 사용")
        
        # 3. 하드웨어 기반 권장사항
        from .korean_plate_optimizer import KoreanPlateOptimizer
        optimizer = KoreanPlateOptimizer()
        system_info = optimizer.system_info
        
        if not system_info['cuda_available']:
            recommendations.append("🖥️ GPU가 없습니다. CPU 학습은 매우 느릴 수 있습니다.")
            recommendations.append("   - 모델 크기를 nano/small로 제한")
            recommendations.append("   - 배치 크기를 1로 설정")
            recommendations.append("   - 이미지 크기를 416x416으로 축소")
        elif system_info['gpu_memory'] < 4:
            recommendations.append("💾 GPU 메모리가 부족합니다 (4GB 미만).")
            recommendations.append("   - 배치 크기를 줄이세요 (1-2)")
            recommendations.append("   - Mixed precision 학습 사용")
        
        # 4. 에포크 및 학습률 권장사항
        if total_images < 1000:
            recommendations.append("📈 학습 파라미터 권장사항:")
            recommendations.append("   - Epochs: 150-200 (작은 데이터셋)")
            recommendations.append("   - Learning rate: 0.001-0.003")
            recommendations.append("   - Patience: 30-50")
        else:
            recommendations.append("📈 학습 파라미터 권장사항:")
            recommendations.append("   - Epochs: 100-150")
            recommendations.append("   - Learning rate: 0.001")
            recommendations.append("   - Patience: 50")
        
        return recommendations
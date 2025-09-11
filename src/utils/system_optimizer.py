import os
import psutil
import torch
import threading
import time
import gc
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import config

"""
로컬 환경 통합 최적화 시스템

PRD 요구사항에 따른 로컬 환경 최적화:
1. CPU/GPU/메모리 실시간 모니터링
2. 동적 성능 조정 및 리소스 관리
3. 프로세스 우선순위 자동 조정
4. 메모리 누수 방지 및 가비지 컬렉션
5. 온도 및 전력 모니터링 (가능한 경우)
"""

@dataclass
class SystemResources:
    """시스템 리소스 상태"""
    cpu_usage: float
    cpu_count: int
    memory_usage: float
    memory_total: float
    memory_available: float
    gpu_available: bool
    gpu_count: int
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_temperature: Optional[float] = None
    disk_usage: float = 0.0
    disk_available: float = 0.0

@dataclass
class PerformanceProfile:
    """성능 프로파일"""
    name: str
    cpu_affinity: Optional[List[int]]
    process_priority: str  # 'high', 'normal', 'low'
    max_memory_mb: int
    gc_frequency: int  # seconds
    optimization_level: str  # 'maximum', 'balanced', 'conservative'

class SystemOptimizer:
    """로컬 환경 통합 최적화 관리자"""
    
    def __init__(self, monitoring_interval: float = 5.0):
        """
        시스템 최적화기 초기화
        
        Args:
            monitoring_interval: 모니터링 간격 (초)
        """
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 성능 히스토리
        self.performance_history: List[Dict] = []
        self.max_history_size = 100
        
        # 최적화 콜백
        self.optimization_callbacks: List[Callable] = []
        
        # 기본 프로파일 생성
        self.profiles = self._create_default_profiles()
        self.current_profile = self._select_optimal_profile()
        
        # 리소스 임계값
        self.thresholds = {
            'cpu_high': 85.0,
            'cpu_critical': 95.0,
            'memory_high': 80.0,
            'memory_critical': 90.0,
            'gpu_memory_high': 85.0,
            'gpu_memory_critical': 95.0,
            'disk_low': 15.0,
            'disk_critical': 5.0
        }
        
        # 현재 상태
        self.current_resources = self._get_system_resources()
        self._apply_initial_optimizations()
        
    def _create_default_profiles(self) -> Dict[str, PerformanceProfile]:
        """기본 성능 프로파일 생성"""
        cpu_count = psutil.cpu_count()
        total_memory = psutil.virtual_memory().total // (1024 ** 2)  # MB
        
        profiles = {}
        
        # 최고 성능 프로파일 (고성능 시스템용)
        profiles['maximum'] = PerformanceProfile(
            name='Maximum Performance',
            cpu_affinity=list(range(min(cpu_count, 8))),  # 최대 8코어 사용
            process_priority='high',
            max_memory_mb=min(total_memory // 2, 8192),  # 최대 8GB
            gc_frequency=30,
            optimization_level='maximum'
        )
        
        # 균형 프로파일 (일반 사용)
        profiles['balanced'] = PerformanceProfile(
            name='Balanced',
            cpu_affinity=list(range(min(cpu_count, 4))),  # 최대 4코어
            process_priority='normal',
            max_memory_mb=min(total_memory // 3, 4096),  # 최대 4GB
            gc_frequency=60,
            optimization_level='balanced'
        )
        
        # 절약 프로파일 (저사양 시스템용)
        profiles['conservative'] = PerformanceProfile(
            name='Conservative',
            cpu_affinity=[0] if cpu_count == 1 else [0, 1],  # 최대 2코어
            process_priority='normal',
            max_memory_mb=min(total_memory // 4, 2048),  # 최대 2GB
            gc_frequency=30,  # 더 자주 정리
            optimization_level='conservative'
        )
        
        return profiles
    
    def _select_optimal_profile(self) -> str:
        """현재 시스템에 최적인 프로파일 자동 선택"""
        resources = self._get_system_resources()
        
        # GPU가 있고 메모리가 충분한 고성능 시스템
        if (resources.gpu_available and 
            resources.gpu_memory_total > 6000 and  # 6GB+ GPU
            resources.memory_total > 16000 and     # 16GB+ RAM
            resources.cpu_count >= 6):             # 6+ CPU cores
            return 'maximum'
        
        # 일반적인 시스템
        elif (resources.memory_total > 8000 and   # 8GB+ RAM
              resources.cpu_count >= 4):          # 4+ CPU cores
            return 'balanced'
        
        # 저사양 시스템
        else:
            return 'conservative'
    
    def _get_system_resources(self) -> SystemResources:
        """현재 시스템 리소스 상태 조회"""
        # CPU 정보
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_total = memory.total // (1024 ** 2)  # MB
        memory_available = memory.available // (1024 ** 2)  # MB
        
        # GPU 정보
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        gpu_temperature = None
        
        if gpu_available and gpu_count > 0:
            try:
                gpu_memory_used = torch.cuda.memory_allocated() // (1024 ** 2)  # MB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)  # MB
            except:
                pass
        
        # 디스크 정보
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        disk_available = disk.free // (1024 ** 3)  # GB
        
        return SystemResources(
            cpu_usage=cpu_usage,
            cpu_count=cpu_count,
            memory_usage=memory_usage,
            memory_total=memory_total,
            memory_available=memory_available,
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_temperature=gpu_temperature,
            disk_usage=disk_usage,
            disk_available=disk_available
        )
    
    def _apply_initial_optimizations(self):
        """초기 시스템 최적화 적용"""
        try:
            current_process = psutil.Process()
            profile = self.profiles[self.current_profile]
            
            # CPU 어피니티 설정
            if profile.cpu_affinity:
                try:
                    current_process.cpu_affinity(profile.cpu_affinity)
                    print(f"CPU 어피니티 설정: {profile.cpu_affinity}")
                except:
                    pass  # 권한이 없거나 지원하지 않는 시스템
            
            # 프로세스 우선순위 설정
            try:
                if profile.process_priority == 'high':
                    current_process.nice(-5)  # 높은 우선순위
                elif profile.process_priority == 'low':
                    current_process.nice(5)   # 낮은 우선순위
                print(f"프로세스 우선순위 설정: {profile.process_priority}")
            except:
                pass  # 권한이 없는 경우
                
            # 초기 가비지 컬렉션
            self._optimize_memory()
            
            print(f"초기 최적화 완료 - 프로파일: {profile.name}")
            
        except Exception as e:
            print(f"초기 최적화 중 오류: {e}")
    
    def start_monitoring(self):
        """리소스 모니터링 시작"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        print("시스템 리소스 모니터링 시작")
    
    def stop_monitoring(self):
        """리소스 모니터링 중지"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        print("시스템 리소스 모니터링 중지")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 리소스 상태 업데이트
                self.current_resources = self._get_system_resources()
                
                # 히스토리 저장
                self._record_performance()
                
                # 자동 최적화 실행
                self._auto_optimize()
                
                # 콜백 실행
                self._execute_callbacks()
                
            except Exception as e:
                print(f"모니터링 루프 오류: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def _record_performance(self):
        """성능 데이터 기록"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': self.current_resources.cpu_usage,
            'memory_usage': self.current_resources.memory_usage,
            'memory_available': self.current_resources.memory_available,
            'gpu_memory_used': self.current_resources.gpu_memory_used,
            'profile': self.current_profile
        }
        
        self.performance_history.append(record)
        
        # 히스토리 크기 제한
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]
    
    def _auto_optimize(self):
        """자동 최적화 실행"""
        resources = self.current_resources
        
        # 메모리 사용량이 높을 때
        if resources.memory_usage > self.thresholds['memory_high']:
            self._optimize_memory()
        
        # GPU 메모리 사용량이 높을 때
        if (resources.gpu_available and 
            resources.gpu_memory_total > 0 and
            (resources.gpu_memory_used / resources.gpu_memory_total * 100) > self.thresholds['gpu_memory_high']):
            self._optimize_gpu_memory()
        
        # CPU 사용량이 높을 때 프로파일 조정
        if resources.cpu_usage > self.thresholds['cpu_high']:
            self._adjust_profile_for_high_load()
    
    def _optimize_memory(self):
        """메모리 최적화"""
        try:
            # Python 가비지 컬렉션
            collected = gc.collect()
            
            # PyTorch 캐시 정리 (GPU가 있는 경우)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"메모리 최적화 실행: {collected}개 객체 정리")
            
        except Exception as e:
            print(f"메모리 최적화 오류: {e}")
    
    def _optimize_gpu_memory(self):
        """GPU 메모리 최적화"""
        if not torch.cuda.is_available():
            return
        
        try:
            # GPU 메모리 캐시 정리
            torch.cuda.empty_cache()
            
            # 미사용 메모리 해제 시도
            if hasattr(torch.cuda, 'memory_stats'):
                stats = torch.cuda.memory_stats()
                print(f"GPU 메모리 최적화: {stats.get('reserved_bytes.all.freed', 0) // (1024**2)}MB 해제")
                
        except Exception as e:
            print(f"GPU 메모리 최적화 오류: {e}")
    
    def _adjust_profile_for_high_load(self):
        """높은 부하 상황에서 프로파일 조정"""
        if self.current_profile == 'maximum':
            self.switch_profile('balanced')
            print("높은 CPU 부하로 인해 균형 모드로 전환")
        elif self.current_profile == 'balanced':
            self.switch_profile('conservative')
            print("높은 CPU 부하로 인해 절약 모드로 전환")
    
    def _execute_callbacks(self):
        """등록된 최적화 콜백 실행"""
        for callback in self.optimization_callbacks:
            try:
                callback(self.current_resources, self.current_profile)
            except Exception as e:
                print(f"최적화 콜백 오류: {e}")
    
    def switch_profile(self, profile_name: str) -> bool:
        """성능 프로파일 전환"""
        if profile_name not in self.profiles:
            print(f"존재하지 않는 프로파일: {profile_name}")
            return False
        
        if profile_name == self.current_profile:
            return True
        
        old_profile = self.current_profile
        self.current_profile = profile_name
        
        # 새 프로파일 적용
        self._apply_initial_optimizations()
        
        print(f"성능 프로파일 전환: {old_profile} → {profile_name}")
        return True
    
    def add_optimization_callback(self, callback: Callable):
        """최적화 콜백 추가"""
        self.optimization_callbacks.append(callback)
    
    def get_optimization_recommendations(self) -> List[str]:
        """현재 상태 기반 최적화 권장사항"""
        recommendations = []
        resources = self.current_resources
        
        # CPU 관련 권장사항
        if resources.cpu_usage > self.thresholds['cpu_critical']:
            recommendations.append("⚠️ CPU 사용률이 매우 높습니다. 작업을 일시 중지하거나 다른 프로그램을 종료하세요.")
        elif resources.cpu_usage > self.thresholds['cpu_high']:
            recommendations.append("💡 CPU 사용률이 높습니다. 배치 크기를 줄이거나 처리 속도를 조절하세요.")
        
        # 메모리 관련 권장사항
        if resources.memory_usage > self.thresholds['memory_critical']:
            recommendations.append("🚨 메모리 부족 위험! 다른 프로그램을 종료하고 메모리를 확보하세요.")
        elif resources.memory_usage > self.thresholds['memory_high']:
            recommendations.append("💾 메모리 사용량이 높습니다. 불필요한 데이터를 정리하세요.")
        
        # GPU 관련 권장사항
        if resources.gpu_available:
            gpu_usage_percent = (resources.gpu_memory_used / resources.gpu_memory_total * 100) if resources.gpu_memory_total > 0 else 0
            
            if gpu_usage_percent > self.thresholds['gpu_memory_critical']:
                recommendations.append("🎮 GPU 메모리 부족! 모델 크기를 줄이거나 배치 크기를 줄이세요.")
            elif gpu_usage_percent > self.thresholds['gpu_memory_high']:
                recommendations.append("🔧 GPU 메모리 사용량이 높습니다. 불필요한 GPU 데이터를 정리하세요.")
        else:
            if resources.cpu_count < 4:
                recommendations.append("💻 CPU 성능 향상을 위해 GPU 사용을 고려하세요.")
        
        # 디스크 관련 권장사항
        if resources.disk_available < self.thresholds['disk_critical']:
            recommendations.append("💿 디스크 공간이 매우 부족합니다! 불필요한 파일을 삭제하세요.")
        elif resources.disk_available < self.thresholds['disk_low']:
            recommendations.append("📁 디스크 공간이 부족합니다. 정리를 권장합니다.")
        
        # 프로파일 관련 권장사항
        if self.current_profile == 'conservative' and resources.memory_usage < 50 and resources.cpu_usage < 50:
            recommendations.append("🚀 시스템 리소스에 여유가 있습니다. 더 높은 성능 모드를 사용해보세요.")
        
        return recommendations
    
    def get_system_report(self) -> Dict:
        """시스템 상태 종합 보고서"""
        resources = self.current_resources
        profile = self.profiles[self.current_profile]
        
        # 최근 성능 평균 계산
        recent_history = self.performance_history[-10:] if self.performance_history else []
        avg_cpu = sum(h['cpu_usage'] for h in recent_history) / len(recent_history) if recent_history else 0
        avg_memory = sum(h['memory_usage'] for h in recent_history) / len(recent_history) if recent_history else 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_profile': {
                'name': profile.name,
                'level': self.current_profile,
                'cpu_cores_used': len(profile.cpu_affinity) if profile.cpu_affinity else resources.cpu_count,
                'max_memory_mb': profile.max_memory_mb,
                'priority': profile.process_priority
            },
            'system_resources': {
                'cpu': {
                    'current_usage': resources.cpu_usage,
                    'average_usage': avg_cpu,
                    'total_cores': resources.cpu_count,
                    'assigned_cores': profile.cpu_affinity
                },
                'memory': {
                    'current_usage_percent': resources.memory_usage,
                    'average_usage_percent': avg_memory,
                    'total_mb': resources.memory_total,
                    'available_mb': resources.memory_available,
                    'limit_mb': profile.max_memory_mb
                },
                'gpu': {
                    'available': resources.gpu_available,
                    'count': resources.gpu_count,
                    'memory_used_mb': resources.gpu_memory_used,
                    'memory_total_mb': resources.gpu_memory_total,
                    'usage_percent': (resources.gpu_memory_used / resources.gpu_memory_total * 100) if resources.gpu_memory_total > 0 else 0
                },
                'disk': {
                    'usage_percent': resources.disk_usage,
                    'available_gb': resources.disk_available
                }
            },
            'recommendations': self.get_optimization_recommendations(),
            'monitoring_active': self.monitoring_active,
            'history_count': len(self.performance_history)
        }
        
        return report
    
    def save_performance_log(self, filepath: Optional[str] = None):
        """성능 로그 저장"""
        if not filepath:
            log_dir = Path(config.DATA_DIR) / "logs"
            log_dir.mkdir(exist_ok=True, parents=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = log_dir / f"system_performance_{timestamp}.json"
        
        log_data = {
            'system_report': self.get_system_report(),
            'performance_history': self.performance_history,
            'profiles_used': list(set(h.get('profile', 'unknown') for h in self.performance_history))
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"성능 로그 저장: {filepath}")
        return filepath
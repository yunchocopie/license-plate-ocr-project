import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from .system_optimizer import SystemOptimizer, SystemResources
import config

"""
자동 튜닝 시스템

로컬 환경 성능을 실시간 모니터링하고 자동으로 최적화 설정을 조정합니다.
- 성능 벤치마킹을 통한 최적 설정 찾기  
- 부하 상황에 따른 동적 설정 조정
- 배터리/전력 상황 감지 및 절전 모드 자동 전환
- 열 관리 및 쓰로틀링 방지
"""

@dataclass
class TuningResult:
    """튜닝 결과"""
    config_name: str
    performance_score: float
    resource_usage: float
    stability_score: float
    recommendation: str

class AutoTuner:
    """자동 튜닝 관리자"""
    
    def __init__(self, system_optimizer: SystemOptimizer):
        self.system_optimizer = system_optimizer
        self.tuning_active = False
        self.tuning_thread = None
        
        # 튜닝 히스토리
        self.tuning_history: List[TuningResult] = []
        self.best_config = None
        self.baseline_performance = None
        
        # 튜닝 설정
        self.tuning_interval = 300  # 5분마다 튜닝 평가
        self.min_samples = 5  # 최소 샘플 수
        self.performance_threshold = 0.1  # 성능 향상 임계값
        
        # 동적 조정 임계값
        self.thresholds = {
            'cpu_overload': 90.0,
            'memory_pressure': 85.0,
            'gpu_overload': 90.0,
            'thermal_throttle': 85.0,  # CPU 온도 (추정)
            'stability_min': 0.8  # 최소 안정성 점수
        }
        
        # 테스트 설정들
        self.test_configs = self._generate_test_configs()
        
    def _generate_test_configs(self) -> List[Dict]:
        """테스트용 설정 조합 생성"""
        configs = []
        
        # 기본 프로파일들
        base_profiles = ['conservative', 'balanced', 'maximum']
        
        # 전처리 모드들
        preprocessing_modes = ['fast', 'balanced', 'high_quality']
        
        # 배치 크기들 (시스템 메모리에 따라)
        memory_gb = self.system_optimizer.current_resources.memory_total / 1024
        
        if memory_gb >= 16:
            batch_sizes = [1, 2, 4]
        elif memory_gb >= 8:
            batch_sizes = [1, 2]
        else:
            batch_sizes = [1]
        
        # 조합 생성
        for profile in base_profiles:
            for preprocess in preprocessing_modes:
                for batch in batch_sizes:
                    configs.append({
                        'profile': profile,
                        'preprocessing_mode': preprocess,
                        'batch_size': batch,
                        'name': f"{profile}_{preprocess}_b{batch}"
                    })
        
        return configs
    
    def start_auto_tuning(self):
        """자동 튜닝 시작"""
        if self.tuning_active:
            return
        
        self.tuning_active = True
        self.tuning_thread = threading.Thread(
            target=self._tuning_loop,
            daemon=True
        )
        self.tuning_thread.start()
        print("자동 튜닝 시작")
    
    def stop_auto_tuning(self):
        """자동 튜닝 중지"""
        self.tuning_active = False
        if self.tuning_thread and self.tuning_thread.is_alive():
            self.tuning_thread.join(timeout=5)
        print("자동 튜닝 중지")
    
    def _tuning_loop(self):
        """튜닝 루프"""
        while self.tuning_active:
            try:
                # 현재 성능 측정
                current_perf = self._measure_current_performance()
                
                # 기준선 성능 설정
                if self.baseline_performance is None:
                    self.baseline_performance = current_perf
                    print(f"기준선 성능 설정: {current_perf:.3f}")
                
                # 시스템 상태 확인 및 동적 조정
                self._check_and_adjust()
                
                # 주기적 최적화 테스트 (매 30분마다)
                if len(self.tuning_history) % 6 == 0:  # 5분 * 6 = 30분
                    self._run_optimization_test()
                
            except Exception as e:
                print(f"자동 튜닝 루프 오류: {e}")
            
            time.sleep(self.tuning_interval)
    
    def _measure_current_performance(self) -> float:
        """현재 성능 측정"""
        try:
            # 간단한 성능 메트릭 계산
            resources = self.system_optimizer.current_resources
            
            # CPU 효율성 (낮은 사용률이 더 좋음)
            cpu_efficiency = max(0, 100 - resources.cpu_usage) / 100
            
            # 메모리 효율성
            memory_efficiency = max(0, 100 - resources.memory_usage) / 100
            
            # GPU 효율성 (사용 가능한 경우)
            gpu_efficiency = 1.0
            if resources.gpu_available and resources.gpu_memory_total > 0:
                gpu_usage = resources.gpu_memory_used / resources.gpu_memory_total * 100
                gpu_efficiency = max(0, 100 - gpu_usage) / 100
            
            # 전체 성능 점수 (0-1)
            performance_score = (cpu_efficiency * 0.4 + 
                               memory_efficiency * 0.4 + 
                               gpu_efficiency * 0.2)
            
            return performance_score
            
        except Exception as e:
            print(f"성능 측정 오류: {e}")
            return 0.5  # 기본값
    
    def _check_and_adjust(self):
        """시스템 상태 확인 및 자동 조정"""
        resources = self.system_optimizer.current_resources
        current_profile = self.system_optimizer.current_profile
        
        # CPU 과부하 감지
        if resources.cpu_usage > self.thresholds['cpu_overload']:
            if current_profile == 'maximum':
                self.system_optimizer.switch_profile('balanced')
                self._log_adjustment("CPU 과부하로 균형 모드로 전환")
            elif current_profile == 'balanced':
                self.system_optimizer.switch_profile('conservative')
                self._log_adjustment("CPU 과부하로 절약 모드로 전환")
        
        # 메모리 압박 감지
        if resources.memory_usage > self.thresholds['memory_pressure']:
            # 강제 메모리 정리
            self.system_optimizer._optimize_memory()
            
            if current_profile != 'conservative':
                self.system_optimizer.switch_profile('conservative')
                self._log_adjustment("메모리 압박으로 절약 모드로 전환")
        
        # GPU 메모리 과부하 감지
        if (resources.gpu_available and 
            resources.gpu_memory_total > 0):
            gpu_usage = resources.gpu_memory_used / resources.gpu_memory_total * 100
            
            if gpu_usage > self.thresholds['gpu_overload']:
                self.system_optimizer._optimize_gpu_memory()
                self._log_adjustment("GPU 메모리 정리 실행")
        
        # 시스템 안정성 확인 (리소스 여유가 있으면 상위 모드로)
        if (resources.cpu_usage < 50 and 
            resources.memory_usage < 60 and
            current_profile == 'conservative'):
            self.system_optimizer.switch_profile('balanced')
            self._log_adjustment("리소스 여유로 균형 모드로 전환")
        
        elif (resources.cpu_usage < 30 and 
              resources.memory_usage < 40 and
              current_profile == 'balanced' and
              resources.gpu_available):
            self.system_optimizer.switch_profile('maximum')
            self._log_adjustment("충분한 리소스로 최고 성능 모드로 전환")
    
    def _log_adjustment(self, message: str):
        """조정 로그 기록"""
        print(f"[자동 튜닝] {message}")
    
    def _run_optimization_test(self):
        """최적화 테스트 실행"""
        if len(self.test_configs) == 0:
            return
        
        print("최적화 테스트 시작...")
        
        # 현재 상태 저장
        original_profile = self.system_optimizer.current_profile
        
        best_result = None
        test_results = []
        
        try:
            # 각 설정 테스트
            for i, config in enumerate(self.test_configs[:3]):  # 최대 3개 설정만 테스트
                print(f"테스트 중... ({i+1}/{min(3, len(self.test_configs))}) - {config['name']}")
                
                # 설정 적용
                self.system_optimizer.switch_profile(config['profile'])
                time.sleep(2)  # 안정화 대기
                
                # 성능 측정
                performance_scores = []
                resource_usages = []
                
                for _ in range(3):  # 3번 측정
                    perf = self._measure_current_performance()
                    performance_scores.append(perf)
                    
                    resources = self.system_optimizer.current_resources
                    resource_usage = (resources.cpu_usage + resources.memory_usage) / 200
                    resource_usages.append(resource_usage)
                    
                    time.sleep(1)
                
                # 결과 계산
                avg_performance = np.mean(performance_scores)
                avg_resource_usage = np.mean(resource_usages)
                stability = 1.0 - np.std(performance_scores)  # 안정성 (변동이 적을수록 좋음)
                
                # 종합 점수 계산
                total_score = (avg_performance * 0.5 + 
                              (1 - avg_resource_usage) * 0.3 + 
                              stability * 0.2)
                
                result = TuningResult(
                    config_name=config['name'],
                    performance_score=avg_performance,
                    resource_usage=avg_resource_usage,
                    stability_score=stability,
                    recommendation=self._generate_recommendation(config, total_score)
                )
                
                test_results.append(result)
                
                if best_result is None or total_score > (best_result.performance_score * 0.5 + 
                                                       (1 - best_result.resource_usage) * 0.3 + 
                                                       best_result.stability_score * 0.2):
                    best_result = result
                    self.best_config = config
        
        except Exception as e:
            print(f"최적화 테스트 오류: {e}")
        
        finally:
            # 원래 설정 복원 (더 좋은 설정이 없으면)
            if best_result and self.best_config:
                improvement = best_result.performance_score - self.baseline_performance
                if improvement > self.performance_threshold:
                    self.system_optimizer.switch_profile(self.best_config['profile'])
                    self.baseline_performance = best_result.performance_score
                    print(f"최적 설정 적용: {self.best_config['name']} (성능 향상: {improvement:.3f})")
                else:
                    self.system_optimizer.switch_profile(original_profile)
                    print("기존 설정 유지 (성능 향상 미미)")
            else:
                self.system_optimizer.switch_profile(original_profile)
                print("기존 설정 복원")
        
        # 결과 저장
        self.tuning_history.extend(test_results)
        
        # 히스토리 크기 제한
        if len(self.tuning_history) > 50:
            self.tuning_history = self.tuning_history[-50:]
    
    def _generate_recommendation(self, config: Dict, score: float) -> str:
        """권장사항 생성"""
        if score > 0.8:
            return f"우수한 성능 - {config['profile']} 모드 권장"
        elif score > 0.6:
            return f"양호한 성능 - 현재 설정 유지 권장"
        else:
            return f"성능 부족 - 더 보수적인 설정 고려"
    
    def get_tuning_summary(self) -> Dict:
        """튜닝 요약 정보"""
        if not self.tuning_history:
            return {"message": "튜닝 히스토리가 없습니다."}
        
        recent_results = self.tuning_history[-10:]  # 최근 10개 결과
        
        avg_performance = np.mean([r.performance_score for r in recent_results])
        avg_resource_usage = np.mean([r.resource_usage for r in recent_results])
        avg_stability = np.mean([r.stability_score for r in recent_results])
        
        best_config = max(self.tuning_history, key=lambda x: x.performance_score)
        
        return {
            "최근_평균_성능": f"{avg_performance:.3f}",
            "최근_평균_리소스_사용률": f"{avg_resource_usage:.3f}",
            "최근_평균_안정성": f"{avg_stability:.3f}",
            "최적_설정": best_config.config_name,
            "최적_성능": f"{best_config.performance_score:.3f}",
            "기준선_성능": f"{self.baseline_performance:.3f}" if self.baseline_performance else "미설정",
            "총_테스트_횟수": len(self.tuning_history),
            "자동_튜닝_활성": self.tuning_active
        }
    
    def force_optimization(self) -> Dict:
        """즉시 최적화 실행"""
        if self.tuning_active:
            return {"error": "자동 튜닝이 이미 실행 중입니다."}
        
        print("즉시 최적화 실행...")
        self._run_optimization_test()
        
        if self.best_config:
            return {
                "status": "완료",
                "best_config": self.best_config['name'],
                "performance_improvement": f"{self.baseline_performance:.3f}",
                "recommendation": "최적 설정이 적용되었습니다."
            }
        else:
            return {
                "status": "실패",
                "message": "최적화할 수 있는 설정을 찾지 못했습니다."
            }
    
    def get_smart_recommendations(self) -> List[str]:
        """지능형 권장사항 생성"""
        recommendations = []
        resources = self.system_optimizer.current_resources
        
        # 하드웨어 기반 권장사항
        if not resources.gpu_available:
            recommendations.append("💡 GPU 사용을 고려해보세요. CPU만으로는 성능에 한계가 있습니다.")
        
        if resources.memory_total < 8000:  # 8GB 미만
            recommendations.append("🔧 메모리가 부족합니다. 배치 처리나 절약 모드 사용을 권장합니다.")
        
        if resources.cpu_count < 4:
            recommendations.append("⚡ CPU 코어 수가 적습니다. 멀티스레딩 최적화가 제한적일 수 있습니다.")
        
        # 사용 패턴 기반 권장사항
        if self.tuning_history:
            recent_performance = [r.performance_score for r in self.tuning_history[-5:]]
            if recent_performance and np.std(recent_performance) > 0.1:
                recommendations.append("📊 성능 변동이 큽니다. 시스템 부하를 확인해보세요.")
            
            if len(self.tuning_history) > 10:
                trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
                if trend < -0.01:
                    recommendations.append("📉 성능이 하락 추세입니다. 시스템 점검을 권장합니다.")
                elif trend > 0.01:
                    recommendations.append("📈 성능이 개선되고 있습니다. 현재 설정을 유지하세요.")
        
        # 시간대 기반 권장사항
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour <= 18:  # 업무시간
            recommendations.append("🕒 업무시간입니다. 안정성을 위해 균형 모드를 권장합니다.")
        elif 22 <= current_hour or current_hour <= 6:  # 야간
            recommendations.append("🌙 야간시간입니다. 절전을 위해 절약 모드를 고려해보세요.")
        
        return recommendations
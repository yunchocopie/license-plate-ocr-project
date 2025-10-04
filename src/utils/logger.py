import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
import config

"""
통합 로깅 시스템

이 모듈은 프로젝트 전체에서 사용할 수 있는 로거를 제공합니다.
- 파일 및 콘솔 출력
- 로그 레벨별 필터링
- 로그 파일 자동 로테이션
"""


def setup_logger(name: str, level: int = logging.INFO, log_to_file: bool = True) -> logging.Logger:
    """
    로거를 설정하고 반환합니다.

    Args:
        name (str): 로거 이름 (일반적으로 __name__ 사용)
        level (int): 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file (bool): 파일에 로그를 기록할지 여부

    Returns:
        logging.Logger: 설정된 로거 인스턴스

    Example:
        >>> from src.utils.logger import setup_logger
        >>> logger = setup_logger(__name__)
        >>> logger.info("처리 시작")
        >>> logger.error("오류 발생", exc_info=True)
    """
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 이미 핸들러가 있으면 중복 추가 방지
    if logger.handlers:
        return logger

    # 포매터 설정
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 콘솔 핸들러 (WARNING 이상만 출력)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (모든 레벨 기록)
    if log_to_file:
        # 로그 디렉토리 생성
        log_dir = Path(config.DATA_DIR) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # 로그 파일 경로
        log_file = log_dir / "license_ocr.log"

        # 로테이팅 파일 핸들러 (최대 10MB, 백업 5개)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    기존 로거를 가져오거나 새로 생성합니다.

    Args:
        name (str): 로거 이름

    Returns:
        logging.Logger: 로거 인스턴스
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# 전역 로거 (하위 호환성용)
default_logger = setup_logger('license_ocr')


# 편의 함수들
def info(msg: str, *args, **kwargs):
    """INFO 레벨 로그 출력"""
    default_logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """WARNING 레벨 로그 출력"""
    default_logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """ERROR 레벨 로그 출력"""
    default_logger.error(msg, *args, **kwargs)


def debug(msg: str, *args, **kwargs):
    """DEBUG 레벨 로그 출력"""
    default_logger.debug(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """CRITICAL 레벨 로그 출력"""
    default_logger.critical(msg, *args, **kwargs)

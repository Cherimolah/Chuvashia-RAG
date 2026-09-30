
import logging
import os
import sys
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from config import LOG_LEVEL

LOG_DIR = Path(__file__).parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)
ROOT_NAME = 'chuvashia'


class ColorFormatter(logging.Formatter):
    """Раскрашивает уровень лога ANSI-кодами для читаемости в терминале."""
    COLORS = {
        'DEBUG':    '\033[36m',    # cyan
        'INFO':     '\033[32m',    # green
        'WARNING':  '\033[33m',    # yellow
        'ERROR':    '\033[31m',    # red
        'CRITICAL': '\033[1;41m',  # bold + red bg
    }
    RESET = '\033[0m'
    DIM = '\033[2m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        levelname = f'{color}{record.levelname:<8}{self.RESET}'
        name = f'{self.DIM}{record.name}:{record.lineno}{self.RESET}'
        ts = self.formatTime(record, self.datefmt)
        msg = record.getMessage()
        if record.exc_info:
            msg += '\n' + self.formatException(record.exc_info)
        return f'{ts} | {levelname} | {name} | {msg}'


def _setup_root() -> logging.Logger:
    """Настраивает корневой логгер проекта (один раз за процесс)."""
    logger = logging.getLogger(ROOT_NAME)
    if getattr(logger, '_chuvashia_configured', False):
        return logger

    logger.setLevel(logging.DEBUG)  # ловим всё, фильтруем хендлерами
    logger.propagate = False

    console_level = getattr(logging, LOG_LEVEL, logging.INFO)

    # --- Консоль ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(ColorFormatter(datefmt='%H:%M:%S'))
    logger.addHandler(console)

    # --- Полный лог в файл (всегда DEBUG) ---
    file_fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    all_file = RotatingFileHandler(
        LOG_DIR / 'app.log',
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8',
    )
    all_file.setLevel(logging.DEBUG)
    all_file.setFormatter(file_fmt)
    logger.addHandler(all_file)

    # --- Только ошибки ---
    err_file = RotatingFileHandler(
        LOG_DIR / 'errors.log',
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8',
    )
    err_file.setLevel(logging.ERROR)
    err_file.setFormatter(file_fmt)
    logger.addHandler(err_file)

    # Гасим излишне болтливые библиотеки
    for noisy in ('httpx', 'httpcore', 'urllib3', 'aiogram.event', 'asyncio'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger._chuvashia_configured = True
    logger.info(f'Логирование инициализировано. Уровень консоли: {LOG_LEVEL}. '
                f'Файлы: {LOG_DIR.resolve()}')
    return logger


_setup_root()


def get_logger(name: str) -> logging.Logger:
    """Логгер для модуля. Используй: ``log = get_logger(__name__)``."""
    short = name.split('.')[-1] if name else 'root'
    return logging.getLogger(f'{ROOT_NAME}.{short}')


def preview(value, max_len: int = 120) -> str:
    """
    Безопасно обрезает длинные строки/коллекции для логов.
    Заменяет переводы строк на видимый символ, чтобы лог не разъезжался.
    """
    if value is None:
        return 'None'
    text = str(value).replace('\n', ' ⏎ ')
    if len(text) <= max_len:
        return text
    return f'{text[:max_len]}… [всего {len(text)} симв.]'


@contextmanager
def timed(log: logging.Logger, action: str, level: int = logging.INFO):
    """
    Контекстный менеджер для замера времени операции.

        with timed(log, 'эмбеддинг'):
            ...
    """
    start = time.perf_counter()
    log.debug(f'▶ Начало: {action}')
    try:
        yield
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        log.exception(f'✖ Ошибка в "{action}" после {elapsed:.1f} мс')
        raise
    else:
        elapsed = (time.perf_counter() - start) * 1000
        log.log(level, f'✔ {action}: {elapsed:.1f} мс')

"""热词文件监视器 - 自动重载热词文件

使用 watchdog 监控热词文件变化，自动触发重载。
"""
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object

logger = logging.getLogger(__name__)


class HotwordFileHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """热词文件变化处理器"""

    def __init__(
        self,
        watched_files: Set[str],
        callback: Callable[[str], None],
        debounce_delay: float = 3.0
    ):
        """
        初始化处理器

        Args:
            watched_files: 要监控的文件名集合
            callback: 文件变化时的回调函数，参数为文件路径
            debounce_delay: 防抖延迟（秒），避免频繁触发
        """
        self.watched_files = watched_files
        self.callback = callback
        self.debounce_delay = debounce_delay
        self._pending: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def on_modified(self, event):
        """文件修改事件"""
        if event.is_directory:
            return
        self._handle_event(event.src_path)

    def on_created(self, event):
        """文件创建事件"""
        if event.is_directory:
            return
        self._handle_event(event.src_path)

    def _handle_event(self, path: str):
        """处理文件事件"""
        filename = Path(path).name
        if filename not in self.watched_files:
            return

        with self._lock:
            self._pending[path] = time.time()

            # 取消现有定时器
            if self._timer is not None:
                self._timer.cancel()

            # 设置新的延迟触发
            self._timer = threading.Timer(self.debounce_delay, self._fire_callbacks)
            self._timer.start()

    def _fire_callbacks(self):
        """触发回调"""
        with self._lock:
            paths = list(self._pending.keys())
            self._pending.clear()
            self._timer = None

        for path in paths:
            try:
                logger.info(f"Hotword file changed: {path}")
                self.callback(path)
            except Exception as e:
                logger.error(f"Error reloading hotword file {path}: {e}")


class HotwordWatcher:
    """热词文件监视器"""

    def __init__(
        self,
        watch_dir: str,
        on_hotwords_change: Optional[Callable[[str], None]] = None,
        on_context_hotwords_change: Optional[Callable[[str], None]] = None,
        on_rules_change: Optional[Callable[[str], None]] = None,
        on_rectify_change: Optional[Callable[[str], None]] = None,
        debounce_delay: float = 3.0,
        hotwords_filename: str = "hotwords.txt",
        context_hotwords_filename: str = "hotwords-context.txt",
        rules_filename: str = "hot-rules.txt",
        rectify_filename: str = "hot-rectify.txt",
    ):
        """
        初始化监视器

        Args:
            watch_dir: 监控目录
            on_hotwords_change: 热词文件变化回调
            on_rules_change: 规则文件变化回调
            on_rectify_change: 纠错历史文件变化回调
            debounce_delay: 防抖延迟（秒）
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not installed, file watching disabled")
            self._observer = None
            return

        self.watch_dir = Path(watch_dir)
        self._callbacks: Dict[str, Callable[[str], None]] = {}
        self._observer: Optional[Observer] = None

        # 注册回调（文件名可配置，便于多环境/多租户部署）
        if on_hotwords_change:
            self._callbacks[str(hotwords_filename)] = on_hotwords_change
        if on_context_hotwords_change:
            self._callbacks[str(context_hotwords_filename)] = on_context_hotwords_change
        if on_rules_change:
            self._callbacks[str(rules_filename)] = on_rules_change
        if on_rectify_change:
            self._callbacks[str(rectify_filename)] = on_rectify_change

        # 创建处理器
        self._handler = HotwordFileHandler(
            watched_files=set(self._callbacks.keys()),
            callback=self._dispatch_callback,
            debounce_delay=debounce_delay
        )

    def _dispatch_callback(self, path: str):
        """分发回调"""
        filename = Path(path).name
        callback = self._callbacks.get(filename)
        if callback:
            callback(path)

    def start(self):
        """启动监视器"""
        if not WATCHDOG_AVAILABLE or not self._callbacks:
            return

        try:
            self._observer = Observer()
            self._observer.schedule(self._handler, str(self.watch_dir), recursive=False)
            self._observer.start()
            logger.info(f"Started watching hotword files in {self.watch_dir}")
        except Exception as e:
            # Common failure modes:
            # - inotify watch limit reached (OSError: [Errno 28] ENOSPC)
            # - permission errors in restricted environments
            # In these cases hotword watching is optional; degrade gracefully.
            logger.warning(f"Failed to start hotword file watcher (disabled): {e}")
            try:
                if self._observer is not None:
                    self._observer.stop()
            except Exception:
                pass
            self._observer = None

    def stop(self):
        """停止监视器"""
        if self._observer is not None:
            try:
                self._observer.stop()
            except Exception:
                pass

            try:
                # `join()` raises if the thread was never started; guard it.
                if getattr(self._observer, "is_alive", None) and self._observer.is_alive():
                    self._observer.join(timeout=5)
            except Exception:
                pass

            self._observer = None
            logger.info("Stopped hotword file watcher")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# 全局监视器实例
_watcher: Optional[HotwordWatcher] = None


def get_hotword_watcher() -> Optional[HotwordWatcher]:
    """获取全局监视器实例"""
    return _watcher


def setup_hotword_watcher(
    watch_dir: str,
    on_hotwords_change: Optional[Callable[[str], None]] = None,
    on_context_hotwords_change: Optional[Callable[[str], None]] = None,
    on_rules_change: Optional[Callable[[str], None]] = None,
    on_rectify_change: Optional[Callable[[str], None]] = None,
    debounce_delay: float = 3.0,
    hotwords_filename: str = "hotwords.txt",
    context_hotwords_filename: str = "hotwords-context.txt",
    rules_filename: str = "hot-rules.txt",
    rectify_filename: str = "hot-rectify.txt",
) -> Optional[HotwordWatcher]:
    """设置并启动全局监视器"""
    global _watcher

    if _watcher is not None:
        try:
            _watcher.stop()
        except Exception:
            pass

    _watcher = HotwordWatcher(
        watch_dir=watch_dir,
        on_hotwords_change=on_hotwords_change,
        on_context_hotwords_change=on_context_hotwords_change,
        on_rules_change=on_rules_change,
        on_rectify_change=on_rectify_change,
        debounce_delay=debounce_delay,
        hotwords_filename=hotwords_filename,
        context_hotwords_filename=context_hotwords_filename,
        rules_filename=rules_filename,
        rectify_filename=rectify_filename,
    )
    try:
        _watcher.start()
    except Exception as e:
        # Defensive: `HotwordWatcher.start()` should already be best-effort, but
        # keep the global setup resilient.
        logger.warning(f"Failed to start global hotword watcher (ignored): {e}")
    return _watcher


def stop_hotword_watcher():
    """停止全局监视器"""
    global _watcher
    if _watcher is not None:
        _watcher.stop()
        _watcher = None

import os
import json
import threading
import time
from typing import Dict, Any, Optional

class StateManager:
    def __init__(self, file_path: str, poll_interval: float = 0.01, log_interval: float = 4.0):
        self.file_path = file_path
        self.poll_interval = poll_interval
        self.log_interval = log_interval  # 日志打印间隔

        # 保护共享状态的锁
        self._lock = threading.Lock()

        # 当前缓存的状态和最后修改时间
        self._state: Optional[Dict[Any, Any]] = None
        self._mtime: float = 0.0

        # 统计变量
        self._update_count = 0          # 自启动以来总更新次数
        self._last_log_time = time.time()
        self._updates_in_last_log = 0   # 上一个周期内的更新数

        # Condition 用于通知等待者
        self._condition = threading.Condition(lock=self._lock)

        # 控制标志
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._logger_thread: Optional[threading.Thread] = None

    def start_monitor(self):
        """启动后台监控线程和日志线程"""
        if self._running:
            return
        self._running = True

        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        self._logger_thread = threading.Thread(target=self._log_loop, daemon=True)
        self._logger_thread.start()

    def stop_monitor(self):
        """停止所有后台线程"""
        self._running = False
        with self._condition:
            self._condition.notify_all()

    def _monitor_loop(self):
        """后台循环：检查文件是否更新"""
        while self._running:
            try:
                current_mtime = os.path.getmtime(self.file_path)
                if current_mtime > self._mtime:
                    new_state = self._read_file()
                    if new_state is not None:
                        with self._condition:
                            self._state = new_state
                            self._mtime = current_mtime
                            self._update_count += 1
                            self._condition.notify_all()  # 广播更新
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[StateManager] Error reading {self.file_path}: {e}")
            time.sleep(self.poll_interval)

    def _log_loop(self):
        """日志线程：每 log_interval 秒打印一次更新频率"""
        while self._running:
            now = time.time()
            with self._lock:
                updates_since_last = self._update_count - self._updates_in_last_log
                print(f"[StateManager | LOG] State updated {updates_since_last} times in the last {self.log_interval:.1f}s "
                      f"(total: {self._update_count})")
                self._updates_in_last_log = self._update_count
            # 等待下一个周期（避免漂移）
            next_time = ((now // self.log_interval) + 1) * self.log_interval
            time.sleep(max(0, next_time - time.time()))

    def write_state(self, data: Dict[Any, Any]) -> bool:
        """
        安全写入状态文件，并触发通知。
        """
        try:
            with self._condition:
                self._state = data.copy()
                self._mtime = time.time()
                self._update_count += 1
                self._condition.notify_all()

            return True
        except Exception as e:
            print(f"[StateManager] Failed to write {self.file_path}: {e}")
            return False

    def get_state(self, blocking: bool = False, timeout: float = None) -> Optional[Dict[Any, Any]]:
        """
        获取最新状态。

        :param blocking: 是否阻塞直到有新状态
        :param timeout: 超时时间（秒），仅在 blocking=True 时有效
        :return: 最新状态字典 or None
        """
        with self._condition:
            if blocking:
                current_time = self._mtime
                if not self._condition.wait_for(lambda: self._mtime > current_time, timeout=timeout):
                    return None
            return self._state.copy() if self._state is not None else None

    @property
    def latest_timestamp(self) -> float:
        with self._lock:
            return self._mtime

    @property
    def total_updates(self) -> int:
        with self._lock:
            return self._update_count
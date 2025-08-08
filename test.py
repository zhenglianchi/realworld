import queue
import threading
class SharedQueue:
    """
    线程安全的共享队列，支持：
    - 写入：传入一个 array 列表，将其元素按顺序放入队列
    - 读取：实时读取当前队列内容（返回 list），无更新时返回旧队列
    """
    def __init__(self):
        self._queue = queue.Queue()
        self._data = []  # 实时可读的列表（保护副本）
        self._lock = threading.Lock()  # 保证线程安全

    def put_all(self, items):
        """
        写入函数：传入一个 array 列表，将其元素按顺序放入队列
        并立即更新实时读取的副本
        """
        with self._lock:
            # 清空旧队列（可选：若要完全替换）
            while not self._queue.empty():
                self._queue.get()

            # 批量写入新元素
            for item in items:
                self._queue.put(item)

            # 更新实时读取的副本
            self._data = list(self._queue.queue)

    def get_all(self):
        """
        实时读取函数：返回当前队列的副本（list）
        如果队列未更新，返回上一次的内容；更新后自动返回新内容
        """
        with self._lock:
            return list(self._data)  # 返回副本，避免外部修改
        
    def remove_front(self):
        """
        外部调用：删除 _data 的第一个元素
        用于表示“该路径点已到达，不再需要”
        """
        with self._lock:
            if len(self._data) > 0:
                self._data.pop(0)

    def clear(self):
        """清空队列"""
        with self._lock:
            while not self._queue.empty():
                self._queue.get()
            self._data.clear()

    def empty(self):
        """检查队列是否为空"""
        with self._lock:
            return self._queue.empty()

    def size(self):
        """获取队列大小"""
        with self._lock:
            return len(self._data)
        
a=[1,2,3,4,5,6]
test = SharedQueue()
test.put_all(a)
c = test.get_all()
for i in range(5):
    test.remove_front()
c = test.get_all()
print(c)
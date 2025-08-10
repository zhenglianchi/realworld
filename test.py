import threading
import logging

# 配置日志格式，包含文件名和行号
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s [%(threadName)s] %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
)

def worker1():
    try:
        print("1开始工作")
        print(1 / 0)
    except Exception as e:
        logging.error("发生异常", exc_info=True)  # exc_info=True 会打印完整堆栈

def worker2():
    try:
        print("2开始工作")
        print(1 / 0)
    except Exception as e:
        logging.error("发生异常", exc_info=True)  # exc_info=True 会打印完整堆栈

t1 = threading.Thread(target=worker1, name="WorkerThread1")
t2 = threading.Thread(target=worker2, name="WorkerThread2")
t1.start()
t2.start()
t1.join()
t2.join()
import threading
import traceback
import logging
# 配置日志格式，包含文件名和行号
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s [%(threadName)s] %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
)

class Update_State_Thread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exc = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self.exc = e
            logging.error(f"Error in Update_State_Thread thread",exc_info=True)
        else:
            self.exc = None

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self.exc:
            raise self.exc
        
class Execute_Thread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exc = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self.exc = e
            logging.error(f"Error in Execute_Thread thread",exc_info=True)
        else:
            self.exc = None

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self.exc:
            raise self.exc

class Low_Execute_Thread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exc = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self.exc = e
            logging.error(f"Error in Low_Execute_Thread thread",exc_info=True)
        else:
            self.exc = None

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self.exc:
            raise self.exc

class traj_Thread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exc = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self.exc = e
            logging.error(f"Error in traj_Thread thread",exc_info=True)
        else:
            self.exc = None

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self.exc:
            raise self.exc
        
class map_Thread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exc = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self.exc = e
            logging.error(f"Error in map_Thread thread",exc_info=True)
        else:
            self.exc = None

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self.exc:
            raise self.exc
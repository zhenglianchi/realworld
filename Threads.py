import threading
import traceback

class Update_State_Thread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exc = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self.exc = e
            traceback.print_exc()  # 打印完整的堆栈信息
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
            traceback.print_exc()  # 打印完整的堆栈信息
        else:
            self.exc = None

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self.exc:
            raise self.exc
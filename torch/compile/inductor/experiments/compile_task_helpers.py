"""可被真正编译 worker 导入、pickle 的普通 Python 任务；不启动设备。"""
import os
import threading

def identity():
    return {'pid':os.getpid(),'thread':threading.get_ident()}

def failing():
    raise ValueError('learning compile task failed')

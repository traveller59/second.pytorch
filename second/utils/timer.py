import time 
from contextlib import contextmanager

@contextmanager
def simple_timer(name=''):
    t = time.time()
    yield 
    print(f"{name} exec time: {time.time() - t}")
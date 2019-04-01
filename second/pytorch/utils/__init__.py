import time 
import contextlib 
import torch 

@contextlib.contextmanager
def torch_timer(name=''):
    torch.cuda.synchronize()
    t = time.time()
    yield
    torch.cuda.synchronize()
    print(name, "time:", time.time() - t)
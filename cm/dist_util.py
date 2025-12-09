"""
Helpers for distributed training with fallback to single-process mode.
"""

import io
import os
import socket
import torch as th

try:
    import torch.distributed as dist
except ImportError:
    # 如果没有安装torch.distributed，创建一个模拟版本
    dist = None

# 强制使用单机模式，避免MPI依赖
class MockMPI:
    class Comm:
        def __init__(self):
            self.rank = 0
            self.size = 1
            
        def Get_rank(self):
            return 0
        
        def Get_size(self):
            return 1
        
        def bcast(self, data, root=0):
            return data
    
    COMM_WORLD = Comm()

# 直接使用模拟MPI对象
MPI = MockMPI()

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group with fallback to single-process mode.
    """
    # 仅在dist可用且未初始化时尝试设置
    if dist is not None and not dist.is_initialized():
        try:
            # 使用本地模式，避免实际的分布式设置
            backend = "gloo"  # 使用CPU后端
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(_find_free_port())
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            dist.init_process_group(backend=backend, init_method="env://")
        except Exception as e:
            print(f"Warning: Failed to initialize distributed training: {e}")
            print("Running in single-process mode.")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file in single-process mode.
    """
    # 简化的文件加载，不使用MPI
    try:
        # 尝试直接使用torch加载
        return th.load(path, **kwargs)
    except Exception:
        # 如果直接加载失败，尝试使用文件读取
        with open(path, "rb") as f:
            data = f.read()
        return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    In single-process mode, this is a no-op.
    """
    # 在单机模式下，不需要同步
    if dist is not None and dist.is_initialized():
        for p in params:
            with th.no_grad():
                dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

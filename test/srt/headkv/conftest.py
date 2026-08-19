"""(current-main) HeadKV 纯 CPU 测试的共享初始化。

HeadReallocAttnBackend 在 __init__ 中读取 get_parallel().attn_tp_size,
而 main 的 attn_tp_size 会落到 get_attn_tp_group()(需要 TP group 已初始化,
无默认回退)。这里用 gloo(CPU)单进程并行初始化 tp=1,供 runtime 测试使用。
纯算法测试(budget/partition/duo/rlkv)不依赖此行,但运行也无害。
"""
import os

import pytest
import torch

from sglang.srt.distributed import parallel_state


@pytest.fixture(scope="session", autouse=True)
def init_headkv_parallel():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29503")
    if not torch.distributed.is_initialized():
        parallel_state.init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend="gloo"
        )
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    yield

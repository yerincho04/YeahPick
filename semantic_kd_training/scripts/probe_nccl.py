"""Show the NCCL library loaded by each torchrun worker and test one collective."""

import os

import torch
import torch.distributed as dist


rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)


def report(stage: str) -> None:
    with open("/proc/self/maps", encoding="utf-8") as maps:
        libraries = sorted({line.rsplit(None, 1)[-1].strip() for line in maps
                            if any(name in line for name in
                                   ("libnccl.so", "libcudart.so", "libcuda.so"))})
    print(f"rank={rank} stage={stage} torch={torch.__version__} "
          f"cuda={torch.version.cuda} nccl={torch.cuda.nccl.version()} "
          f"preload={os.environ.get('LD_PRELOAD', '')} "
          f"libraries={libraries}", flush=True)


report("before")
dist.init_process_group("nccl")
value = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
dist.all_reduce(value)
report("after")
print(f"rank={rank} all_reduce={value.item()}", flush=True)
dist.destroy_process_group()

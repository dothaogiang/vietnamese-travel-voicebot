"""Report the actual PyTorch/CUDA runtime visible to this Python process."""

import argparse
import platform
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="exit nonzero unless CUDA is usable")
    parser.add_argument("--min-free-vram-gib", type=float, default=0., help="require this much free VRAM on cuda:0")
    args = parser.parse_args()
    print(f"python={sys.version.split()[0]}")
    print(f"python_executable={sys.executable}")
    print(f"conda_environment={sys.prefix if (Path(sys.prefix) / 'conda-meta').is_dir() else 'not-conda'}")
    print(f"platform={platform.platform()}")
    try:
        import torch
    except ImportError:
        print("torch=not-installed")
        return 1 if args.require_cuda else 0
    available = torch.cuda.is_available()
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")
    print(f"cuda_available={available}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory_gib = properties.total_memory / (1024**3)
            print(f"cuda:{index}={properties.name}; vram_gib={memory_gib:.2f}")
        free_bytes, _ = torch.cuda.mem_get_info(0)
        print(f"cuda:0_free_vram_gib={free_bytes / (1024**3):.2f}")
        if free_bytes / (1024**3) < args.min_free_vram_gib:
            print("insufficient free VRAM; stop other GPU workloads before benchmarking")
            return 1
    if args.min_free_vram_gib > 0 and not available:
        return 1
    return 1 if args.require_cuda and not available else 0


if __name__ == "__main__":
    raise SystemExit(main())

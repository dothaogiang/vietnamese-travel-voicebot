"""Report the actual PyTorch/CUDA runtime visible to this Python process."""

import argparse
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="exit nonzero unless CUDA is usable")
    args = parser.parse_args()
    print(f"python={sys.version.split()[0]}")
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
    return 1 if args.require_cuda and not available else 0


if __name__ == "__main__":
    raise SystemExit(main())

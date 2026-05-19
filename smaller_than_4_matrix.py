import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import gc

def setup(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12356')  # 避免端口冲突
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def gpu_stress_worker(rank, world_size, args):
    setup(rank, world_size)
    
    # Matrix size < 4000, e.g., 3000
    N = args.matrix_size  # e.g., 3000
    print(f"[GPU {rank}] Starting stress test with matrix size {N}x{N}")

    # Estimate how many such matrices we can fit (roughly)
    # Each float32 matrix of size NxN uses N*N*4 bytes
    # We'll try to allocate multiple chunks to fill most of the memory
    try:
        # Get total memory and leave some margin (e.g., 1.5 GB free)
        total_mem = torch.cuda.get_device_properties(rank).total_memory
        reserved = 1.5 * 1024**3  # 1.5 GB reserved for system
        usable_mem = total_mem - reserved
        
        tensor_bytes = N * N * 4  # float32
        max_tensors = int(usable_mem // tensor_bytes // 3)  # 3 tensors per group (A, B, C)
        num_groups = max(1, min(max_tensors, 8))  # at least 1, at most 8 groups

        print(f"[GPU {rank}] Allocating {num_groups} tensor groups to fill ~{(1 - reserved/total_mem)*100:.1f}% of memory")

        # Pre-allocate memory pool
        pool = []
        for _ in range(num_groups):
            A = torch.randn(N, N, device=rank, dtype=torch.float32)
            B = torch.randn(N, N, device=rank, dtype=torch.float32)
            C = torch.zeros(N, N, device=rank, dtype=torch.float32)
            pool.append((A, B, C))

        # Main stress loop
        iteration = 0
        while True:
            for i in range(len(pool)):
                A, B, C = pool[i]

                # Intensive computation sequence
                C = torch.matmul(A, B)
                C = torch.nn.functional.gelu(C)          # more expensive than ReLU
                C = torch.matmul(C, B.transpose(0, 1))
                C = torch.sinh(torch.clamp(C, -5, 5))    # non-linear, compute-heavy
                C = torch.matmul(C, A)

                # Prevent fusion/optimization by reassigning
                pool[i] = (
                    torch.randn_like(A),
                    torch.randn_like(B),
                    C
                )

                iteration += 1
                if iteration % 50 == 0:
                    torch.cuda.synchronize(rank)
                    # print(f"[GPU {rank}] Completed {iteration} iterations")

            torch.cuda.synchronize(rank)

            # Optional: gather (adds inter-GPU traffic, increases load)
            if args.gather:
                dummy = torch.randn(N // 10, device=rank)
                gathered = [torch.zeros_like(dummy) for _ in range(world_size)]
                dist.all_gather(gathered, dummy)

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"[GPU {rank}] OOM error! Reduce matrix size or num_groups.")
        else:
            print(f"[GPU {rank}] RuntimeError: {e}")
    except KeyboardInterrupt:
        print(f"[GPU {rank}] Stopped by user")
    finally:
        cleanup()
        gc.collect()
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="GPU stress test with matrices smaller than 4K")
    parser.add_argument('--matrix-size', type=int, default=3000,
                        help='Matrix size (must be < 4000, default: 3000)')
    parser.add_argument('--gather', action='store_true',
                        help='Enable all_gather to add inter-GPU communication load')
    args = parser.parse_args()

    if args.matrix_size >= 4000:
        raise ValueError("Matrix size must be smaller than 4000!")

    if not torch.cuda.is_available():
        print("No CUDA GPUs available. Exiting.")
        return

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPUs detected.")
        return

    print(f"Detected {world_size} GPU(s). Starting stress test...")

    # Optimize for performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)  # Disable autograd for speed

    mp.spawn(
        gpu_stress_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['PYTHONUNBUFFERED'] = '1'  # Ensure print flushes immediately
    main()

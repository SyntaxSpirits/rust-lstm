"""PyTorch counterpart of `examples/benchmark.rs`.

    python validation/benchmark_pytorch.py --threads 1 --dtype float64
"""

import argparse
import statistics
import time

import torch

INPUT = 16
SEQ_LEN = 50


def median_micros(repeats, f):
    for _ in range(-(-repeats // 10)):
        f()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        f()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)

    print("task,hidden,batch,microseconds")
    for hidden in (16, 64, 256):
        lstm = torch.nn.LSTM(INPUT, hidden, 1).to(dtype)
        x = torch.full((1, 1, INPUT), 0.1, dtype=dtype)
        state = (torch.zeros(1, 1, hidden, dtype=dtype),) * 2

        def step():
            with torch.no_grad():
                lstm(x, state)

        print(f"inference_step,{hidden},1,{median_micros(2000, step):.2f}")

        for batch in (1, 16):
            t = torch.arange(SEQ_LEN, dtype=dtype)[:, None, None] * 0.1
            xs = torch.sin(t).expand(SEQ_LEN, batch, INPUT).contiguous()
            ys = torch.cos(t).expand(SEQ_LEN, batch, hidden).contiguous()

            def train():
                lstm.zero_grad()
                out, _ = lstm(xs)
                ((out - ys) ** 2).mean(dim=(1, 2)).sum().backward()

            repeats = 30 if hidden == 256 else 200
            print(f"forward_backward_T{SEQ_LEN},{hidden},{batch},{median_micros(repeats, train):.2f}")


if __name__ == "__main__":
    main()

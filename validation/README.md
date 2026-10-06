# Validation

Reference checks that are not part of `cargo test` because they need PyTorch.

```bash
pip install -r validation/requirements.txt

# Outputs and BPTT gradients against torch.nn.LSTM (float64)
cargo run --release --example export_reference_cases -- validation/cases.json
python validation/pytorch_parity.py validation/cases.json

# Timings
cargo run --release --example benchmark > validation/results/rust_timings.csv
python validation/benchmark_pytorch.py --threads 1 --dtype float64 > validation/results/pytorch_f64_1t.csv
```

`results/` holds the output of these commands on an Apple M1 Pro (macOS 14.5,
Rust 1.86, PyTorch 2.9.1). Finite-difference gradient checks for every layer run
with `cargo test --test gradient_check`.

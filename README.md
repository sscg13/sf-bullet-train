# Fairy-Stockfish Shatranj NNUE experiment

Branch `fsf-shatranj` starts from Yoshie2000's `fix-inputs` commit
`5c171718ca0d93353e557260043051dddbb6bd7f`. It replaces the SFNNv9
network with Fairy-Stockfish's **HalfKAv2Variants / HalfKAv2(Friend)**
architecture and exports networks that the unmodified engine can load.

## Architecture verified from source

Reference checkouts are in `.upstream/` (ignored by Git):

- Fairy-Stockfish: [`9f778da667f6e07dae1e85d3e2ea204fc6dee94d`](https://github.com/fairy-stockfish/Fairy-Stockfish/tree/9f778da667f6e07dae1e85d3e2ea204fc6dee94d).
- variant-nnue-pytorch: [`b15df38a9aae8ab9b40b2378020b3099c7c5d179`](https://github.com/fairy-stockfish/variant-nnue-pytorch/tree/b15df38a9aae8ab9b40b2378020b3099c7c5d179).
- Bullet remains pinned to `e37db79ee3ee54be2a2f310004b1b82f3b60c0ca`.

The [engine architecture](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/9f778da667f6e07dae1e85d3e2ea204fc6dee94d/src/nnue/nnue_architecture.h)
and [PyTorch model](https://github.com/fairy-stockfish/variant-nnue-pytorch/blob/b15df38a9aae8ab9b40b2378020b3099c7c5d179/model.py)
agree on the following structure:

| Component | Shatranj configuration |
| --- | --- |
| Real input features | 64 king squares × 11 piece planes × 64 squares = **45,056** |
| Feature transformer | Shared affine layer, **512** outputs per perspective |
| Transformer activation | Clipped ReLU to [0, 1] during training, [0, 127] in engine |
| Dense stack | STM then opponent concatenation: **1024 → 16 → 32 → 1** |
| Dense activations | Clipped ReLU after 16 and 32 neurons |
| Material buckets | **8** stacks and **8** PSQT outputs per feature |
| Bucket selection | `min((piece_count - 1) * 8 / 32, 7)` |
| PSQT contribution | `(STM PSQT - opponent PSQT) / 2`, added to dense output |

There is no horizontal mirroring, pairwise multiplication, squared activation,
or skip neuron in this engine architecture. Both kings share plane 10.
Black's perspective flips ranks (`square ^ 56`) and swaps friendly/enemy
piece planes. Its king square is rank-flipped as well.

FEN symbols follow the usual chess letters, with their Shatranj meanings:

| FEN symbol | Meaning | Friendly / enemy feature planes |
| --- | --- | --- |
| P/p | Pawn | 0 / 1 |
| N/n | Knight | 2 / 3 |
| R/r | Rook | 4 / 5 |
| Q/q | Ferz (engine `FERS`) | 6 / 7 |
| B/b | Alfil | 8 / 9 |
| K/k | King | 10 / 10 |

The unusual plane ordering comes from [variant initialization](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/9f778da667f6e07dae1e85d3e2ea204fc6dee94d/src/variant.cpp):
non-king piece types are enumerated in engine `PieceType` order, and the king
is placed last. `src/inputs.rs` maps Bullet's P,N,B,R,Q,K storage to this order.
The optional 768-input training factorizer distinguishes friendly/enemy kings;
its weights are merged into the 45,056 real features before export.

## Prepare data, train, export

Use scored **Shatranj** positions with records in this format:

```text
rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w - - 0 1 | 0 | 0.5
7k/8/3q4/5b2/2B5/4Q3/8/K7 b - - 0 1 | 120 | 1.0
```

Both score and result are **White-relative**. Result is 0.0 for a Black win,
0.5 for a draw, and 1.0 for a White win. Preparation changes labels, colours,
and square orientation to the side to move. B/b means alfil and Q/q means
ferz; neither letter is replaced. Scores must fit in [-32767, 32767].
FENs must have exactly one king per colour, at most 32 pieces, and no castling
or en passant. Structural validation does not determine position legality or
filter checks, captures, terminal positions, or search outliers; curate and
shuffle the training positions before conversion.

The default build uses CUDA and needs a compatible GPU and CUDA toolkit:

```sh
cargo build --release --locked
cargo run --release --locked -- prepare shatranj.txt shatranj.bf
cargo run --release --locked -- train shatranj.bf --superbatches 150 --output checkpoints
python convert_quantised_to_pytorch.py checkpoints/shatranj-final.bin shatranj.nnue
```

`prepare` streams text into 32-byte Bullet `ChessBoard` records, refuses to
overwrite an existing output, and removes incomplete output on failure.
Already prepared Bullet datasets may be passed directly to `train`.
Orthodox Stockfish binpacks and variant-nnue-pytorch's packed datasets are
different formats and must be converted to the documented text/Bullet format.
No chess move generator or orthodox check/capture filter is used by this branch.

Use `--help` for schedule and data loading options. Training uses Ranger,
`sigmoid(score / 410)` targets and `sigmoid(network_output * 600 / 361)`
predictions, following the PyTorch model's scales. `--wdl` blends game results
into the targets; 0 uses search scores and 1 uses results only. The feature
transformer and PSQT are trained jointly. Quantization happens when saving;
this is a floating-point trainer, so exported integer evaluations can differ
slightly from training predictions.

To resume a checkpoint saved after superbatch 10:

```sh
cargo run --release --locked -- train shatranj.bf --resume checkpoints/shatranj-10 --start-superbatch 11 --superbatches 140
```

The batch size and batches per superbatch should stay the same when resuming
so the loader's position in the dataset remains consistent. Resume only
checkpoints produced by this architecture, not SFNNv9 checkpoints.

## Export format and engine use

The converter keeps its historical name `convert_quantised_to_pytorch.py`,
but writes a standard Fairy-Stockfish `.nnue` directly, using only Python's
standard library. The dimensions are fixed to this branch's architecture;
an SFNNv9 weight file is rejected.

| Header / scale | Value |
| --- | --- |
| File version | `0x7AF32F20` |
| Feature set hash | `0x5F234CB8` |
| Feature transformer hash | `0x5F2348B8` |
| Dense stack hash | `0x633376CA` |
| Network hash | `0x3C103E72` |
| FT bias and weights | int16, scale 127 |
| PSQT weights | int32, scale 9600 |
| Hidden bias / weights | int32 scale 8128 / int8 scale 64 |
| Output bias / weights | int32 scale 9600 / int8 scale **9600/127** |

The output weight scale is fractional, not truncated to 75. Dense inputs are
padded to multiples of 32 (16 real inputs plus 16 zero bytes per row in the
second layer). Transformer data is feature-major; dense weights are
output-major. The converter interleaves each stack's biases and weights,
writes raw little-endian arrays without LEB128 compression, and removes
Bullet's trailing alignment bytes. Hash construction and layout follow the
[upstream serializer](https://github.com/fairy-stockfish/variant-nnue-pytorch/blob/b15df38a9aae8ab9b40b2378020b3099c7c5d179/serialize.py).

**The network basename must begin with `shatranj`**, or the engine will not
select it for this variant. Set these UCI options, using an absolute file path:

```text
setoption name UCI_Variant value shatranj
setoption name Use NNUE value true
setoption name EvalFile value /absolute/path/shatranj.nnue
isready
position startpos
eval
```

The engine should report `NNUE evaluation using ... enabled`. Its final
evaluation also applies engine-side scaling and adjustments; it need not
equal the raw NNUE output used by the compatibility tests.

## Reproduce validation

CPU builds work without a GPU; select exactly one backend:

```sh
cargo test --locked --no-default-features --features cpu
python -m unittest discover -s tests -p 'test_*.py' -v
cargo run --no-default-features --features cpu -- prepare tests/fixtures/shatranj.txt checkpoints/shatranj-smoke.bf
cargo run --no-default-features --features cpu -- train checkpoints/shatranj-smoke.bf --batch-size 4 --batches-per-superbatch 2 --superbatches 1 --save-rate 1 --threads 1 --output checkpoints/fsf-smoke
python convert_quantised_to_pytorch.py checkpoints/fsf-smoke/shatranj-final.bin checkpoints/fsf-smoke/shatranj-smoke.nnue
```

Create the checkpoint directory before preparing the fixture if it does not
exist. The four fixture records exercise formats and training; they are not
a dataset for playing strength.

For exact engine validation, build the unmodified reference engine in its
`src` directory with `make -j4 ARCH=x86-64 build`. From this project root in
a POSIX shell, link the probe against the engine's original object files
(excluding `main.o` and compiler-generated LTO temporary objects):

```sh
objects=$(find .upstream/Fairy-Stockfish/src -maxdepth 1 -name '*.o' ! -name 'main.o' ! -name '*.*.o')
g++ -std=c++17 -O2 -flto -DNDEBUG -DNNUE_EMBEDDING_OFF -DUSE_PTHREADS -DIS_64BIT -DUSE_SSE2 -msse2 -I.upstream/Fairy-Stockfish/src tests/fsf_probe.cpp $objects -lpthread -o checkpoints/fsf-probe
python tests/check_engine.py checkpoints/fsf-probe checkpoints/fsf-smoke/shatranj-final.bin checkpoints/fsf-smoke/shatranj-smoke.nnue
FSF_PROBE="$PWD/checkpoints/fsf-probe" FSF_NETWORK="$PWD/checkpoints/fsf-smoke/shatranj-smoke.nnue" cargo test --locked --no-default-features --features cpu feature_indices_match_fairy_stockfish -- --ignored
```

On Windows/MSYS2 use `.exe` for the probe and engine executables. The probe's
flags must match the engine build; the commands above use `ARCH=x86-64`, with
SSE2 and standard 8×8 boards. `check_engine.py` checks all 64 king squares,
both perspectives, every material bucket and every piece type, plus the
training fixtures: **132 positions**. It requires exact agreement between
independent integer inference of the saved tensors and Fairy-Stockfish's
own feature transformer, PSQT, dense layers and raw NNUE evaluation.

Validated locally: Rust unit tests and Clippy, Python exporter tests, a
two-batch CPU training run and checkpoint resume, engine loading, direct Rust/engine feature
comparison, and exact integer parity on all 132 positions. Full CUDA
training and playing strength remain to be evaluated with a real dataset.

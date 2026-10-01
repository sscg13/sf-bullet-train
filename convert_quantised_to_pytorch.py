"""Convert Bullet tensors to Fairy-Stockfish's raw HalfKAv2 NNUE format.

Standard library only. The historical filename is retained for existing scripts.
This is HalfKAv2 512x2-16-32-1, not SFNNv9 or compressed LEB128.
"""
import argparse
import io
from pathlib import Path
import struct

VERSION = 0x7AF32F20
FEATURE_SET_HASH = 0x5F234CB8
INPUTS = 45056
L1, L2, L3 = 512, 16, 32
BUCKETS = 8
DEFAULT_DESCRIPTION = "Shatranj HalfKAv2 trained with sf-bullet-train."


def fc_hash():
    previous = 0xEC42E90D ^ (2 * L1)
    for outputs in (L2, L3, 1):
        value = (0xCC03DAE4 + outputs) ^ (previous >> 1) ^ ((previous << 31) & 0xFFFFFFFF)
        if outputs != 1:
            value = (value + 0x538D24C7) & 0xFFFFFFFF
        previous = value
    return previous


FC_HASH = fc_hash()
FT_HASH = FEATURE_SET_HASH ^ (2 * L1)
NETWORK_HASH = FC_HASH ^ FT_HASH
TENSORS = (
    ("l0b", 2 * L1), ("l0w", 2 * INPUTS * L1), ("pst", 4 * INPUTS * BUCKETS),
    ("l1b", 4 * L2 * BUCKETS), ("l1w", 2 * L1 * L2 * BUCKETS),
    ("l2b", 4 * L3 * BUCKETS), ("l2w", L2 * L3 * BUCKETS),
    ("l3b", 4 * BUCKETS), ("l3w", L3 * BUCKETS),
)
RAW_SIZE = sum(size for _, size in TENSORS)


def read_exact(source, size, label):
    value = source.read(size)
    if len(value) != size:
        raise ValueError(f"truncated {label}: expected {size} bytes, got {len(value)}")
    return value


def convert_stream(source, target, description=DEFAULT_DESCRIPTION):
    # Bullet pads to a multiple of 64 with the repeating word 'bullet'.
    # Validate first and never carry this padding into an NNUE file.
    source.seek(0, io.SEEK_END)
    length = source.tell()
    source.seek(0)
    padding = (-RAW_SIZE) % 64
    if length not in (RAW_SIZE, RAW_SIZE + padding):
        raise ValueError(f"expected {RAW_SIZE} raw bytes (or {RAW_SIZE + padding} with Bullet padding), got {length}; wrong architecture?")
    if length > RAW_SIZE:
        source.seek(RAW_SIZE)
        expected = (b"bullet" * ((padding + 5) // 6))[:padding]
        if source.read() != expected:
            raise ValueError("invalid Bullet alignment padding")
        source.seek(0)
    encoded = description.encode("utf-8")
    target.write(struct.pack("<III", VERSION, NETWORK_HASH, len(encoded)))
    target.write(encoded)
    target.write(struct.pack("<I", FT_HASH))
    for label, size in TENSORS[:3]:
        remaining = size
        while remaining:
            chunk_size = min(remaining, 1024 * 1024)
            target.write(read_exact(source, chunk_size, label))
            remaining -= chunk_size
    dense = {name: read_exact(source, size, name) for name, size in TENSORS[3:]}
    for bucket in range(BUCKETS):
        target.write(struct.pack("<I", FC_HASH))
        for layer, inputs, outputs in ((1, 2 * L1, L2), (2, L2, L3), (3, L3, 1)):
            biases, weights = dense[f"l{layer}b"], dense[f"l{layer}w"]
            target.write(biases[bucket * outputs * 4:(bucket + 1) * outputs * 4])
            start, extra = bucket * inputs * outputs, (-inputs) % 32
            for row in range(outputs):
                offset = start + row * inputs
                target.write(weights[offset:offset + inputs])
                target.write(bytes(extra))


def convert_binary_format(input_file, output_file, description=DEFAULT_DESCRIPTION):
    source_path, target_path = Path(input_file), Path(output_file)
    if source_path.resolve() == target_path.resolve():
        raise ValueError("input and output must be different files")
    temporary = target_path.with_name(target_path.name + ".tmp")
    with source_path.open("rb") as source:
        with temporary.open("xb") as target:
            try:
                convert_stream(source, target, description)
            except Exception:
                target.close()
                temporary.unlink()
                raise
    temporary.replace(target_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="quantised.bin or shatranj-final.bin from this branch")
    parser.add_argument("output_file", help="engine network; basename must start with shatranj")
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    args = parser.parse_args()
    try:
        convert_binary_format(args.input_file, args.output_file, args.description)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {args.output_file}; network hash {NETWORK_HASH:#010x}, FC hash {FC_HASH:#010x}")


if __name__ == "__main__":
    main()

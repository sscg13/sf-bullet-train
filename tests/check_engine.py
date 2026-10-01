"""Compare exported integer inference and features to unmodified FSF objects.

Usage: python tests/check_engine.py PROBE_EXE QUANTISED_BIN SHATRANJ_NNUE
The probe prints engine-produced features, PSQT, dense output and evaluation.
"""
import argparse
from array import array
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert_quantised_to_pytorch as converter


def read_tensors(path):
    data = {}
    with open(path, "rb") as file:
        for name, size in converter.TENSORS:
            kind = "h" if name in ("l0b", "l0w") else "i" if name == "pst" or name.endswith("b") else "b"
            values = array(kind)
            values.frombytes(converter.read_exact(file, size, name))
            if sys.byteorder != "little":
                values.byteswap()
            data[name] = values
    return data


def feature_indices(pieces, perspective):
    king = next(square for square, piece in pieces.items() if piece == ("K" if perspective == 0 else "k"))
    flip = perspective * 56
    king ^= flip
    order = {"P": 0, "N": 1, "R": 2, "Q": 3, "B": 4, "K": 5}
    return sorted((square ^ flip) + 64 * min(2 * order[piece.upper()] + (piece.islower() ^ bool(perspective)), 10) + 704 * king
                  for square, piece in pieces.items())


def fen_from_pieces(pieces, side):
    ranks = []
    for rank in range(7, -1, -1):
        row, empty = "", 0
        for file in range(8):
            piece = pieces.get(rank * 8 + file)
            if piece:
                if empty:
                    row += str(empty)
                row += piece
                empty = 0
            else:
                empty += 1
        if empty:
            row += str(empty)
        ranks.append(row)
    return "/".join(ranks) + f" {side} - - 0 1"


def positions():
    for line in (Path(__file__).parent / "fixtures" / "shatranj.txt").read_text().splitlines():
        fen = line.split("|")[0].strip()
        board, side, *_ = fen.split()
        pieces = {}
        for rank, row in enumerate(reversed(board.split("/"))):
            file = 0
            for symbol in row:
                if symbol.isdigit():
                    file += int(symbol)
                else:
                    pieces[8 * rank + file] = symbol
                    file += 1
        yield pieces, side, fen
    # Every king square, both sides to move, all material buckets, all types.
    for king in range(64):
        count = (2, 6, 10, 14, 18, 22, 26, 32)[king % 8]
        pieces = {king: "K", (king + 31) % 64: "k"}
        available = ((king + offset * 17) % 64 for offset in range(64))
        symbols = "PNBRQpnbrq"
        i = 0
        for square in available:
            if len(pieces) == count:
                break
            if square not in pieces:
                pieces[square] = symbols[i % len(symbols)]
                i += 1
        for side in ("w", "b"):
            yield pieces, side, fen_from_pieces(pieces, side)


def trunc_div(value, divisor):
    return (1 if value >= 0 else -1) * (abs(value) // divisor)


def infer(data, white, black, side, bucket):
    def accumulator(indices):
        return [max(0, min(127, data["l0b"][i] + sum(data["l0w"][feature * 512 + i] for feature in indices))) for i in range(512)]
    ours, theirs = (white, black) if side == "w" else (black, white)
    psqt = trunc_div(sum(data["pst"][i * 8 + bucket] for i in ours) - sum(data["pst"][i * 8 + bucket] for i in theirs), 2)
    value = accumulator(ours) + accumulator(theirs)
    for layer, inputs, outputs in ((1, 1024, 16), (2, 16, 32), (3, 32, 1)):
        biases, weights = data[f"l{layer}b"], data[f"l{layer}w"]
        result = []
        for row in range(outputs):
            start = (bucket * outputs + row) * inputs
            total = biases[bucket * outputs + row] + sum(weights[start + i] * value[i] for i in range(inputs))
            result.append(max(0, min(127, total >> 6)) if layer < 3 else total)
        value = result
    return psqt, value[0], trunc_div(psqt + value[0], 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe")
    parser.add_argument("quantised")
    parser.add_argument("nnue")
    args = parser.parse_args()
    data = read_tensors(args.quantised)
    cases = list(positions())
    result = subprocess.run([str(Path(args.probe).resolve()), str(Path(args.nnue).resolve())],
                            input="\n".join(fen for _, _, fen in cases) + "\n", text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr or f"probe exited with {result.returncode}")
    print(result.stderr.strip())
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(records) == len(cases), "probe omitted positions"
    for (pieces, side, fen), actual in zip(cases, records):
        white, black = feature_indices(pieces, 0), feature_indices(pieces, 1)
        assert sorted(actual["white"]) == white, f"White features differ: {fen}"
        assert sorted(actual["black"]) == black, f"Black features differ: {fen}"
        bucket = min((len(pieces) - 1) // 4, 7)
        assert actual["bucket"] == bucket, f"bucket differs: {fen}"
        expected = infer(data, white, black, side, bucket)
        observed = actual["psqt"], actual["dense"], actual["value"]
        assert observed == expected, f"integer inference differs: {fen}: {observed} != {expected}"
    print(f"Passed {len(cases)} positions: every king square, both perspectives, all eight buckets; exact integer evaluation matches.")


if __name__ == "__main__":
    main()

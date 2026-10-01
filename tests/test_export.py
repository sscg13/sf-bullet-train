import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert_quantised_to_pytorch as converter


class ExportTests(unittest.TestCase):
    def test_hashes_match_engine_architecture(self):
        self.assertEqual(converter.FC_HASH, 0x633376CA)
        self.assertEqual(converter.FT_HASH, 0x5F2348B8)
        self.assertEqual(converter.NETWORK_HASH, 0x3C103E72)

    def test_all_stacks_signed_weights_padding_and_utf8_description(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, net = Path(directory) / "raw.bin", Path(directory) / "shatranj.nnue"
            # Sparse file keeps this test cheap while preserving full dimensions.
            with raw.open("wb") as file:
                file.truncate(converter.RAW_SIZE)
            offsets, offset = {}, 0
            with raw.open("r+b") as file:
                for name, size in converter.TENSORS:
                    offsets[name] = offset
                    offset += size
                for bucket in range(8):
                    for layer, inputs, outputs in ((1, 1024, 16), (2, 16, 32), (3, 32, 1)):
                        file.seek(offsets[f"l{layer}b"] + 4 * bucket * outputs)
                        file.write(struct.pack("<i", -1000 - 10 * bucket - layer))
                        for row in range(outputs):
                            file.seek(offsets[f"l{layer}w"] + (bucket * outputs + row) * inputs)
                            file.write(struct.pack("<b", -100 + bucket + row))
            converter.convert_binary_format(raw, net, "Shatranj ♞")
            with net.open("rb") as file:
                self.assertEqual(struct.unpack("<III", file.read(12)), (0x7AF32F20, 0x3C103E72, len("Shatranj ♞".encode())))
                self.assertEqual(file.read(len("Shatranj ♞".encode())), "Shatranj ♞".encode())
                self.assertEqual(struct.unpack("<I", file.read(4))[0], 0x5F2348B8)
                file.seek(sum(size for _, size in converter.TENSORS[:3]), io.SEEK_CUR)
                for bucket in range(8):
                    self.assertEqual(struct.unpack("<I", file.read(4))[0], 0x633376CA)
                    for layer, inputs, outputs in ((1, 1024, 16), (2, 16, 32), (3, 32, 1)):
                        self.assertEqual(struct.unpack("<i", file.read(4))[0], -1000 - 10 * bucket - layer)
                        file.seek((outputs - 1) * 4, io.SEEK_CUR)
                        for row in range(outputs):
                            self.assertEqual(struct.unpack("<b", file.read(1))[0], -100 + bucket + row)
                            file.seek(inputs - 1, io.SEEK_CUR)
                            self.assertEqual(file.read((-inputs) % 32), bytes((-inputs) % 32))
                self.assertEqual(file.read(), b"")

    def test_invalid_sizes_are_rejected_before_writing(self):
        for size in (0, converter.RAW_SIZE - 1, converter.RAW_SIZE + 1, converter.RAW_SIZE + 64):
            with tempfile.TemporaryFile() as source:
                source.truncate(size)
                target = io.BytesIO()
                with self.assertRaises(ValueError):
                    converter.convert_stream(source, target)
                self.assertEqual(target.getvalue(), b"")

    def test_failure_preserves_existing_output_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "bad.bin", Path(directory) / "shatranj.nnue"
            source.write_bytes(b"bad")
            target.write_bytes(b"existing network")
            with self.assertRaises(ValueError):
                converter.convert_binary_format(source, target)
            self.assertEqual(target.read_bytes(), b"existing network")
            self.assertFalse(target.with_name(target.name + ".tmp").exists())

    def test_bullet_alignment_padding_is_removed(self):
        with tempfile.TemporaryFile() as source, tempfile.TemporaryFile() as target:
            source.truncate(converter.RAW_SIZE)
            source.seek(converter.RAW_SIZE)
            padding = (-converter.RAW_SIZE) % 64
            source.write((b"bullet" * 11)[:padding])
            converter.convert_stream(source, target, "")
            self.assertEqual(target.tell(), converter.RAW_SIZE + 16 + 8 * (4 + 32 * 16))
            source.seek(converter.RAW_SIZE)
            source.write(b"x" * padding)
            with self.assertRaises(ValueError):
                converter.convert_stream(source, io.BytesIO())


if __name__ == "__main__":
    unittest.main()

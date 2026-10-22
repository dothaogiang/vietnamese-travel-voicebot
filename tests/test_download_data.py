"""Offline tests for verified and safe VIVOS archive extraction."""

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
import wave

from vietnamese_travel_voicebot.data.download import install_vivos_archive
from vietnamese_travel_voicebot.data.manifest import build_vivos_manifests, read_manifest


class VivosArchiveTests(unittest.TestCase):
    @staticmethod
    def _wav_bytes() -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 160)
        return buffer.getvalue()

    def _archive(self, path: Path, *, malicious: bool = False) -> None:
        with tarfile.open(path, "w:gz") as archive:
            for split, speaker, utterance in (
                ("train", "speaker1", "speaker1_0001"),
                ("test", "speaker2", "speaker2_0001"),
            ):
                for name, data in (
                    (f"vivos/{split}/prompts.txt", f"{utterance} Xin chao\n".encode()),
                    (f"vivos/{split}/waves/{speaker}/{utterance}.wav", self._wav_bytes()),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            if malicious:
                info = tarfile.TarInfo("../../outside.txt")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))

    def test_installs_expected_layout_and_records_verified_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "vivos.tar.gz"
            self._archive(archive)
            expected_md5 = hashlib.md5(archive.read_bytes()).hexdigest()
            metadata = install_vivos_archive(archive, root / "raw" / "VIVOS", expected_md5=expected_md5)
            self.assertTrue((root / "raw" / "VIVOS" / "train" / "prompts.txt").is_file())
            self.assertEqual(metadata["md5"], expected_md5)
            self.assertEqual(metadata["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertTrue((root / "raw" / "vivos_metadata.json").is_file())
            manifests = build_vivos_manifests(root / "raw" / "VIVOS", root / "raw" / "manifests")
            self.assertEqual(len(read_manifest(manifests["train"])), 1)
            self.assertEqual(len(read_manifest(manifests["test"])), 1)

    def test_rejects_wrong_checksum_and_unsafe_archive_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "vivos.tar.gz"
            self._archive(archive)
            with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                install_vivos_archive(archive, root / "wrong" / "VIVOS", expected_md5="0" * 32)
            malicious = root / "unsafe.tar.gz"
            self._archive(malicious, malicious=True)
            expected_md5 = hashlib.md5(malicious.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                install_vivos_archive(malicious, root / "unsafe" / "VIVOS", expected_md5=expected_md5)
            self.assertFalse((root / "outside.txt").exists())

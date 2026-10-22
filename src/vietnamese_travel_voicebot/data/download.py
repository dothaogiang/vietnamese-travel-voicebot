"""Verified VIVOS archive download and safe local extraction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


VIVOS_URL = "https://zenodo.org/records/7068130/files/vivos.tar.gz?download=1"
VIVOS_MD5 = "72972a8b14050f3f11ea7c3debabd7af"


def _hash_file(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _verified_ssl_context() -> ssl.SSLContext:
    """Build a verified TLS context using configured or installed CA bundles.

    Conda Python may be built with a CA path from another machine. Prefer an
    explicitly configured bundle, then a valid OpenSSL path, the Linux system
    bundle, and finally certifi. Certificate verification remains enabled.
    """
    configured = (
        os.environ.get("SSL_CERT_FILE"),
        os.environ.get("REQUESTS_CA_BUNDLE"),
        os.environ.get("CURL_CA_BUNDLE"),
    )
    verify_paths = ssl.get_default_verify_paths()
    candidates = [
        *configured,
        verify_paths.cafile,
        "/etc/ssl/certs/ca-certificates.crt",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ssl.create_default_context(cafile=candidate)
    if verify_paths.capath and Path(verify_paths.capath).is_dir():
        return ssl.create_default_context(capath=verify_paths.capath)
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"unsafe archive member path: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"unsupported archive member type: {member.name}")
    return members


def _locate_dataset_root(extraction_root: Path) -> Path:
    candidates = [extraction_root]
    candidates.extend(path.parent.parent for path in extraction_root.rglob("prompts.txt"))
    valid = {
        candidate.resolve()
        for candidate in candidates
        if (candidate / "train" / "prompts.txt").is_file()
        and (candidate / "test" / "prompts.txt").is_file()
    }
    if len(valid) != 1:
        raise ValueError(
            "archive must contain exactly one VIVOS root with train/prompts.txt and test/prompts.txt; "
            f"found {len(valid)} candidate roots"
        )
    return next(iter(valid))


def install_vivos_archive(
    archive_path: str | Path,
    dataset_dir: str | Path,
    *,
    expected_md5: str = VIVOS_MD5,
    source_url: str = VIVOS_URL,
) -> dict[str, Any]:
    """Verify archive checksum, safely extract it, normalize its root, and record metadata."""
    archive = Path(archive_path).expanduser().resolve()
    destination = Path(dataset_dir).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"VIVOS archive not found: {archive}")
    actual_md5, actual_sha256 = _hash_file(archive)
    if actual_md5.lower() != expected_md5.lower():
        raise ValueError(f"VIVOS archive MD5 mismatch: expected {expected_md5}, got {actual_md5}")
    if destination.exists():
        raise FileExistsError(f"dataset destination already exists; inspect or remove it first: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".vivos-extract-", dir=destination.parent))
    extraction_root = staging / "content"
    extraction_root.mkdir()
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = _safe_members(tar)
            if hasattr(tarfile, "data_filter"):
                tar.extractall(extraction_root, members=members, filter="data")
            else:  # Python 3.10: member paths/types were validated above.
                tar.extractall(extraction_root, members=members)
        source_root = _locate_dataset_root(extraction_root)
        shutil.move(str(source_root), str(destination))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    metadata: dict[str, Any] = {
        "dataset": "VIVOS",
        "version": "1.0",
        "source_url": source_url,
        "doi": "10.5281/zenodo.7068130",
        "license": "CC BY-NC-SA 4.0",
        "purpose": "academic research; non-commercial use only",
        "archive_name": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "md5": actual_md5,
        "sha256": actual_sha256,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": destination.name,
    }
    metadata_path = destination.parent / "vivos_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def download_vivos(
    dataset_dir: str | Path,
    *,
    archive_path: str | Path | None = None,
    source_url: str = VIVOS_URL,
) -> dict[str, Any]:
    """Download the official VIVOS archive if needed, then verify and install it."""
    destination = Path(dataset_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"dataset destination already exists; inspect or remove it first: {destination}")
    archive = (
        Path(archive_path).expanduser().resolve()
        if archive_path is not None
        else destination.parent / "downloads" / "vivos.tar.gz"
    )
    if not archive.is_file():
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(archive.suffix + ".part")
        request = urllib.request.Request(source_url, headers={"User-Agent": "vietnamese-travel-voicebot/0.1"})
        response_url = source_url
        content_type = "unknown"
        content_length: int | None = None
        try:
            with urllib.request.urlopen(
                request, timeout=60, context=_verified_ssl_context()
            ) as response, partial.open("wb") as output:
                response_url = response.geturl()
                content_type = response.headers.get("Content-Type", "unknown")
                length_header = response.headers.get("Content-Length")
                content_length = int(length_header) if length_header and length_header.isdigit() else None
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    if output.tell() % (100 * 1024 * 1024) < len(chunk):
                        print(f"downloaded {output.tell() / (1024**2):.0f} MiB", file=sys.stderr)
        except Exception as exc:
            partial_size = partial.stat().st_size if partial.exists() else 0
            raise OSError(
                f"download interrupted after {partial_size} bytes; partial file kept at {partial}: {exc}"
            ) from exc
        actual_size = partial.stat().st_size
        if content_length is not None and actual_size != content_length:
            raise ValueError(
                f"download size mismatch: expected {content_length} bytes, got {actual_size}; "
                f"response Content-Type={content_type!r}, final URL={response_url!r}; "
                f"partial file kept at {partial}"
            )
        downloaded_md5, _ = _hash_file(partial)
        if downloaded_md5.lower() != VIVOS_MD5:
            raise ValueError(
                f"downloaded VIVOS archive MD5 mismatch: expected {VIVOS_MD5}, got {downloaded_md5}; "
                f"size={actual_size} bytes, Content-Type={content_type!r}, final URL={response_url!r}; "
                f"file kept at {partial} for inspection"
            )
        partial.replace(archive)
    return install_vivos_archive(archive, destination, source_url=source_url)

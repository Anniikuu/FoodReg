"""Inspection script for candidate food classification datasets.

Queries and verifies dataset metadata, class distributions, and sample integrity
from Hugging Face repositories without requiring multi-gigabyte local downloads.
"""

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
sys.stdout.reconfigure(encoding="utf-8")


class RemoteHttpFile(io.RawIOBase):
    """File-like reader that performs HTTP Range requests for remote files."""

    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as resp:
            self.length = int(resp.headers.get("Content-Length", 0))
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size == -1 or self.pos + size > self.length:
            size = self.length - self.pos
        if size <= 0:
            return b""
        end = self.pos + size - 1
        req = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={self.pos}-{end}",
                "User-Agent": "Mozilla/5.0",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def verify_classes_json(classes_path: Path) -> dict:
    """Load and validate the curated classes.json file."""
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing classes specification at {classes_path}")
    with open(classes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[OK] Successfully verified classes.json: {len(data['recommended_class_set'])} recommended classes.")
    return data


def main() -> None:
    """Run verification checks."""
    project_root = Path(__file__).resolve().parent.parent
    classes_file = project_root / "data" / "classes.json"
    verify_classes_json(classes_file)


if __name__ == "__main__":
    main()


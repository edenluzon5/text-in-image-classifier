"""Download the assignment images from the public Google Drive folder.

Usage:
    python download_data.py --out data/images
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import gdown

CLASS_FOLDERS = {
    "alcohol": "13tidhAjj6gMEK1uL86a_LkcNvCNtsukd",
    "non_alcohol": "1pRn5lMI2pAqUcQA3oBbFDVk8yK6XDdkd",
}


def _download_one(file_id: str, dest: str) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return dest
    for _ in range(3):
        gdown.download(id=file_id, output=dest, quiet=True)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            return dest
    raise RuntimeError(f"failed to download {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/images")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    jobs = []
    for label, folder_id in CLASS_FOLDERS.items():
        out_dir = os.path.join(args.out, label)
        os.makedirs(out_dir, exist_ok=True)
        files = gdown.download_folder(id=folder_id, skip_download=True, quiet=True)
        print(f"{label}: {len(files)} files")
        for f in files:
            jobs.append((f.id, os.path.join(out_dir, os.path.basename(f.path))))

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_one, fid, dest): dest for fid, dest in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as exc:
                failed.append((futures[fut], str(exc)))
            if i % 100 == 0:
                print(f"{i}/{len(jobs)}")
    print(f"done: {len(jobs) - len(failed)} ok, {len(failed)} failed")
    for dest, err in failed:
        print("FAILED", dest, err)


if __name__ == "__main__":
    main()

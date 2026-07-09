#!/usr/bin/env python3
"""
scripts/download_data.py
Script to download and extract datasets for the Intrinsic Image Decomposition project.
Datasets:
1. MIT Intrinsic Images (Grosse et al. 2009) - Data and Code
2. MPI-Sintel (Butler et al. 2012) - Training Images (Albedo & Clean)
"""

import os
import sys
import urllib.request
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm

MIT_DATA_URL = "http://people.csail.mit.edu/rgrosse/intrinsic/intrinsic-data.tar.gz"
MIT_CODE_URL = "http://people.csail.mit.edu/rgrosse/intrinsic/intrinsic-code-python.tar.gz"
SINTEL_URL = "http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url: str, output_path: Path):
    print(f"Downloading {url} to {output_path}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_archive(archive_path: Path, extract_dir: Path):
    print(f"Extracting {archive_path} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.suffixes[-2:] == ['.tar', '.gz'] or archive_path.suffix == '.tgz':
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    # 1. MIT Intrinsic Data
    mit_data_zip = data_dir / "intrinsic-data.tar.gz"
    mit_dir = data_dir / "mit"
    if not mit_dir.exists():
        if not mit_data_zip.exists():
            try:
                download_url(MIT_DATA_URL, mit_data_zip)
            except Exception as e:
                print(f"Error downloading MIT data: {e}", file=sys.stderr)
                sys.exit(1)
        try:
            extract_archive(mit_data_zip, mit_dir)
        except Exception as e:
            print(f"Error extracting MIT data: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("MIT Intrinsic data already exists.")

    # 2. MIT Intrinsic Code (Python)
    mit_code_zip = data_dir / "intrinsic-code-python.tar.gz"
    mit_code_dir = data_dir / "mit_code"
    if not mit_code_dir.exists():
        if not mit_code_zip.exists():
            try:
                download_url(MIT_CODE_URL, mit_code_zip)
            except Exception as e:
                print(f"Error downloading MIT code: {e}", file=sys.stderr)
        if mit_code_zip.exists():
            try:
                extract_archive(mit_code_zip, mit_code_dir)
            except Exception as e:
                print(f"Error extracting MIT code: {e}", file=sys.stderr)
    else:
        print("MIT Intrinsic code already exists.")

    # 3. MPI-Sintel Data
    sintel_zip = data_dir / "MPI-Sintel-training_images.zip"
    sintel_dir = data_dir / "sintel"
    if not sintel_dir.exists():
        if not sintel_zip.exists():
            try:
                download_url(SINTEL_URL, sintel_zip)
            except Exception as e:
                print(f"Error downloading Sintel data: {e}", file=sys.stderr)
                print("Note: If the Sintel download failed or is too slow, please check internet connection or manually place files.", file=sys.stderr)
                sys.exit(1)
        try:
            extract_archive(sintel_zip, sintel_dir)
        except Exception as e:
            print(f"Error extracting Sintel data: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("MPI-Sintel training images already exist.")

    print("Setup completed successfully.")

if __name__ == "__main__":
    main()

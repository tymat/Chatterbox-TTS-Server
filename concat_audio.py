#!/usr/bin/env python3
"""
Concatenate WAV files from a folder into a single WAV file using ffmpeg.
Files are joined in sorted filename order with no re-encoding.

Usage:
    python concat_audio.py <input_folder> [output_file]

Examples:
    python concat_audio.py outputs/batch_20260403_094710_29648494
    python concat_audio.py outputs/batch_20260403_094710_29648494 my_audiobook.wav
    python concat_audio.py outputs/batch_20260403_094710_29648494 output.wav --pattern "*.mp3"
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def concat_audio(input_folder: str, output_file: str = None, pattern: str = "*.wav"):
    folder = Path(input_folder)
    if not folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    files = sorted(folder.glob(pattern))
    if not files:
        print(f"Error: No {pattern} files found in '{input_folder}'.", file=sys.stderr)
        sys.exit(1)

    if output_file is None:
        output_file = str(folder / f"combined{Path(pattern).suffix}")

    print(f"Found {len(files)} files to concatenate:")
    for f in files:
        print(f"  {f.name}")

    # Create ffmpeg concat list file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as listfile:
        for f in files:
            # ffmpeg concat demuxer requires escaped single quotes in paths
            escaped = str(f.resolve()).replace("'", "'\\''")
            listfile.write(f"file '{escaped}'\n")
        listfile_path = listfile.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", listfile_path,
            "-c", "copy",
            output_file,
        ]
        print(f"\nConcatenating to: {output_file}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        output_path = Path(output_file)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Done. Output: {output_file} ({size_mb:.1f} MB)")
    finally:
        Path(listfile_path).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate audio files from a folder into a single file using ffmpeg (no re-encoding)."
    )
    parser.add_argument("input_folder", help="Folder containing audio files to concatenate.")
    parser.add_argument("output_file", nargs="?", default=None, help="Output file path. Defaults to <input_folder>/combined.wav")
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern for audio files (default: *.wav)")
    args = parser.parse_args()
    concat_audio(args.input_folder, args.output_file, args.pattern)

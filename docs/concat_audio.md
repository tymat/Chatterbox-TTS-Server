# concat_audio.py

Concatenate audio files from a folder into a single file using ffmpeg. Files are joined in sorted filename order with no re-encoding.

## Requirements

- Python 3
- ffmpeg (`brew install ffmpeg`)

## Usage

```bash
python concat_audio.py <input_folder> [output_file] [--pattern PATTERN]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `input_folder` | Yes | Folder containing audio files to concatenate |
| `output_file` | No | Output file path. Defaults to `<input_folder>/combined.wav` |
| `--pattern` | No | Glob pattern for matching files. Default: `*.wav` |

## Examples

### Concatenate WAV files from a batch output folder

```bash
python concat_audio.py outputs/batch_20260403_094710_29648494
```

Produces `outputs/batch_20260403_094710_29648494/combined.wav`

### Specify a custom output path

```bash
python concat_audio.py outputs/batch_20260403_094710_29648494 my_audiobook.wav
```

### Concatenate MP3 files instead of WAV

```bash
python concat_audio.py outputs/batch_20260403_094710_29648494 audiobook.mp3 --pattern "*.mp3"
```

## How it works

1. Scans the input folder for files matching the pattern (default `*.wav`)
2. Sorts them by filename (so `000.wav`, `001.wav`, `002.wav` are in order)
3. Uses ffmpeg's concat demuxer to join them without re-encoding (`-c copy`)
4. Outputs a single combined file

No audio quality is lost since ffmpeg copies the audio streams directly.

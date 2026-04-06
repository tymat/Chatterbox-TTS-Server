# Mobile TTS Voice Cloner — Context & Requirements

## Purpose of This Document

This document provides a new Claude session with full context to spec and build a native iOS TTS voice cloning app. It describes the existing desktop server application's functionality, the mobile-specific requirements, and the technical landscape for on-device TTS inference.

---

## Part 1: Existing Desktop Application

### What It Is

A self-hosted TTS (text-to-speech) server with a web UI for generating audiobooks with voice cloning. Runs on Apple Silicon via MLX framework. The primary TTS model is **Qwen3-TTS 0.6B/1.7B** (MLX, voice cloning, 10 languages) with fallback support for Chatterbox models (PyTorch).

### Core Workflow

1. **Input text** — paste or type text to convert to speech
2. **Select voice** — choose a predefined voice OR upload reference audio for voice cloning (3-30 seconds of audio + transcript)
3. **Configure** — set language, temperature, speed, and other generation parameters
4. **Generate** — produces audio, playable in-browser with download option
5. **Batch mode** — split long text into sections, generate each independently with per-section voice/language control

### Key Features

#### Single Generation Mode
- Paste text, select voice (predefined or clone), click Generate
- Parameters: temperature (0.0-1.5), speed (0.25-4.0), language (10 supported)
- Pause markup: `[pause:1.5s]` tags for custom silence insertion
- Text automatically split into sentence-sized chunks for reliable generation
- Output formats: WAV, MP3, Opus
- Output sample rate configurable (default 44100 Hz, engine outputs 24000 Hz)

#### Batch/Chapter Mode
- Split text into sections by empty lines
- Each section is an editable text box (collapsible)
- Per-section controls: voice sample selector, language selector
- "Generate All" processes sections sequentially
- Per-section: audio player, download, redo (regenerate), reload audio buttons
- Progress bar with real-time status per section
- Sequential playback: Play All button chains sections one after another
- Download All as ZIP

#### Multi-Sample Voice Cloning
- Upload multiple reference audio files (each with its own transcript)
- Selection mode per project: "Random" (varies per section) or pinned to a specific sample
- Per-section override: each section can use a different sample from the pool
- Purpose: variety in voice characteristics across long audiobooks
- Voice samples stored as: `{audio_filename, transcript}` pairs
- Files live in `reference_audio/` directory

#### Project Save/Load
- Projects saved as `project.json` in output directory
- Auto-saves after each section generates, after regeneration, and on text edits (debounced)
- Atomic writes (temp file + rename) to prevent corruption
- Project includes: all section texts, voice samples, generation params, per-section voice/language overrides, audio filenames
- Load Project: lists saved projects, click to restore full state (texts, audio players, voice config)
- Project naming: user can name projects (folder renamed on save)

#### Pause Tag System
- Format: `[pause:Xs]` where X is seconds (e.g., `[pause:0.8s]`, `[pause:2s]`)
- Parsed server-side: text split at tags, silence generated as zero-valued audio samples
- Works with any TTS model (not model-dependent)
- User configurable: "Pause between sections" input auto-appends pause tags when splitting

#### Audio Processing Pipeline
1. Text split into chunks (sentence boundaries, respects CJK punctuation)
2. Each chunk synthesized independently via TTS model
3. 80ms silence padding appended to each chunk (prevents crossfade from clipping last word)
4. Chunks stitched with equal-power crossfade (20ms overlap)
5. Inter-chunk silence: 300ms (configurable)
6. 200ms silence tail appended to final audio
7. Peak normalization (prevent clipping)
8. Optional: speed factor applied via librosa time-stretch
9. Resampling to target sample rate (e.g., 24000→44100)
10. Encoding to output format (WAV/MP3/Opus)

#### CJK (Chinese/Japanese/Korean) Support
- Sentence splitting on full-width punctuation: 。！？
- Character detection for CJK Unicode ranges
- Proper handling of mixed CJK/Latin text

#### Configuration
- YAML-based config file (`config.yaml`)
- Generation defaults: temperature, exaggeration, cfg_weight, seed, speed_factor, language
- Audio output: format, sample_rate, max_reference_duration_sec
- UI state persistence: last text, voice mode, reference file, theme, etc.
- All settings editable via web UI

#### Audio Concatenation Tool
- `concat_audio.py` — standalone script using ffmpeg
- Joins section audio files into single audiobook file (no re-encoding)
- Usage: `python concat_audio.py <folder> [output_file] [--pattern "*.mp3"]`

### Technical Architecture

- **Server**: Python FastAPI, uvicorn ASGI server
- **TTS Engine**: Qwen3-TTS via mlx-audio (MLX framework, Apple Silicon native)
- **Background Processing**: Python threading for batch generation
- **State**: In-memory dict with thread-safe locks, persisted to project.json
- **Audio**: numpy arrays, soundfile for encoding, librosa for resampling/time-stretch
- **UI**: Vanilla HTML/JS/CSS, WaveSurfer.js for audio visualization

### Data Models

```
VoiceSample: {audio_filename: str, transcript: str}

BatchTTSRequest: {
  text, project_name, separator,
  voice_mode (predefined|clone),
  predefined_voice_id, reference_audio_filename,
  output_format (wav|opus|mp3),
  split_text, chunk_size,
  temperature, exaggeration, cfg_weight, seed, speed_factor, language,
  qwen3_speaker, qwen3_instruct, qwen3_ref_text,
  voice_samples: [VoiceSample],
  sample_selection_mode (random|0|1|2...),
  section_sample_overrides: [int|null],
  section_language_overrides: [str|null]
}

Project JSON: {
  version: 1,
  batch_id, project_name,
  created_at, updated_at,
  status (queued|generating|completed|partial|failed),
  total_chapters, completed_chapters,
  voice_config: {voice_mode, voice_samples, sample_selection_mode, predefined_voice_id, reference_audio_filename},
  generation_params: {output_format, split_text, chunk_size, temperature, ...},
  chapters: [{index, title, text, status, filename, voice_sample_used, voice_sample_override, language, error}]
}
```

### Qwen3-TTS Model Details

- **Architecture**: Discrete multi-codebook Language Model, 12Hz token rate
- **Sizes**: 0.6B (Base/Custom/VoiceDesign) and 1.7B variants
- **Voice Cloning**: In-Context Learning (ICL) — reference audio encoded into codec tokens, prepended to generation
- **Inputs**: text, language code, ref_audio path, ref_text transcript
- **Output**: audio numpy array at 24000 Hz
- **MLX format**: 8-bit quantized weights from mlx-community on HuggingFace
- **RAM**: ~2-3GB for 0.6B 8-bit, ~4-6GB for 1.7B 8-bit
- **Key parameters**: temperature (0.7-0.9), repetition_penalty (1.05 default, 1.2 for ICL), top_k (50), top_p (1.0), speed (0.8-1.3)
- **Sentence splitting**: `split_pattern=r"(?<=[.!?])\s+"` for reliable generation of long text
- **Known issue**: Occasional word cutoff at end of generation — mitigated by sentence-level splitting and tail padding

---

## Part 2: Mobile App Requirements

### Platform & Architecture

- **Platform**: iOS (Swift, native UI)
- **Minimum iOS version**: iOS 17+ (for MLX/CoreML capabilities)
- **Target devices**: iPhone 12+ (4GB+ RAM for voice cloning), iPad
- **Architecture**: 100% offline, no server required
- **Model download**: On first launch (not bundled in app binary)
- **Framework**: SwiftUI for UI, MLX Swift or CoreML for inference

### TTS Model for Mobile

**Primary**: Qwen3-TTS 0.6B (8-bit quantized)
- CoreML conversion exists: [FluidInference/qwen3-tts-coreml](https://huggingface.co/FluidInference/qwen3-tts-coreml) (W8A16)
- MLX Swift SDK exists: [swift-qwen3-tts](https://github.com/AtomGradient/swift-qwen3-tts)
- Also available: [mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift) — Swift SDK for TTS via MLX
- Voice cloning: Yes, from 3s reference audio
- Languages: 10 (EN, ZH, JA, KO, DE, FR, RU, PT, ES, IT)
- Size: ~1.2GB download (W8A16 quantized)
- RAM: ~2-3GB at inference

**Fallback option**: Kokoro-82M (no voice cloning, but only 80-160MB, excellent quality, 10 languages)

### Core Features (Mobile)

#### 1. Text Input
- Standard text editor for typing/pasting
- **Speech-to-text input**: tap a mic button to dictate text (use iOS built-in Speech framework or on-device Whisper)
- Character/word count display
- Pause tag support: `[pause:Xs]` — same syntax as desktop
- Auto-insert pause between paragraphs (configurable duration)

#### 2. Voice Cloning
- Record reference audio directly in-app (3-30 seconds)
- Or import from Files app / voice memos
- Transcript input per sample (text field + speech-to-text option)
- Multiple voice samples with per-section assignment (same as desktop)
- Voice sample library persisted on device

#### 3. Batch/Section Mode
- Split text by paragraphs (empty lines)
- Collapsible section cards (same UX as desktop)
- Per-section: voice sample selector, language selector
- Generate All: sequential processing with progress
- Per-section: play, redo, download
- Sequential playback (Play All)

#### 4. Generation
- On-device inference via MLX Swift or CoreML
- Background processing (not blocking UI)
- Progress per section
- Cancel support (stop mid-generation)
- Parameters: temperature, speed, language

#### 5. Project Management
- Save/load projects (same JSON format as desktop for interop potential)
- Project naming
- Project list with status, chapter counts, dates
- Export project as ZIP (audio files + project.json)
- Share individual audio files via iOS share sheet

#### 6. Audio Playback
- Native iOS audio player per section
- Sequential playback across sections
- Background audio playback (plays when app is backgrounded)
- AirPlay / Bluetooth support (comes free with AVFoundation)

#### 7. Export
- Export full audiobook: concatenate sections into single file
- Export individual sections
- Share via iOS share sheet (AirDrop, Files, etc.)
- Output formats: WAV, M4A (AAC — better iOS native support than MP3/Opus)

### Mobile-Specific Considerations

#### Model Management
- Download model on first launch (~1.2GB for Qwen3 0.6B)
- Show download progress
- Store in app's documents directory
- Option to delete model to free space
- Check for model updates

#### Memory Management
- iPhone 12: 4GB RAM — tight for 0.6B model + app
- iPhone 15 Pro: 8GB RAM — comfortable
- Strategy: generate one section at a time, free audio arrays after encoding to disk
- Monitor memory pressure, pause generation if needed

#### Storage
- Audio files: ~1-5MB per section (WAV at 24kHz)
- A 60-section audiobook ≈ 60-300MB
- Projects stored in app documents directory
- iCloud sync potential (future)

#### Battery
- MLX inference is GPU-intensive
- Warn user about battery consumption for long batches
- Respect Low Power Mode

#### Speech-to-Text for Input
- iOS Speech framework (SFSpeechRecognizer) — on-device, no network
- Or bundle a small Whisper model for better accuracy
- Use for: text input, voice sample transcripts
- Language detection for automatic language setting

### UI Design Principles

- **Native iOS feel**: SwiftUI components, SF Symbols, standard navigation patterns
- **Single-screen focused**: avoid deep navigation hierarchies
- **Thumb-friendly**: important controls reachable with one hand
- **Dark mode support**: follow system setting
- **Accessibility**: VoiceOver support, Dynamic Type
- **Haptic feedback**: on generation complete, errors

### Suggested Screen Layout

```
[Projects List]
  → [Project Editor]
       ├── Text Input (with mic button for speech-to-text)
       ├── Voice Samples (record/import, transcripts)
       ├── Settings (temperature, speed, language)
       ├── Sections List
       │    ├── Section 1 [collapsed: preview | expanded: editor + voice/lang picker]
       │    │    └── [Player] [Redo] [Share]
       │    ├── Section 2 ...
       │    └── ...
       ├── [Generate All] [Play All] [Export]
       └── Progress Bar
```

---

## Part 3: Technical Resources for Mobile Implementation

### Existing Mobile TTS Libraries & SDKs

| Library | What It Does | Link |
|---------|-------------|------|
| **mlx-audio-swift** | Swift SDK for TTS via MLX (Qwen3, Orpheus, Pocket TTS) | [GitHub](https://github.com/Blaizzy/mlx-audio-swift) |
| **swift-qwen3-tts** | Dedicated Qwen3-TTS Swift wrapper via MLX | [GitHub](https://github.com/AtomGradient/swift-qwen3-tts) |
| **FluidAudio** | CoreML TTS/STT in Swift (Kokoro, Qwen3-TTS) | [GitHub](https://github.com/FluidInference/FluidAudio) |
| **qwen3-tts-coreml** | Pre-converted CoreML models (W8A16 quantized) | [HuggingFace](https://huggingface.co/FluidInference/qwen3-tts-coreml) |
| **sherpa-onnx** | ONNX Runtime mobile SDK (iOS/Android/React Native) | [GitHub](https://github.com/k2-fsa/sherpa-onnx) |
| **Kokoro CoreML** | Kokoro-82M for CoreML (no voice cloning) | [HuggingFace](https://huggingface.co/FluidInference/kokoro-82m-coreml) |

### Mobile Inference Benchmarks (from FluidAudio)

Kokoro-82M on CoreML (Apple Silicon):
- Short text: RTFx 28.6x (generates 28x faster than real-time)
- Long text: RTFx 3.7x
- Peak RAM: 1.5GB
- Latency: 0.3-3.0s for varying text lengths

Qwen3-TTS 0.6B (estimated for mobile):
- First-token latency: ~97ms (server benchmark)
- 10 seconds of audio: ~2-5 seconds inference on modern iPhone
- Peak RAM: ~2-3GB (W8A16 quantized)

### Key HuggingFace Models

| Model | Size | Voice Cloning | Languages | HF Link |
|-------|------|--------------|-----------|---------|
| Qwen3-TTS-0.6B-Base-8bit (MLX) | ~1.2GB | Yes | 10 | mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit |
| Qwen3-TTS-0.6B-CustomVoice-8bit (MLX) | ~1.2GB | Preset speakers | 10 | mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit |
| Qwen3-TTS-0.6B-Base CoreML (W8A16) | ~1.2GB | Yes | 10 | FluidInference/qwen3-tts-coreml |
| Kokoro-82M CoreML | ~160MB | No (54 voices) | 10 | FluidInference/kokoro-82m-coreml |

### iOS Frameworks to Use

| Purpose | Framework |
|---------|-----------|
| UI | SwiftUI |
| TTS Inference | MLX Swift (via mlx-audio-swift) or CoreML |
| Audio Playback | AVFoundation (AVAudioPlayer / AVQueuePlayer) |
| Audio Recording | AVAudioRecorder |
| Speech-to-Text | Speech framework (SFSpeechRecognizer) |
| File Management | FileManager, UIDocumentPickerViewController |
| Sharing | UIActivityViewController |
| Model Download | URLSession with background download |
| JSON Persistence | Codable + JSONEncoder/Decoder |
| Background Tasks | BGTaskScheduler (for long generation) |

---

## Part 4: Interoperability

### Desktop ↔ Mobile Project Format

The mobile app should use the same `project.json` format as the desktop app. This enables:
- Generate on desktop, transfer to mobile for playback/editing
- Start a project on mobile, continue on desktop
- Share projects via AirDrop or Files

### Shared Reference Audio

Voice sample files (.wav/.mp3) can be shared between desktop and mobile via:
- AirDrop
- iCloud Drive
- Direct file transfer

The mobile app should store voice samples in its own documents directory but support importing from Files app.

---

## Part 5: Development Phases

### Phase 1: Minimal Viable Product
- Single text input → generate speech with one voice sample
- On-device Qwen3-TTS 0.6B inference
- Basic audio playback
- Save/export audio file
- Model download on first launch

### Phase 2: Batch Mode
- Split text into sections
- Generate All with progress
- Per-section playback
- Per-section redo
- Project save/load

### Phase 3: Multi-Sample & Polish
- Multiple voice samples with per-section assignment
- Per-section language selection
- Speech-to-text input
- Sequential playback (Play All)
- Export as concatenated audiobook
- Project sharing

### Phase 4: Advanced
- iCloud sync for projects
- Background generation
- Siri Shortcuts integration
- Widget for quick generation
- Apple Watch companion (trigger generation, monitor progress)

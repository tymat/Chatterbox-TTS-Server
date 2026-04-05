# Feature Spec: Multi-Sample Voice Cloning for Batch Generation

## Problem

Voice cloning quality varies because a single 30-second reference audio can't capture the full range of a speaker's voice. Different samples capture different intonations, pacing, and emotional tones. Using a single sample for an entire audiobook produces monotonous output and occasional artifacts where the model struggles to match the reference.

## Solution

Allow users to upload multiple voice samples (each with its own transcript), and randomly select from them during batch generation. This produces more natural-sounding audiobooks because the model draws from a richer set of vocal characteristics.

## User Flow

### 1. Upload & Configure Voice Samples

In the batch mode UI, a new "Voice Samples" panel replaces the single reference audio selector:

```
+--------------------------------------------------+
| Voice Samples for Cloning                        |
|                                                  |
| Sample 1: [ACE_sample1.wav ▼] [Remove]          |
| Transcript: [Hello, this is a sample of my...]   |
|                                                  |
| Sample 2: [ACE_sample2.wav ▼] [Remove]          |
| Transcript: [The quick brown fox jumps over...]  |
|                                                  |
| Sample 3: [ACE_sample3.wav ▼] [Remove]          |
| Transcript: [In this recording I demonstrate...] |
|                                                  |
| [+ Add Sample]                                   |
|                                                  |
| Selection Mode: ( ) Random  ( ) Sample 1         |
|                 ( ) Sample 2  ( ) Sample 3       |
+--------------------------------------------------+
```

- Each sample has a dropdown (existing reference audio files) and a transcript textarea
- Users can add/remove samples dynamically
- Minimum 1 sample, no hard maximum (practical limit ~5-10)
- Each sample's audio must be under 30 seconds (existing validation)
- Selection mode: "Random" picks a different sample per section, or user can pin a specific sample

### 2. Batch Generation

When "Generate All" is clicked:
- **Random mode**: For each section, randomly select one of the configured samples. The selected sample index is recorded per-section in the project file for reproducibility.
- **Pinned mode**: All sections use the selected sample.
- The selected sample's audio path and transcript are passed to `engine.synthesize()` for that section.

### 3. Section-Level Override

After generation, each section's result area shows which sample was used (e.g., "Generated with Sample 2"). When clicking "Redo", the user can optionally pick a different sample for that specific section.

### 4. Project Persistence

The project.json file stores:
- All voice samples (filenames + transcripts)
- The selection mode ("random" or sample index)
- Per-section: which sample was used for generation

## Data Model Changes

### Request Models (models.py)

```python
class VoiceSample(BaseModel):
    audio_filename: str          # Reference audio file in reference_audio/
    transcript: str              # What is spoken in the audio

class BatchTTSRequest(BaseModel):
    # ... existing fields ...
    voice_samples: Optional[List[VoiceSample]] = None   # Multiple samples
    sample_selection_mode: str = "random"                # "random" or "0", "1", "2" (sample index)
```

### Project File (project.json)

```json
{
  "version": 2,
  "voice_config": {
    "voice_mode": "clone",
    "voice_samples": [
      {
        "audio_filename": "ACE_sample1.wav",
        "transcript": "Hello, this is a sample of my voice..."
      },
      {
        "audio_filename": "ACE_sample2.wav",
        "transcript": "The quick brown fox jumps over..."
      }
    ],
    "sample_selection_mode": "random"
  },
  "chapters": [
    {
      "index": 0,
      "text": "...",
      "status": "completed",
      "filename": "000.wav",
      "voice_sample_used": 1,
      "error": null
    }
  ]
}
```

### Backward Compatibility

- `version: 1` projects with single `reference_audio_filename` + `qwen3_ref_text` continue to load normally
- On load, single-sample projects are internally represented as a 1-element `voice_samples` array
- The old `reference_audio_filename` and `qwen3_ref_text` fields remain supported as fallback

## Server Changes (server.py)

### Batch Start Endpoint

```
POST /batch/start
```

- If `voice_samples` is provided, resolve each audio path and validate
- Store all samples in the job dict and voice_config
- If `voice_samples` is not provided, fall back to existing single-sample behavior

### Batch Worker

In `_batch_worker()`, for each chapter:

1. Determine which sample to use:
   - If `sample_selection_mode == "random"`: `random.choice(range(len(voice_samples)))`
   - If `sample_selection_mode` is a digit: use that index
2. Set `audio_prompt_path` and `ref_text` for this chapter from the selected sample
3. Record `voice_sample_used: int` on the chapter dict
4. Pass to `_generate_chapter_audio()` as before (single path + single transcript)

### Regeneration Endpoint

```
POST /batch/regenerate/{batch_id}/{chapter_index}
```

- Accept optional `voice_sample_index: int` in the request body
- If provided, use that sample; otherwise re-roll random or use the original

### Project Save/Load

- `_save_project_file()`: Include `voice_samples` array and per-chapter `voice_sample_used`
- `_load_project_to_job()`: Reconstruct voice_samples, validate all audio files exist

## Engine Changes (engine.py, engine_qwen3.py)

No changes needed. The engine already accepts a single `audio_prompt_path` + `ref_text` per call. The multi-sample selection happens at the batch worker level, not the engine level. Each `engine.synthesize()` call still receives one sample.

## UI Changes (index.html, script.js)

### HTML

New "Voice Samples" panel inside `batch-controls`, shown when voice_mode is "clone":

- Dynamic list of sample rows (audio dropdown + transcript textarea + remove button)
- "Add Sample" button
- Radio buttons for selection mode: Random / Sample 1 / Sample 2 / ...
- Sample rows update radio button labels dynamically

### JavaScript

- `batchVoiceSamples: Array<{audio_filename: string, transcript: string}>` state
- `addVoiceSample()` / `removeVoiceSample(index)` functions
- `renderVoiceSamplesPanel()` to build the dynamic UI
- Update `submitBatchFromSections()` to include `voice_samples` and `sample_selection_mode` in the request
- Update `updateBatchSectionsFromStatus()` to show which sample was used per section
- Update `_batchRegenerate()` to optionally accept a sample index
- Save/restore voice samples in UI state persistence

### Load Project

When loading a project with voice_samples:
- Populate the voice samples panel with saved samples
- Set the selection mode radio
- Per-section results show which sample was used

## File Storage

Voice sample audio files remain in the global `reference_audio/` directory (not copied into project folders). The project.json stores filenames, not paths. On load, filenames are resolved against `reference_audio/`.

If a referenced audio file is missing on load, a warning is shown but the project still loads (existing audio plays fine, regeneration is blocked for that sample).

## Implementation Order

1. **Models**: Add `VoiceSample` model, update `BatchTTSRequest` with `voice_samples` and `sample_selection_mode`
2. **Server - batch_start**: Handle `voice_samples` array, resolve and validate all audio paths
3. **Server - batch_worker**: Sample selection logic per chapter, record `voice_sample_used`
4. **Server - regenerate**: Accept optional `voice_sample_index`
5. **Server - save/load**: Persist voice_samples in project.json, restore on load
6. **UI - HTML**: Voice samples panel with dynamic add/remove
7. **UI - JS**: State management, form submission, status display

## Edge Cases

- **All samples deleted after generation**: Project loads with warnings, audio plays, Redo blocked
- **Duplicate samples**: Allowed (same file with different transcripts is valid)
- **Empty transcript**: Allowed for Qwen3 (auto-transcribes via Whisper, but slower)
- **Mixed formats**: .wav and .mp3 samples can be mixed
- **Zero samples in random mode**: Error at batch_start validation
- **Sample index out of range**: Clamp to valid range with warning

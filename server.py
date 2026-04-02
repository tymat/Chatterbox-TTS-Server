# File: server.py
# Main FastAPI application for the TTS Server.
# Handles API requests for text-to-speech generation, UI serving,
# configuration management, and file uploads.

import os
import io
import logging
import logging.handlers  # For RotatingFileHandler
import shutil
import time
import uuid
import yaml  # For loading presets
import numpy as np
import librosa  # For potential direct use if needed, though utils.py handles most
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Literal, Tuple
import webbrowser  # For automatic browser opening
import threading  # For automatic browser opening

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    File,
    UploadFile,
    Form,
    BackgroundTasks,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
    FileResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# --- Internal Project Imports ---
from config import (
    config_manager,
    get_host,
    get_port,
    get_log_file_path,
    get_output_path,
    get_reference_audio_path,
    get_predefined_voices_path,
    get_ui_title,
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
    get_gen_default_speed_factor,
    get_gen_default_language,
    get_audio_sample_rate,
    get_full_config_for_template,
    get_audio_output_format,
)

import engine  # TTS Engine interface
from models import (  # Pydantic models
    CustomTTSRequest,
    BatchTTSRequest,
    BatchStatusResponse,
    BatchChapterStatus,
    ErrorResponse,
    UpdateStatusResponse,
)
import utils  # Utility functions

from pydantic import BaseModel, Field


class OpenAISpeechRequest(BaseModel):
    model: str
    input_: str = Field(..., alias="input")
    voice: str
    response_format: Literal["wav", "opus", "mp3"] = "wav"  # Add "mp3"
    speed: float = 1.0
    seed: Optional[int] = None


# --- Logging Configuration ---
log_file_path_obj = get_log_file_path()
log_file_max_size_mb = config_manager.get_int("server.log_file_max_size_mb", 10)
log_backup_count = config_manager.get_int("server.log_file_backup_count", 5)

log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.handlers.RotatingFileHandler(
            str(log_file_path_obj),
            maxBytes=log_file_max_size_mb * 1024 * 1024,
            backupCount=log_backup_count,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Global Variables & Application Setup ---
startup_complete_event = threading.Event()  # For coordinating browser opening


def _delayed_browser_open(host: str, port: int):
    """
    Waits for the startup_complete_event, then opens the web browser
    to the server's main page after a short delay.
    """
    try:
        startup_complete_event.wait(timeout=30)
        if not startup_complete_event.is_set():
            logger.warning(
                "Server startup did not signal completion within timeout. Browser will not be opened automatically."
            )
            return

        time.sleep(1.5)
        display_host = "localhost" if host == "0.0.0.0" else host
        browser_url = f"http://{display_host}:{port}/"
        logger.info(f"Attempting to open web browser to: {browser_url}")
        webbrowser.open(browser_url)
    except Exception as e:
        logger.error(f"Failed to open browser automatically: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    logger.info("TTS Server: Initializing application...")
    try:
        logger.info(f"Configuration loaded. Log file at: {get_log_file_path()}")

        paths_to_ensure = [
            get_output_path(),
            get_reference_audio_path(),
            get_predefined_voices_path(),
            Path("ui"),
            config_manager.get_path(
                "paths.model_cache", "./model_cache", ensure_absolute=True
            ),
        ]
        for p in paths_to_ensure:
            p.mkdir(parents=True, exist_ok=True)

        if not engine.load_model():
            logger.critical(
                "CRITICAL: TTS Model failed to load on startup. Server might not function correctly."
            )
        else:
            logger.info("TTS Model loaded successfully via engine.")
            host_address = get_host()
            server_port = get_port()
            browser_thread = threading.Thread(
                target=lambda: _delayed_browser_open(host_address, server_port),
                daemon=True,
            )
            browser_thread.start()

        logger.info("Application startup sequence complete.")
        startup_complete_event.set()
        yield
    except Exception as e_startup:
        logger.error(
            f"FATAL ERROR during application startup: {e_startup}", exc_info=True
        )
        startup_complete_event.set()
        yield
    finally:
        logger.info("TTS Server: Application shutdown sequence initiated...")
        logger.info("TTS Server: Application shutdown complete.")


# --- FastAPI Application Instance ---
app = FastAPI(
    title=get_ui_title(),
    description="Text-to-Speech server with advanced UI and API capabilities.",
    version="2.0.2",  # Version Bump
    lifespan=lifespan,
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Static Files and HTML Templates ---
ui_static_path = Path(__file__).parent / "ui"
if ui_static_path.is_dir():
    app.mount("/ui", StaticFiles(directory=ui_static_path), name="ui_static_assets")
else:
    logger.warning(
        f"UI static assets directory not found at '{ui_static_path}'. UI may not load correctly."
    )

# This will serve files from 'ui_static_path/vendor' when requests come to '/vendor/*'
if (ui_static_path / "vendor").is_dir():
    app.mount(
        "/vendor", StaticFiles(directory=ui_static_path / "vendor"), name="vendor_files"
    )
else:
    logger.warning(
        f"Vendor directory not found at '{ui_static_path}' /vendor. Wavesurfer might not load."
    )


@app.get("/styles.css", include_in_schema=False)
async def get_main_styles():
    styles_file = ui_static_path / "styles.css"
    if styles_file.is_file():
        return FileResponse(styles_file)
    raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/script.js", include_in_schema=False)
async def get_main_script():
    script_file = ui_static_path / "script.js"
    if script_file.is_file():
        return FileResponse(script_file)
    raise HTTPException(status_code=404, detail="script.js not found")


outputs_static_path = get_output_path(ensure_absolute=True)
try:
    app.mount(
        "/outputs",
        StaticFiles(directory=str(outputs_static_path)),
        name="generated_outputs",
    )
except RuntimeError as e_mount_outputs:
    logger.error(
        f"Failed to mount /outputs directory '{outputs_static_path}': {e_mount_outputs}. "
        "Output files may not be accessible via URL."
    )

templates = Jinja2Templates(directory=str(ui_static_path))

# --- API Endpoints ---

# --- Audio Stitching Helper Functions ---
# These functions support smart audio chunk concatenation with crossfading


def _generate_equal_power_curves(n_samples: int):
    """
    Generate equal-power crossfade curves using cos²/sin² functions.
    These curves maintain perceptually constant loudness during transitions.

    Args:
        n_samples: Number of samples in the fade region

    Returns:
        Tuple of (fade_out, fade_in) numpy arrays
    """
    t = np.linspace(0, np.pi / 2, n_samples, dtype=np.float32)
    fade_out = np.cos(t) ** 2  # 1 → 0
    fade_in = np.sin(t) ** 2  # 0 → 1
    return fade_out, fade_in


def _crossfade_with_overlap(
    chunk_a: np.ndarray, chunk_b: np.ndarray, fade_samples: int
) -> np.ndarray:
    """
    Perform true crossfade by overlapping and summing audio regions.

    This creates a seamless transition by:
    1. Taking the tail of chunk_a and head of chunk_b
    2. Applying equal-power fade curves
    3. Summing the overlapped regions

    Result length = len(chunk_a) + len(chunk_b) - fade_samples

    Args:
        chunk_a: First audio chunk (numpy float32 array)
        chunk_b: Second audio chunk (numpy float32 array)
        fade_samples: Number of samples to overlap

    Returns:
        Crossfaded audio as numpy float32 array
    """
    # Handle edge cases
    fade_samples = min(fade_samples, len(chunk_a), len(chunk_b))
    if fade_samples <= 0:
        return np.concatenate([chunk_a, chunk_b])

    fade_out, fade_in = _generate_equal_power_curves(fade_samples)

    # Extract overlap regions
    a_tail = chunk_a[-fade_samples:]
    b_head = chunk_b[:fade_samples]

    # Crossfade: weighted sum of overlapping regions
    crossfaded_region = (a_tail * fade_out) + (b_head * fade_in)

    # Assemble: [chunk_a without tail] + [crossfaded region] + [chunk_b without head]
    return np.concatenate(
        [chunk_a[:-fade_samples], crossfaded_region, chunk_b[fade_samples:]]
    )


def _apply_edge_fades(
    chunk: np.ndarray, fade_samples: int, fade_in: bool = True, fade_out: bool = True
) -> np.ndarray:
    """
    Apply minimal linear edge fades for click protection.

    This is used in fallback mode when full crossfading is disabled.
    Linear fades are acceptable for ultra-short safety fades (2-3ms).

    Args:
        chunk: Audio chunk (numpy array)
        fade_samples: Number of samples to fade
        fade_in: Whether to apply fade-in at start
        fade_out: Whether to apply fade-out at end

    Returns:
        Audio chunk with edge fades applied (numpy float32 array)
    """
    # Skip if chunk is too short for fading
    if len(chunk) < fade_samples * 2:
        return chunk.astype(np.float32, copy=False)

    result = chunk.astype(np.float32, copy=True)

    if fade_in:
        result[:fade_samples] *= np.linspace(0, 1, fade_samples, dtype=np.float32)
    if fade_out:
        result[-fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)

    return result


def _remove_dc_offset(
    audio: np.ndarray, sample_rate: int, cutoff_hz: float = 15.0
) -> np.ndarray:
    """
    Remove DC offset using a high-pass Butterworth filter.

    DC offset can cause low-frequency thumps when concatenating audio chunks.
    This applies a 2nd-order high-pass filter at the specified cutoff frequency.

    Args:
        audio: Audio data (numpy array)
        sample_rate: Sample rate in Hz
        cutoff_hz: High-pass filter cutoff frequency (default 15 Hz)

    Returns:
        Audio with DC offset removed (numpy float32 array)

    Note:
        Requires scipy. If scipy is not available, returns audio unchanged
        with a warning logged.
    """
    try:
        from scipy.signal import butter, filtfilt

        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_hz / nyquist

        # 2nd-order Butterworth high-pass filter
        b, a = butter(2, normalized_cutoff, btype="high")

        # Zero-phase filtering (no phase distortion)
        return filtfilt(b, a, audio).astype(np.float32)

    except ImportError:
        logger.warning(
            "scipy not available for DC offset removal. "
            "Install scipy to enable this feature: pip install scipy"
        )
        return audio.astype(np.float32, copy=False)
    except Exception as e:
        logger.error(f"DC offset removal failed: {e}")
        return audio.astype(np.float32, copy=False)


def _stitch_audio_segments(
    segments: List[np.ndarray],
    segment_is_pause: List[bool],
    sample_rate: int,
    sentence_pause_ms: int = 300,
    crossfade_ms: int = 20,
    safety_fade_ms: int = 3,
    enable_dc_removal: bool = False,
    dc_highpass_hz: int = 15,
) -> np.ndarray:
    """
    Stitches multiple audio segments into a single audio array with crossfades
    and silence insertion. Shared by /tts endpoint and batch worker.

    Args:
        segments: List of numpy audio arrays.
        segment_is_pause: List of bools indicating which segments are explicit pauses.
        sample_rate: Audio sample rate in Hz.
        sentence_pause_ms: Silence duration between non-pause segments.
        crossfade_ms: Crossfade duration for smart stitching.
        safety_fade_ms: Edge fade duration for fallback mode.
        enable_dc_removal: Whether to apply DC offset removal.
        dc_highpass_hz: Highpass cutoff for DC removal.

    Returns:
        Stitched audio as a numpy float32 array.
    """
    enable_smart_stitching = config_manager.get_bool(
        "audio_processing.enable_crossfade", True
    )
    peak_normalize_threshold = 0.99
    peak_normalize_target = 0.95

    if not segments:
        return np.array([], dtype=np.float32)

    if not sample_rate or sample_rate <= 0:
        logger.error(f"Invalid sample rate: {sample_rate}, falling back to raw concatenation")
        return np.concatenate(segments) if len(segments) > 1 else segments[0]

    if len(segments) == 1:
        final_audio = segments[0]
        logger.info("Single audio segment - no stitching required")

    elif enable_smart_stitching:
        fade_samples = int(crossfade_ms / 1000 * sample_rate)
        desired_silence_samples = int(sentence_pause_ms / 1000 * sample_rate)
        silence_buffer_samples = desired_silence_samples + (fade_samples * 2)

        # Pad each non-pause segment with a small silence tail so crossfade
        # doesn't eat into the last word of speech
        tail_pad_samples = int(0.08 * sample_rate)  # 80ms safety padding

        chunks = []
        for idx, chunk in enumerate(segments):
            processed = chunk.astype(np.float32, copy=True)
            if enable_dc_removal:
                processed = _remove_dc_offset(processed, sample_rate, dc_highpass_hz)
            is_pause = segment_is_pause[idx] if idx < len(segment_is_pause) else False
            if not is_pause and tail_pad_samples > 0:
                processed = np.concatenate([processed, np.zeros(tail_pad_samples, dtype=np.float32)])
            chunks.append(processed)

        result = chunks[0]
        for i in range(1, len(chunks)):
            prev_is_pause = segment_is_pause[i - 1] if i - 1 < len(segment_is_pause) else False
            curr_is_pause = segment_is_pause[i] if i < len(segment_is_pause) else False

            if prev_is_pause or curr_is_pause:
                result = np.concatenate([result, chunks[i]])
            else:
                silence = np.zeros(silence_buffer_samples, dtype=np.float32)
                result = _crossfade_with_overlap(result, silence, fade_samples)
                result = _crossfade_with_overlap(result, chunks[i], fade_samples)

        final_audio = result
        logger.info(
            f"Smart stitching applied: {len(chunks)} chunks, "
            f"{crossfade_ms}ms crossfades, {sentence_pause_ms}ms pauses"
        )
    else:
        fade_samples = int(safety_fade_ms / 1000 * sample_rate)
        num_chunks = len(segments)
        processed_chunks = []
        for i, chunk in enumerate(segments):
            processed = _apply_edge_fades(
                chunk, fade_samples,
                fade_in=(i != 0),
                fade_out=(i != num_chunks - 1),
            )
            processed_chunks.append(processed)
        final_audio = np.concatenate(processed_chunks)
        logger.info(f"Safety edge fades applied: {num_chunks} chunks, {safety_fade_ms}ms fades")

    final_audio = final_audio.astype(np.float32, copy=False)

    # Normalize to prevent clipping
    peak_amplitude = np.abs(final_audio).max()
    if peak_amplitude > peak_normalize_threshold:
        final_audio = final_audio * (peak_normalize_target / peak_amplitude)
        logger.warning(f"Audio normalized to prevent clipping (peak was {peak_amplitude:.3f})")

    # Global post-processing
    if config_manager.get_bool("audio_processing.enable_silence_trimming", False):
        final_audio = utils.trim_lead_trail_silence(final_audio, sample_rate)

    if config_manager.get_bool("audio_processing.enable_internal_silence_fix", False):
        final_audio = utils.fix_internal_silence(final_audio, sample_rate)

    if (config_manager.get_bool("audio_processing.enable_unvoiced_removal", False)
            and utils.PARSELMOUTH_AVAILABLE):
        final_audio = utils.remove_long_unvoiced_segments(final_audio, sample_rate)

    return final_audio


# --- Engine Lock & Batch State Management ---
import threading
import json
import zipfile

_engine_lock = threading.Lock()
_batch_jobs: Dict[str, Dict[str, Any]] = {}
_batch_jobs_lock = threading.Lock()

# --- End Audio Stitching Helper Functions ---


# --- Main UI Route ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_web_ui(request: Request):
    """Serves the main web interface (index.html)."""
    logger.info("Request received for main UI page ('/').")
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e_render:
        logger.error(f"Error rendering main UI page: {e_render}", exc_info=True)
        return HTMLResponse(
            "<html><body><h1>Internal Server Error</h1><p>Could not load the TTS interface. "
            "Please check server logs for more details.</p></body></html>",
            status_code=500,
        )


# --- API Endpoint for Model Information ---
@app.get("/api/model-info", tags=["Model Information"])
async def get_model_info_endpoint():
    """
    Returns detailed information about the currently loaded TTS model.
    This endpoint is used by the UI to display model status and
    conditionally show features like paralinguistic tags.
    """
    logger.debug("Request received for /api/model-info")
    try:
        model_info = engine.get_model_info()
        return model_info
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve model information"
        )


# --- API Endpoint for Initial UI Data ---
@app.get("/api/ui/initial-data", tags=["UI Helpers"])
async def get_ui_initial_data():
    """
    Provides all necessary initial data for the UI to render,
    including configuration, file lists, presets, and model information.
    """
    logger.info("Request received for /api/ui/initial-data.")
    try:
        full_config = get_full_config_for_template()
        reference_files = utils.get_valid_reference_files()
        predefined_voices = utils.get_predefined_voices()

        # Get model information for UI
        model_info = engine.get_model_info()

        loaded_presets = []
        presets_file = ui_static_path / "presets.yaml"
        if presets_file.exists():
            with open(presets_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
                if isinstance(yaml_content, list):
                    loaded_presets = yaml_content
                else:
                    logger.warning(
                        f"Invalid format in {presets_file}. Expected a list, got {type(yaml_content)}."
                    )
        else:
            logger.info(
                f"Presets file not found: {presets_file}. No presets will be loaded for initial data."
            )

        initial_gen_result_placeholder = {
            "outputUrl": None,
            "filename": None,
            "genTime": None,
            "submittedVoiceMode": None,
            "submittedPredefinedVoice": None,
            "submittedCloneFile": None,
        }

        return {
            "config": full_config,
            "reference_files": reference_files,
            "predefined_voices": predefined_voices,
            "presets": loaded_presets,
            "initial_gen_result": initial_gen_result_placeholder,
            "model_info": model_info,  # NEW: Include model information
        }
    except Exception as e:
        logger.error(f"Error preparing initial UI data for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to load initial data for UI."
        )


# --- Configuration Management API Endpoints ---
@app.post("/save_settings", response_model=UpdateStatusResponse, tags=["Configuration"])
async def save_settings_endpoint(request: Request):
    """
    Saves partial configuration updates to the config.yaml file.
    Merges the update with the current configuration.
    """
    logger.info("Request received for /save_settings.")
    try:
        partial_update = await request.json()
        if not isinstance(partial_update, dict):
            raise ValueError("Request body must be a JSON object for /save_settings.")
        logger.debug(f"Received partial config data to save: {partial_update}")

        if config_manager.update_and_save(partial_update):
            restart_needed = any(
                key in partial_update
                for key in ["server", "tts_engine", "paths", "model"]
            )
            message = "Settings saved successfully."
            if restart_needed:
                message += " A server restart may be required for some changes to take full effect."
            return UpdateStatusResponse(message=message, restart_needed=restart_needed)
        else:
            logger.error(
                "Failed to save configuration via config_manager.update_and_save."
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to save configuration file due to an internal error.",
            )
    except ValueError as ve:
        logger.error(f"Invalid data format for /save_settings: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(ve)}")
    except Exception as e:
        logger.error(f"Error processing /save_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings save: {str(e)}",
        )


@app.post(
    "/reset_settings", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def reset_settings_endpoint():
    """Resets the configuration in config.yaml back to hardcoded defaults."""
    logger.warning("Request received to reset all configurations to default values.")
    try:
        if config_manager.reset_and_save():
            logger.info("Configuration successfully reset to defaults and saved.")
            return UpdateStatusResponse(
                message="Configuration reset to defaults. Please reload the page. A server restart may be beneficial.",
                restart_needed=True,
            )
        else:
            logger.error("Failed to reset and save configuration via config_manager.")
            raise HTTPException(
                status_code=500, detail="Failed to reset and save configuration file."
            )
    except Exception as e:
        logger.error(f"Error processing /reset_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings reset: {str(e)}",
        )


@app.post(
    "/restart_server", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def restart_server_endpoint():
    """
    Triggers a hot-swap of the TTS model engine.
    Unloads the current model, clears VRAM, and loads the model defined in config.
    """
    logger.info("Request received for /restart_server (Model Hot-Swap).")

    try:
        # Attempt to reload the engine with the new configuration
        success = engine.reload_model()

        if success:
            model_info = engine.get_model_info()
            new_model_name = model_info.get("class_name", "Unknown Model")
            new_model_type = model_info.get("type", "unknown")
            message = f"Model hot-swap successful. Now running: {new_model_name} ({new_model_type})"
            logger.info(message)

            # restart_needed=False because we just performed the hot-swap successfully
            return UpdateStatusResponse(message=message, restart_needed=False)
        else:
            error_msg = "Model reload failed. The server may be in an inconsistent state. Check logs for details."
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error during model hot-swap: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during model reload: {str(e)}",
        )


# --- UI Helper API Endpoints ---
@app.get("/get_reference_files", response_model=List[str], tags=["UI Helpers"])
async def get_reference_files_api():
    """Returns a list of valid reference audio filenames (.wav, .mp3)."""
    logger.debug("Request for /get_reference_files.")
    try:
        return utils.get_valid_reference_files()
    except Exception as e:
        logger.error(f"Error getting reference files for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve reference audio files."
        )


@app.get(
    "/get_predefined_voices", response_model=List[Dict[str, str]], tags=["UI Helpers"]
)
async def get_predefined_voices_api():
    """Returns a list of predefined voices with display names and filenames."""
    logger.debug("Request for /get_predefined_voices.")
    try:
        return utils.get_predefined_voices()
    except Exception as e:
        logger.error(f"Error getting predefined voices for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve predefined voices list."
        )


# --- File Upload Endpoints ---
@app.post("/upload_reference", tags=["File Management"])
async def upload_reference_audio_endpoint(files: List[UploadFile] = File(...)):
    """
    Handles uploading of reference audio files (.wav, .mp3) for voice cloning.
    Validates files and saves them to the configured reference audio path.
    """
    logger.info(f"Request to /upload_reference with {len(files)} file(s).")
    ref_path = get_reference_audio_path(ensure_absolute=True)
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append(
                {"filename": "Unknown", "error": "File received with no filename."}
            )
            logger.warning("Upload attempt with no filename.")
            continue

        safe_filename = utils.sanitize_filename(file.filename)
        destination_path = ref_path / safe_filename

        try:
            if not (
                safe_filename.lower().endswith(".wav")
                or safe_filename.lower().endswith(".mp3")
            ):
                raise ValueError("Invalid file type. Only .wav and .mp3 are allowed.")

            if destination_path.exists():
                logger.info(
                    f"Reference file '{safe_filename}' already exists. Skipping duplicate upload."
                )
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(
                f"Successfully saved uploaded reference file to: {destination_path}"
            )

            max_duration = config_manager.get_int(
                "audio_output.max_reference_duration_sec", 30
            )
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration
            )
            if not is_valid:
                logger.warning(
                    f"Uploaded file '{safe_filename}' failed validation: {validation_msg}. Deleting."
                )
                destination_path.unlink(missing_ok=True)
                upload_errors.append(
                    {"filename": safe_filename, "error": validation_msg}
                )
            else:
                uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    all_current_reference_files = utils.get_valid_reference_files()
    response_data = {
        "message": f"Processed {len(files)} file(s).",
        "uploaded_files": uploaded_filenames_successfully,
        "all_reference_files": all_current_reference_files,
        "errors": upload_errors,
    }
    status_code = (
        200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    )
    if upload_errors:
        logger.warning(
            f"Upload to /upload_reference completed with {len(upload_errors)} error(s)."
        )
    return JSONResponse(content=response_data, status_code=status_code)


@app.post("/upload_predefined_voice", tags=["File Management"])
async def upload_predefined_voice_endpoint(files: List[UploadFile] = File(...)):
    """
    Handles uploading of predefined voice files (.wav, .mp3).
    Validates files and saves them to the configured predefined voices path.
    """
    logger.info(f"Request to /upload_predefined_voice with {len(files)} file(s).")
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append(
                {"filename": "Unknown", "error": "File received with no filename."}
            )
            logger.warning("Upload attempt for predefined voice with no filename.")
            continue

        safe_filename = utils.sanitize_filename(file.filename)
        destination_path = predefined_voices_path / safe_filename

        try:
            if not (
                safe_filename.lower().endswith(".wav")
                or safe_filename.lower().endswith(".mp3")
            ):
                raise ValueError(
                    "Invalid file type. Only .wav and .mp3 are allowed for predefined voices."
                )

            if destination_path.exists():
                logger.info(
                    f"Predefined voice file '{safe_filename}' already exists. Skipping duplicate upload."
                )
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(
                f"Successfully saved uploaded predefined voice file to: {destination_path}"
            )
            # Basic validation (can be extended if predefined voices have specific requirements)
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration_sec=None
            )  # No duration limit for predefined
            if not is_valid:
                logger.warning(
                    f"Uploaded predefined voice '{safe_filename}' failed basic validation: {validation_msg}. Deleting."
                )
                destination_path.unlink(missing_ok=True)
                upload_errors.append(
                    {"filename": safe_filename, "error": validation_msg}
                )
            else:
                uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing predefined voice file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    all_current_predefined_voices = (
        utils.get_predefined_voices()
    )  # Fetches formatted list
    response_data = {
        "message": f"Processed {len(files)} predefined voice file(s).",
        "uploaded_files": uploaded_filenames_successfully,  # List of raw filenames uploaded
        "all_predefined_voices": all_current_predefined_voices,  # Formatted list for UI
        "errors": upload_errors,
    }
    status_code = (
        200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    )
    if upload_errors:
        logger.warning(
            f"Upload to /upload_predefined_voice completed with {len(upload_errors)} error(s)."
        )
    return JSONResponse(content=response_data, status_code=status_code)


# --- TTS Generation Endpoint ---


@app.post(
    "/tts",
    tags=["TTS Generation"],
    summary="Generate speech with custom parameters",
    responses={
        200: {
            "content": {"audio/wav": {}, "audio/opus": {}},
            "description": "Successful audio generation.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or input.",
        },
        404: {
            "model": ErrorResponse,
            "description": "Required resource not found (e.g., voice file).",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during generation.",
        },
        503: {
            "model": ErrorResponse,
            "description": "TTS engine not available or model not loaded.",
        },
    },
)
async def custom_tts_endpoint(
    request: CustomTTSRequest, background_tasks: BackgroundTasks
):
    """
    Generates speech audio from text using specified parameters.
    Handles various voice modes (predefined, clone) and audio processing options.
    Returns audio as a stream (WAV or Opus).
    """
    perf_monitor = utils.PerformanceMonitor(
        enabled=config_manager.get_bool("server.enable_performance_monitor", False)
    )
    perf_monitor.record("TTS request received")

    if not engine.MODEL_LOADED:
        logger.error("TTS request failed: Model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    logger.info(
        f"Received /tts request: mode='{request.voice_mode}', format='{request.output_format}'"
    )
    logger.debug(
        f"TTS params: seed={request.seed}, split={request.split_text}, chunk_size={request.chunk_size}"
    )
    logger.debug(f"Input text (first 100 chars): '{request.text[:100]}...'")

    audio_prompt_path_for_engine: Optional[Path] = None
    if request.voice_mode == "predefined":
        if not request.predefined_voice_id:
            raise HTTPException(
                status_code=400,
                detail="Missing 'predefined_voice_id' for 'predefined' voice mode.",
            )
        voices_dir = get_predefined_voices_path(ensure_absolute=True)
        potential_path = voices_dir / request.predefined_voice_id
        if not potential_path.is_file():
            logger.error(f"Predefined voice file not found: {potential_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Predefined voice file '{request.predefined_voice_id}' not found.",
            )
        audio_prompt_path_for_engine = potential_path
        logger.info(f"Using predefined voice: {request.predefined_voice_id}")

    elif request.voice_mode == "clone":
        if not request.reference_audio_filename:
            raise HTTPException(
                status_code=400,
                detail="Missing 'reference_audio_filename' for 'clone' voice mode.",
            )
        ref_dir = get_reference_audio_path(ensure_absolute=True)
        potential_path = ref_dir / request.reference_audio_filename
        if not potential_path.is_file():
            logger.error(
                f"Reference audio file for cloning not found: {potential_path}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Reference audio file '{request.reference_audio_filename}' not found.",
            )
        max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
        is_valid, msg = utils.validate_reference_audio(potential_path, max_dur)
        if not is_valid:
            raise HTTPException(
                status_code=400, detail=f"Invalid reference audio: {msg}"
            )
        audio_prompt_path_for_engine = potential_path
        logger.info(
            f"Using reference audio for cloning: {request.reference_audio_filename}"
        )

    perf_monitor.record("Parameters and voice path resolved")

    all_audio_segments_np: List[np.ndarray] = []
    segment_is_pause: List[bool] = []  # Track which segments are explicit pauses
    final_output_sample_rate = (
        get_audio_sample_rate()
    )  # Target SR for the final output file
    engine_output_sample_rate: Optional[int] = (
        None  # SR from the TTS engine (e.g., 24000 Hz)
    )

    # --- Parse pause tags from input text ---
    pause_segments = utils.split_text_on_pause_tags(request.text)
    has_pause_tags = any(seg["type"] == "pause" for seg in pause_segments)

    if has_pause_tags:
        logger.info(f"Found pause tags in text. Split into {len(pause_segments)} segments.")

    # Build the final list of work items: each is either a text chunk to synthesize
    # or a pause (silence) to insert.
    work_items = []  # List of dicts: {"type": "text", "content": str} or {"type": "pause", "duration_ms": int}

    for seg in pause_segments:
        if seg["type"] == "pause":
            work_items.append(seg)
        else:
            # For text segments, apply chunking if enabled
            seg_text = seg["content"]
            if request.split_text and len(seg_text) > (
                request.chunk_size * 1.5 if request.chunk_size else 120 * 1.5
            ):
                chunk_size_to_use = (
                    request.chunk_size if request.chunk_size is not None else 120
                )
                text_chunks = utils.chunk_text_by_sentences(seg_text, chunk_size_to_use)
                for tc in text_chunks:
                    work_items.append({"type": "text", "content": tc})
            else:
                work_items.append({"type": "text", "content": seg_text})

    text_work_items = [w for w in work_items if w["type"] == "text"]
    if not text_work_items:
        raise HTTPException(
            status_code=400, detail="Text processing resulted in no usable text chunks."
        )

    perf_monitor.record(f"Text processed into {len(work_items)} work items ({len(text_work_items)} text, {len(work_items) - len(text_work_items)} pauses)")

    synth_index = 0
    for i, item in enumerate(work_items):
        if item["type"] == "pause":
            # Insert silence directly as a numpy array
            if engine_output_sample_rate is None:
                # Use default sample rate if we haven't synthesized yet
                pause_sr = 24000
            else:
                pause_sr = engine_output_sample_rate
            silence_samples = int(item["duration_ms"] / 1000 * pause_sr)
            silence_np = np.zeros(silence_samples, dtype=np.float32)
            all_audio_segments_np.append(silence_np)
            segment_is_pause.append(True)
            logger.info(f"Inserted {item['duration_ms']}ms pause")
            continue

        chunk = item["content"]
        # Skip chunks that are just punctuation/whitespace (e.g. lone em-dash from pause splitting)
        if not chunk or not chunk.strip() or all(c in ' \t\n\r—–-.,;:!?\u3000' for c in chunk):
            logger.info(f"Skipping trivial chunk: {repr(chunk)}")
            continue
        synth_index += 1
        logger.info(f"Synthesizing chunk {synth_index}/{len(text_work_items)}...")
        try:
            chunk_audio_tensor, chunk_sr_from_engine = engine.synthesize(
                text=chunk,
                audio_prompt_path=(
                    str(audio_prompt_path_for_engine)
                    if audio_prompt_path_for_engine
                    else None
                ),
                temperature=(
                    request.temperature
                    if request.temperature is not None
                    else get_gen_default_temperature()
                ),
                exaggeration=(
                    request.exaggeration
                    if request.exaggeration is not None
                    else get_gen_default_exaggeration()
                ),
                cfg_weight=(
                    request.cfg_weight
                    if request.cfg_weight is not None
                    else get_gen_default_cfg_weight()
                ),
                seed=(
                    request.seed if request.seed is not None else get_gen_default_seed()
                ),
                language=(
                    request.language
                    if request.language is not None
                    else get_gen_default_language()
                ),
                # Qwen3-specific params
                speaker=getattr(request, 'qwen3_speaker', None),
                instruct=getattr(request, 'qwen3_instruct', None),
                ref_text=getattr(request, 'qwen3_ref_text', None),
            )
            perf_monitor.record(f"Engine synthesized chunk {i+1}")

            if chunk_audio_tensor is None or chunk_sr_from_engine is None:
                error_detail = f"TTS engine failed to synthesize audio for chunk {i+1}."
                logger.error(error_detail)
                raise HTTPException(status_code=500, detail=error_detail)

            if engine_output_sample_rate is None:
                engine_output_sample_rate = chunk_sr_from_engine
            elif engine_output_sample_rate != chunk_sr_from_engine:
                logger.warning(
                    f"Inconsistent sample rate from engine: chunk {i+1} ({chunk_sr_from_engine}Hz) "
                    f"differs from previous ({engine_output_sample_rate}Hz). Using first chunk's SR."
                )

            current_processed_audio_tensor = chunk_audio_tensor

            speed_factor_to_use = (
                request.speed_factor
                if request.speed_factor is not None
                else get_gen_default_speed_factor()
            )
            if speed_factor_to_use != 1.0:
                # For numpy arrays (Qwen3), apply speed via librosa directly
                if isinstance(current_processed_audio_tensor, np.ndarray):
                    audio_1d = current_processed_audio_tensor.squeeze()
                    stretched = librosa.effects.time_stretch(
                        y=audio_1d.astype(np.float32), rate=speed_factor_to_use
                    )
                    current_processed_audio_tensor = stretched
                else:
                    current_processed_audio_tensor, _ = utils.apply_speed_factor(
                        current_processed_audio_tensor,
                        chunk_sr_from_engine,
                        speed_factor_to_use,
                    )
                perf_monitor.record(f"Speed factor applied to chunk {i+1}")

            # ### MODIFICATION ###
            # All other processing is REMOVED from the loop.
            # We will process the final concatenated audio clip.
            # Handle both torch.Tensor (Chatterbox) and numpy.ndarray (Qwen3)
            if isinstance(current_processed_audio_tensor, np.ndarray):
                processed_audio_np = current_processed_audio_tensor.squeeze()
            else:
                processed_audio_np = current_processed_audio_tensor.cpu().numpy().squeeze()
            all_audio_segments_np.append(processed_audio_np)
            segment_is_pause.append(False)

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e_chunk:
            error_detail = f"Error processing audio chunk {i+1}: {str(e_chunk)}"
            logger.error(error_detail, exc_info=True)
            raise HTTPException(status_code=500, detail=error_detail)

    if not all_audio_segments_np:
        logger.error("No audio segments were successfully generated.")
        raise HTTPException(
            status_code=500, detail="Audio generation resulted in no output."
        )

    if engine_output_sample_rate is None:
        logger.error("Engine output sample rate could not be determined.")
        raise HTTPException(
            status_code=500, detail="Failed to determine engine sample rate."
        )
    try:
        final_audio_np = _stitch_audio_segments(
            all_audio_segments_np, segment_is_pause, engine_output_sample_rate
        )
        perf_monitor.record("Audio chunks stitched")

    except ValueError as e_concat:
        logger.error(f"Audio concatenation/stitching failed: {e_concat}", exc_info=True)
        for idx, seg in enumerate(all_audio_segments_np):
            logger.error(f"Segment {idx} shape: {seg.shape}, dtype: {seg.dtype}")
        raise HTTPException(
            status_code=500, detail=f"Audio stitching error: {e_concat}"
        )

    output_format_str = (
        request.output_format if request.output_format else get_audio_output_format()
    )

    encoded_audio_bytes = utils.encode_audio(
        audio_array=final_audio_np,
        sample_rate=engine_output_sample_rate,
        output_format=output_format_str,
        target_sample_rate=final_output_sample_rate,
    )
    perf_monitor.record(
        f"Final audio encoded to {output_format_str} (target SR: {final_output_sample_rate}Hz from engine SR: {engine_output_sample_rate}Hz)"
    )

    if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
        logger.error(
            f"Failed to encode final audio to format: {output_format_str} or output is too small ({len(encoded_audio_bytes or b'')} bytes)."
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode audio to {output_format_str} or generated invalid audio.",
        )

    media_type = f"audio/{output_format_str}"
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    suggested_filename_base = f"tts_output_{timestamp_str}"
    download_filename = utils.sanitize_filename(
        f"{suggested_filename_base}.{output_format_str}"
    )
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}

    logger.info(
        f"Successfully generated audio: {download_filename}, {len(encoded_audio_bytes)} bytes, type {media_type}."
    )
    logger.debug(perf_monitor.report())

    # Optional: Save to disk if enabled
    if config_manager.get_bool("audio_output.save_to_disk", False):
        output_dir = get_output_path(ensure_absolute=True)
        output_file_path = output_dir / download_filename
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "wb") as f:
                f.write(encoded_audio_bytes)
            if not output_file_path.exists() or output_file_path.stat().st_size < 100:
                logger.error(f"File save verification failed for {output_file_path}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save audio file to {output_file_path}",
                )
            logger.info(f"Audio saved to disk: {output_file_path}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to save audio to {output_file_path}: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to save audio file: {e}"
            )

    return StreamingResponse(
        io.BytesIO(encoded_audio_bytes), media_type=media_type, headers=headers
    )


@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech_endpoint(request: OpenAISpeechRequest):
    # Determine the audio prompt path based on the voice parameter
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    reference_audio_path = get_reference_audio_path(ensure_absolute=True)
    voice_path_predefined = predefined_voices_path / request.voice
    voice_path_reference = reference_audio_path / request.voice

    if voice_path_predefined.is_file():
        audio_prompt_path = voice_path_predefined
    elif voice_path_reference.is_file():
        audio_prompt_path = voice_path_reference
    else:
        raise HTTPException(
            status_code=404, detail=f"Voice file '{request.voice}' not found."
        )

    # Check if the TTS model is loaded
    if not engine.MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    try:
        # Use the provided seed or the default
        seed_to_use = (
            request.seed if request.seed is not None else get_gen_default_seed()
        )

        # Synthesize the audio
        audio_tensor, sr = engine.synthesize(
            text=request.input_,
            audio_prompt_path=str(audio_prompt_path),
            temperature=get_gen_default_temperature(),
            exaggeration=get_gen_default_exaggeration(),
            cfg_weight=get_gen_default_cfg_weight(),
            seed=seed_to_use,
            language=get_gen_default_language(),
        )

        if audio_tensor is None or sr is None:
            raise HTTPException(
                status_code=500, detail="TTS engine failed to synthesize audio."
            )

        # Apply speed factor if not 1.0
        if request.speed != 1.0:
            audio_tensor, _ = utils.apply_speed_factor(audio_tensor, sr, request.speed)

        # Convert tensor to numpy array
        audio_np = audio_tensor.cpu().numpy()

        # Ensure it's 1D
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze()

        # Encode the audio to the requested format
        encoded_audio = utils.encode_audio(
            audio_array=audio_np,
            sample_rate=sr,
            output_format=request.response_format,
            target_sample_rate=get_audio_sample_rate(),
        )

        if encoded_audio is None:
            raise HTTPException(status_code=500, detail="Failed to encode audio.")

        # Determine the media type
        media_type = f"audio/{request.response_format}"

        # Optional: Save to disk if enabled
        if config_manager.get_bool("audio_output.save_to_disk", False):
            output_dir = get_output_path(ensure_absolute=True)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            download_filename = f"openai_tts_{timestamp_str}.{request.response_format}"
            output_file_path = output_dir / download_filename
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_file_path, "wb") as f:
                    f.write(encoded_audio)
                if (
                    not output_file_path.exists()
                    or output_file_path.stat().st_size < 100
                ):
                    logger.error(
                        f"File save verification failed for {output_file_path}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to save audio file to {output_file_path}",
                    )
                logger.info(
                    f"OpenAI-compatible audio saved to disk: {output_file_path}"
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(
                    f"Failed to save audio to {output_file_path}: {e}", exc_info=True
                )
                raise HTTPException(
                    status_code=500, detail=f"Failed to save audio file: {e}"
                )

        # Return the streaming response
        return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

    except Exception as e:
        logger.error(f"Error in openai_speech_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- Batch/Chapter Generation ---

def _generate_chapter_audio(
    chapter_text: str,
    audio_prompt_path: Optional[str],
    request_params: dict,
) -> Tuple[np.ndarray, int]:
    """
    Generate audio for a single chapter. Uses the same pipeline as /tts.
    Returns (audio_numpy_array, sample_rate).
    """
    # Parse pause tags
    pause_segments = utils.split_text_on_pause_tags(chapter_text)

    work_items = []
    for seg in pause_segments:
        if seg["type"] == "pause":
            work_items.append(seg)
        else:
            seg_text = seg["content"]
            if request_params.get("split_text", True) and len(seg_text) > (
                request_params.get("chunk_size", 120) * 1.5
            ):
                text_chunks = utils.chunk_text_by_sentences(
                    seg_text, request_params.get("chunk_size", 120)
                )
                for tc in text_chunks:
                    work_items.append({"type": "text", "content": tc})
            else:
                work_items.append({"type": "text", "content": seg_text})

    all_audio_segments = []
    segment_is_pause = []
    engine_sr = None

    for item in work_items:
        if item["type"] == "pause":
            sr = engine_sr or 24000
            silence = np.zeros(int(item["duration_ms"] / 1000 * sr), dtype=np.float32)
            all_audio_segments.append(silence)
            segment_is_pause.append(True)
            continue

        chunk = item["content"]
        if not chunk or not chunk.strip() or all(c in ' \t\n\r\u2014\u2013-.,;:!?\u3000' for c in chunk):
            continue

        with _engine_lock:
            audio_data, sr = engine.synthesize(
                text=chunk,
                audio_prompt_path=audio_prompt_path,
                temperature=request_params.get("temperature", get_gen_default_temperature()),
                exaggeration=request_params.get("exaggeration", get_gen_default_exaggeration()),
                cfg_weight=request_params.get("cfg_weight", get_gen_default_cfg_weight()),
                seed=request_params.get("seed", get_gen_default_seed()),
                language=request_params.get("language", get_gen_default_language()),
                speaker=request_params.get("qwen3_speaker"),
                instruct=request_params.get("qwen3_instruct"),
                ref_text=request_params.get("qwen3_ref_text"),
            )

        if audio_data is None or sr is None:
            raise RuntimeError(f"Engine failed to synthesize chunk: {chunk[:50]}...")

        if engine_sr is None:
            engine_sr = sr

        # Handle speed factor
        speed = request_params.get("speed_factor", 1.0)
        if speed and speed != 1.0:
            if isinstance(audio_data, np.ndarray):
                audio_data = librosa.effects.time_stretch(
                    y=audio_data.squeeze().astype(np.float32), rate=speed
                )
            else:
                audio_data, _ = utils.apply_speed_factor(audio_data, sr, speed)

        # Convert to numpy
        if isinstance(audio_data, np.ndarray):
            all_audio_segments.append(audio_data.squeeze())
        else:
            all_audio_segments.append(audio_data.cpu().numpy().squeeze())
        segment_is_pause.append(False)

    if not all_audio_segments:
        raise RuntimeError("No audio segments generated for chapter")

    # Default sample rate if no synthesis occurred (all pauses)
    if engine_sr is None:
        engine_sr = 24000

    # Stitch
    final_audio = _stitch_audio_segments(all_audio_segments, segment_is_pause, engine_sr)
    return final_audio, engine_sr


def _batch_worker(batch_id: str):
    """Background worker that generates audio for each chapter in a batch."""
    with _batch_jobs_lock:
        job = _batch_jobs.get(batch_id)
        if not job:
            logger.error(f"Batch {batch_id}: job not found")
            return
        job["status"] = "generating"

    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    request_params = job["request_params"]
    audio_prompt_path = job.get("audio_prompt_path")
    output_format = request_params.get("output_format", "wav")
    completed = 0
    failed = 0

    for idx, chapter_info in enumerate(job["chapters"]):
        with _batch_jobs_lock:
            job["current_chapter"] = idx
            job["chapters"][idx]["status"] = "generating"

        try:
            logger.info(f"Batch {batch_id}: Generating chapter {idx + 1}/{len(job['chapters'])}: {chapter_info['title']}")

            audio_np, sr = _generate_chapter_audio(
                chapter_info["text"],
                audio_prompt_path,
                request_params,
            )

            # Encode
            target_sr = get_audio_sample_rate()
            encoded = utils.encode_audio(
                audio_array=audio_np,
                sample_rate=sr,
                output_format=output_format,
                target_sample_rate=target_sr,
            )

            if not encoded or len(encoded) < 100:
                raise RuntimeError("Encoding produced empty or invalid audio")

            # Save to disk
            filename = f"{idx:03d}.{output_format}"
            filepath = output_dir / filename
            with open(filepath, "wb") as f:
                f.write(encoded)

            with _batch_jobs_lock:
                job["chapters"][idx]["status"] = "completed"
                job["chapters"][idx]["filename"] = filename
                job["chapters"][idx]["download_url"] = f"/outputs/{batch_id}/{filename}"
                job["completed_chapters"] = job.get("completed_chapters", 0) + 1
                completed += 1

            logger.info(f"Batch {batch_id}: Chapter {idx + 1} completed: {filename} ({len(encoded)} bytes)")

            # Free memory
            del audio_np, encoded
            import gc
            gc.collect()

        except Exception as e:
            logger.error(f"Batch {batch_id}: Chapter {idx + 1} failed: {e}", exc_info=True)
            with _batch_jobs_lock:
                job["chapters"][idx]["status"] = "failed"
                job["chapters"][idx]["error"] = str(e)
                failed += 1

    # Write manifest
    try:
        manifest = {
            "batch_id": batch_id,
            "chapters": [
                {"index": ch["index"], "title": ch["title"], "status": ch["status"],
                 "filename": ch.get("filename")}
                for ch in job["chapters"]
            ],
            "output_format": output_format,
            "total": len(job["chapters"]),
            "completed": completed,
            "failed": failed,
        }
        with open(output_dir / "batch_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        pass

    # Set final status
    with _batch_jobs_lock:
        job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        job["current_chapter"] = None
        if failed == 0:
            job["status"] = "completed"
        elif completed == 0:
            job["status"] = "failed"
            job["error"] = "All chapters failed to generate"
        else:
            job["status"] = "partial"

    logger.info(f"Batch {batch_id}: Finished. {completed} completed, {failed} failed.")


@app.post("/batch/start", tags=["Batch Generation"])
async def batch_start(request: BatchTTSRequest):
    """Start a batch/chapter TTS generation job."""
    if not engine.MODEL_LOADED:
        raise HTTPException(status_code=503, detail="TTS engine model is not loaded.")

    # Split text into chapters
    chapters = utils.split_text_into_chapters(request.text, request.separator)
    if not chapters:
        raise HTTPException(status_code=400, detail="No chapters found in text with the given separator.")

    # Resolve voice path (same validation as /tts)
    audio_prompt_path = None
    if request.voice_mode == "predefined" and request.predefined_voice_id:
        voices_dir = get_predefined_voices_path(ensure_absolute=True)
        path = voices_dir / request.predefined_voice_id
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Predefined voice '{request.predefined_voice_id}' not found.")
        audio_prompt_path = str(path)
    elif request.voice_mode == "clone" and request.reference_audio_filename:
        ref_dir = get_reference_audio_path(ensure_absolute=True)
        path = ref_dir / request.reference_audio_filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Reference audio '{request.reference_audio_filename}' not found.")
        audio_prompt_path = str(path)

    # Create batch ID and output dir
    batch_id = f"batch_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output_dir = get_output_path(ensure_absolute=True) / batch_id

    # Build chapter list
    chapter_list = []
    for idx, ch in enumerate(chapters):
        chapter_list.append({
            "index": idx,
            "title": ch["title"],
            "text": ch["text"],
            "status": "pending",
            "filename": None,
            "download_url": None,
            "error": None,
        })

    # Store request params
    params = {
        "output_format": request.output_format or "wav",
        "split_text": request.split_text,
        "chunk_size": request.chunk_size,
        "temperature": request.temperature,
        "exaggeration": request.exaggeration,
        "cfg_weight": request.cfg_weight,
        "seed": request.seed,
        "speed_factor": request.speed_factor,
        "language": request.language,
        "qwen3_speaker": request.qwen3_speaker,
        "qwen3_instruct": request.qwen3_instruct,
        "qwen3_ref_text": request.qwen3_ref_text,
    }

    job = {
        "batch_id": batch_id,
        "status": "queued",
        "total_chapters": len(chapter_list),
        "completed_chapters": 0,
        "current_chapter": None,
        "chapters": chapter_list,
        "output_dir": str(output_dir),
        "audio_prompt_path": audio_prompt_path,
        "request_params": params,
        "error": None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "completed_at": None,
    }

    with _batch_jobs_lock:
        _batch_jobs[batch_id] = job

    # Launch worker thread
    worker = threading.Thread(target=_batch_worker, args=(batch_id,), daemon=True)
    worker.start()

    logger.info(f"Batch {batch_id}: Started with {len(chapter_list)} chapters")

    return {
        "batch_id": batch_id,
        "total_chapters": len(chapter_list),
        "status": "queued",
        "message": f"Batch job started with {len(chapter_list)} chapters.",
    }


@app.get("/batch/status/{batch_id}", response_model=BatchStatusResponse, tags=["Batch Generation"])
async def batch_status(batch_id: str):
    """Get the status of a batch generation job."""
    with _batch_jobs_lock:
        job = _batch_jobs.get(batch_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Batch job '{batch_id}' not found.")

    return BatchStatusResponse(
        batch_id=job["batch_id"],
        status=job["status"],
        total_chapters=job["total_chapters"],
        completed_chapters=job.get("completed_chapters", 0),
        current_chapter=job.get("current_chapter"),
        chapters=[
            BatchChapterStatus(
                index=ch["index"],
                title=ch["title"],
                status=ch["status"],
                filename=ch.get("filename"),
                download_url=ch.get("download_url"),
                error=ch.get("error"),
            )
            for ch in job["chapters"]
        ],
        error=job.get("error"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
    )


@app.get("/batch/download/{batch_id}", tags=["Batch Generation"])
async def batch_download(batch_id: str):
    """Download all completed chapters as a ZIP file."""
    with _batch_jobs_lock:
        job = _batch_jobs.get(batch_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Batch job '{batch_id}' not found.")

    if job["status"] not in ("completed", "partial"):
        raise HTTPException(
            status_code=400,
            detail=f"Batch is still {job['status']}. Wait for completion before downloading."
        )

    output_dir = Path(job["output_dir"])
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for ch in job["chapters"]:
            if ch["status"] == "completed" and ch.get("filename"):
                filepath = output_dir / ch["filename"]
                if filepath.is_file():
                    zf.write(filepath, ch["filename"])

    zip_buffer.seek(0)
    zip_filename = f"{batch_id}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


@app.get("/batch/audio/{batch_id}/{filename}", tags=["Batch Generation"])
async def batch_serve_audio(batch_id: str, filename: str):
    """Serve a batch chapter audio file with proper range request support."""
    with _batch_jobs_lock:
        job = _batch_jobs.get(batch_id)

    if not job:
        raise HTTPException(status_code=404, detail="Batch not found.")

    output_dir = Path(job["output_dir"])
    filepath = output_dir / filename

    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    # FileResponse supports range requests for proper audio streaming
    ext = filepath.suffix.lstrip('.')
    media_type = f"audio/{ext}" if ext in ('wav', 'mp3', 'opus') else "application/octet-stream"
    return FileResponse(filepath, media_type=media_type, filename=filename)


class BatchRegenerateRequest(BaseModel):
    text: Optional[str] = None  # Updated text for the section (optional)


@app.post("/batch/regenerate/{batch_id}/{chapter_index}", tags=["Batch Generation"])
async def batch_regenerate_chapter(batch_id: str, chapter_index: int, body: BatchRegenerateRequest = None):
    """Regenerate a single chapter in a batch job, optionally with updated text."""
    if not engine.MODEL_LOADED:
        raise HTTPException(status_code=503, detail="TTS engine model is not loaded.")

    with _batch_jobs_lock:
        job = _batch_jobs.get(batch_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Batch job '{batch_id}' not found.")

    if chapter_index < 0 or chapter_index >= len(job["chapters"]):
        raise HTTPException(status_code=400, detail=f"Invalid chapter index {chapter_index}.")

    chapter = job["chapters"][chapter_index]

    # Update text if provided
    if body and body.text is not None:
        with _batch_jobs_lock:
            chapter["text"] = body.text

    # Mark as regenerating
    with _batch_jobs_lock:
        chapter["status"] = "generating"
        chapter["error"] = None
        job["status"] = "generating"
        job["current_chapter"] = chapter_index

    def _regen_worker():
        try:
            output_dir = Path(job["output_dir"])
            request_params = job["request_params"]
            audio_prompt_path = job.get("audio_prompt_path")
            output_format = request_params.get("output_format", "wav")

            logger.info(f"Batch {batch_id}: Regenerating chapter {chapter_index + 1}: {chapter['title']}")

            audio_np, sr = _generate_chapter_audio(
                chapter["text"],
                audio_prompt_path,
                request_params,
            )

            target_sr = get_audio_sample_rate()
            encoded = utils.encode_audio(
                audio_array=audio_np, sample_rate=sr,
                output_format=output_format, target_sample_rate=target_sr,
            )

            if not encoded or len(encoded) < 100:
                raise RuntimeError("Encoding produced empty or invalid audio")

            # Delete old file if exists
            if chapter.get("filename"):
                old_path = output_dir / chapter["filename"]
                if old_path.is_file():
                    old_path.unlink()

            filename = f"{chapter_index:03d}.{output_format}"
            filepath = output_dir / filename
            with open(filepath, "wb") as f:
                f.write(encoded)

            with _batch_jobs_lock:
                chapter["status"] = "completed"
                chapter["filename"] = filename
                chapter["download_url"] = f"/outputs/{batch_id}/{filename}"
                chapter["error"] = None
                job["current_chapter"] = None
                # Recalculate overall status
                statuses = [ch["status"] for ch in job["chapters"]]
                if all(s == "completed" for s in statuses):
                    job["status"] = "completed"
                elif any(s == "failed" for s in statuses):
                    job["status"] = "partial"
                else:
                    job["status"] = "completed"
                job["completed_chapters"] = sum(1 for s in statuses if s == "completed")

            logger.info(f"Batch {batch_id}: Chapter {chapter_index + 1} regenerated: {filename}")

            del audio_np, encoded
            import gc
            gc.collect()

        except Exception as e:
            logger.error(f"Batch {batch_id}: Regenerate chapter {chapter_index + 1} failed: {e}", exc_info=True)
            with _batch_jobs_lock:
                chapter["status"] = "failed"
                chapter["error"] = str(e)
                job["current_chapter"] = None
                statuses = [ch["status"] for ch in job["chapters"]]
                if all(s == "failed" for s in statuses):
                    job["status"] = "failed"
                else:
                    job["status"] = "partial"

    worker = threading.Thread(target=_regen_worker, daemon=True)
    worker.start()

    return {"message": f"Regenerating chapter {chapter_index + 1}: {chapter['title']}"}


# --- Main Execution ---
if __name__ == "__main__":
    server_host = get_host()
    server_port = get_port()

    logger.info(f"Starting TTS Server directly on http://{server_host}:{server_port}")
    logger.info(
        f"API documentation will be available at http://{server_host}:{server_port}/docs"
    )
    logger.info(f"Web UI will be available at http://{server_host}:{server_port}/")

    import uvicorn

    uvicorn.run(
        "server:app",
        host=server_host,
        port=server_port,
        log_level="info",
        workers=1,
        reload=False,
        timeout_keep_alive=600,  # 10 min keep-alive for long generation jobs
    )

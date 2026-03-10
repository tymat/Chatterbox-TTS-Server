# File: engine.py
# Core TTS model loading and speech generation logic.

import gc
import logging
import random
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

# Defensive PyTorch import - not needed for MLX-based engines
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Defensive Chatterbox imports - not available in qwen3 venv
try:
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.models.s3gen.const import S3GEN_SR
    CHATTERBOX_AVAILABLE = True
except ImportError:
    ChatterboxTTS = None
    S3GEN_SR = 24000
    CHATTERBOX_AVAILABLE = False

# Defensive Turbo import - Turbo may not be available in older package versions
try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    TURBO_AVAILABLE = True
except ImportError:
    ChatterboxTurboTTS = None
    TURBO_AVAILABLE = False

# Defensive Multilingual import
try:
    from chatterbox import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

    MULTILINGUAL_AVAILABLE = True
except ImportError:
    ChatterboxMultilingualTTS = None
    SUPPORTED_LANGUAGES = {}
    MULTILINGUAL_AVAILABLE = False

# Defensive Qwen3-TTS import (MLX-based)
try:
    from engine_qwen3 import (
        Qwen3TTSAdapter,
        QWEN3_SUPPORTED_LANGUAGES,
        QWEN3_SPEAKERS_CUSTOM_VOICE,
        QWEN3_MODEL_VARIANTS,
    )
    QWEN3_AVAILABLE = True
except ImportError:
    Qwen3TTSAdapter = None
    QWEN3_SUPPORTED_LANGUAGES = {}
    QWEN3_SPEAKERS_CUSTOM_VOICE = {}
    QWEN3_MODEL_VARIANTS = {}
    QWEN3_AVAILABLE = False

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# Log Turbo availability status at module load time
if CHATTERBOX_AVAILABLE:
    logger.info("ChatterboxTTS (original) is available.")
else:
    logger.info("ChatterboxTTS not available (chatterbox package not installed).")

if TURBO_AVAILABLE:
    logger.info("ChatterboxTurboTTS is available in the installed chatterbox package.")
else:
    logger.info("ChatterboxTurboTTS not available in installed chatterbox package.")

# Log Multilingual availability status at module load time
if MULTILINGUAL_AVAILABLE:
    logger.info("ChatterboxMultilingualTTS is available in the installed chatterbox package.")
    logger.info(f"Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
else:
    logger.info("ChatterboxMultilingualTTS not available in installed chatterbox package.")

# Log Qwen3-TTS availability status
if QWEN3_AVAILABLE:
    logger.info("Qwen3-TTS (MLX) is available.")
    logger.info(f"Qwen3 model variants: {list(QWEN3_MODEL_VARIANTS.keys())}")
else:
    logger.info("Qwen3-TTS not available (mlx-audio not installed).")

# Model selector whitelist - maps config values to model types
MODEL_SELECTOR_MAP = {
    # Original model selectors
    "chatterbox": "original",
    "original": "original",
    "resembleai/chatterbox": "original",
    # Turbo model selectors
    "chatterbox-turbo": "turbo",
    "turbo": "turbo",
    "resembleai/chatterbox-turbo": "turbo",
    # Multilingual model selectors
    "chatterbox-multilingual": "multilingual",
    "multilingual": "multilingual",
    # Qwen3-TTS model selectors (MLX)
    "qwen3-tts": "qwen3",
    "qwen3-tts-base": "qwen3",
    "qwen3-tts-custom": "qwen3",
    "qwen3-tts-voice-design": "qwen3",
    "qwen3-tts-base-lite": "qwen3",
    "qwen3-tts-custom-lite": "qwen3",
    "qwen3-tts-voice-design-lite": "qwen3",
}

# Paralinguistic tags supported by Turbo model
TURBO_PARALINGUISTIC_TAGS = [
    "laugh",
    "chuckle",
    "sigh",
    "gasp",
    "cough",
    "clear throat",
    "sniff",
    "groan",
    "shush",
]

# --- Global Module Variables ---
chatterbox_model: Optional[ChatterboxTTS] = None
MODEL_LOADED: bool = False
model_device: Optional[str] = (
    None  # Stores the resolved device string ('cuda' or 'cpu')
)

# Track which model type is loaded
loaded_model_type: Optional[str] = None  # "original" or "turbo"
loaded_model_class_name: Optional[str] = None  # "ChatterboxTTS" or "ChatterboxTurboTTS"


def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    """
    if TORCH_AVAILABLE:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")


def _test_cuda_functionality() -> bool:
    """
    Tests if CUDA is actually functional, not just available.

    Returns:
        bool: True if CUDA works, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.cuda()
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA functionality test failed: {e}")
        return False


def _test_mps_functionality() -> bool:
    """
    Tests if MPS is actually functional, not just available.

    Returns:
        bool: True if MPS works, False otherwise.
    """
    if not torch.backends.mps.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.to("mps")
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS functionality test failed: {e}")
        return False


def _get_model_class(selector: str) -> tuple:
    """
    Determines which model class to use based on the config selector value.

    Args:
        selector: The value from config model.repo_id

    Returns:
        Tuple of (model_class, model_type_string)

    Raises:
        ImportError: If Turbo or Multilingual is selected but not available in the package
    """
    selector_normalized = selector.lower().strip()
    model_type = MODEL_SELECTOR_MAP.get(selector_normalized)

    if model_type == "turbo":
        if not TURBO_AVAILABLE:
            raise ImportError(
                f"Model selector '{selector}' requires ChatterboxTurboTTS, "
                f"but it is not available in the installed chatterbox package. "
                f"Please update the chatterbox-tts package to the latest version, "
                f"or use 'chatterbox' to select the original model."
            )
        logger.info(
            f"Model selector '{selector}' resolved to Turbo model (ChatterboxTurboTTS)"
        )
        return ChatterboxTurboTTS, "turbo"

    if model_type == "multilingual":
        if not MULTILINGUAL_AVAILABLE:
            raise ImportError(
                f"Model selector '{selector}' requires ChatterboxMultilingualTTS, "
                f"but it is not available in the installed chatterbox package. "
                f"Please update the chatterbox-tts package to the latest version, "
                f"or use 'chatterbox' to select the original model."
            )
        logger.info(
            f"Model selector '{selector}' resolved to Multilingual model (ChatterboxMultilingualTTS)"
        )
        return ChatterboxMultilingualTTS, "multilingual"

    if model_type == "qwen3":
        if not QWEN3_AVAILABLE:
            raise ImportError(
                f"Model selector '{selector}' requires Qwen3-TTS (MLX), "
                f"but it is not available. Please install mlx-audio, "
                f"or use 'chatterbox' to select the original model."
            )
        logger.info(
            f"Model selector '{selector}' resolved to Qwen3-TTS model (MLX)"
        )
        return Qwen3TTSAdapter, "qwen3"

    if model_type == "original":
        if not CHATTERBOX_AVAILABLE:
            raise ImportError(
                f"Model selector '{selector}' requires ChatterboxTTS, "
                f"but it is not available. Please install chatterbox-tts."
            )
        logger.info(
            f"Model selector '{selector}' resolved to Original model (ChatterboxTTS)"
        )
        return ChatterboxTTS, "original"

    # Unknown selector - try to find something available
    logger.warning(
        f"Unknown model selector '{selector}'. "
        f"Valid values: chatterbox, chatterbox-turbo, chatterbox-multilingual, "
        f"qwen3-tts, qwen3-tts-base, qwen3-tts-custom, original, turbo, multilingual. "
        f"Attempting to find an available model."
    )
    if CHATTERBOX_AVAILABLE:
        return ChatterboxTTS, "original"
    elif QWEN3_AVAILABLE:
        return Qwen3TTSAdapter, "qwen3"
    else:
        raise ImportError("No TTS model packages are available.")


def get_model_info() -> dict:
    """
    Returns information about the currently loaded model.
    Used by the API to expose model details to the UI.

    Returns:
        Dictionary containing model information
    """
    is_qwen3 = loaded_model_type == "qwen3"
    is_multilingual = loaded_model_type == "multilingual"

    # Determine supported languages based on model type
    if is_qwen3:
        supported_langs = QWEN3_SUPPORTED_LANGUAGES
    elif is_multilingual:
        supported_langs = SUPPORTED_LANGUAGES
    else:
        supported_langs = {"en": "English"}

    return {
        "loaded": MODEL_LOADED,
        "type": loaded_model_type,  # "original", "turbo", "multilingual", or "qwen3"
        "class_name": loaded_model_class_name,
        "device": model_device,
        "sample_rate": chatterbox_model.sr if chatterbox_model else None,
        "supports_paralinguistic_tags": loaded_model_type == "turbo",
        "available_paralinguistic_tags": (
            TURBO_PARALINGUISTIC_TAGS if loaded_model_type == "turbo" else []
        ),
        "turbo_available_in_package": TURBO_AVAILABLE,
        "multilingual_available_in_package": MULTILINGUAL_AVAILABLE,
        "qwen3_available_in_package": QWEN3_AVAILABLE,
        "supports_multilingual": is_multilingual or is_qwen3,
        "supported_languages": supported_langs,
        # Qwen3-specific info
        "is_qwen3": is_qwen3,
        "qwen3_speakers": (
            QWEN3_SPEAKERS_CUSTOM_VOICE if is_qwen3 and chatterbox_model else {}
        ),
        "qwen3_model_variant": (
            chatterbox_model._model_variant if is_qwen3 and chatterbox_model else None
        ),
    }


def load_model() -> bool:
    """
    Loads the TTS model.
    This version directly attempts to load from the Hugging Face repository (or its cache)
    using `from_pretrained`, bypassing the local `paths.model_cache` directory.
    Updates global variables `chatterbox_model`, `MODEL_LOADED`, and `model_device`.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device
    global loaded_model_type, loaded_model_class_name

    if MODEL_LOADED:
        logger.info("TTS model is already loaded.")
        return True

    try:
        # Get the model selector from config
        model_selector = config_manager.get_string("model.repo_id", "chatterbox-turbo")
        logger.info(f"Model selector from config: '{model_selector}'")

        try:
            # Determine which model class to use
            model_class, model_type = _get_model_class(model_selector)
        except ImportError as e_import:
            logger.error(
                f"Failed to resolve model class: {e_import}",
                exc_info=True,
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False

        # Qwen3-TTS uses MLX (Apple Silicon native) - skip PyTorch device detection
        if model_type == "qwen3":
            model_device = "mlx"
            logger.info("Qwen3-TTS uses MLX framework (Apple Silicon native). Device: mlx")

            # Resolve HuggingFace model ID from selector
            qwen3_model_id = QWEN3_MODEL_VARIANTS.get(
                model_selector.lower().strip(),
                # Default to base model if just "qwen3-tts"
                "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
            )
            logger.info(f"Qwen3-TTS HuggingFace model: {qwen3_model_id}")

            try:
                logger.info(f"Initializing {model_class.__name__}...")
                logger.info(f"Model type: {model_type}")

                chatterbox_model = model_class.from_pretrained(
                    device="mlx",
                    model_id=qwen3_model_id,
                )

                loaded_model_type = model_type
                loaded_model_class_name = model_class.__name__

                logger.info(f"Successfully loaded {model_class.__name__}")
                logger.info(f"Model sample rate: {chatterbox_model.sr} Hz")
                logger.info(f"Model variant: {chatterbox_model._model_variant}")
            except Exception as e_qwen3:
                logger.error(
                    f"Failed to load Qwen3-TTS model: {e_qwen3}",
                    exc_info=True,
                )
                chatterbox_model = None
                MODEL_LOADED = False
                return False
        else:
            # Standard PyTorch-based models (Chatterbox original/turbo/multilingual)
            # Determine processing device with robust CUDA detection and intelligent fallback
            device_setting = config_manager.get_string("tts_engine.device", "auto")

            if device_setting == "auto":
                if TORCH_AVAILABLE and _test_cuda_functionality():
                    resolved_device_str = "cuda"
                    logger.info("CUDA functionality test passed. Using CUDA.")
                elif TORCH_AVAILABLE and _test_mps_functionality():
                    resolved_device_str = "mps"
                    logger.info("MPS functionality test passed. Using MPS.")
                else:
                    resolved_device_str = "cpu"
                    logger.info("CUDA and MPS not functional or not available. Using CPU.")

            elif device_setting == "cuda":
                if TORCH_AVAILABLE and _test_cuda_functionality():
                    resolved_device_str = "cuda"
                    logger.info("CUDA requested and functional. Using CUDA.")
                else:
                    resolved_device_str = "cpu"
                    logger.warning(
                        "CUDA was requested in config but functionality test failed. "
                        "PyTorch may not be compiled with CUDA support. "
                        "Automatically falling back to CPU."
                    )

            elif device_setting == "mps":
                if TORCH_AVAILABLE and _test_mps_functionality():
                    resolved_device_str = "mps"
                    logger.info("MPS requested and functional. Using MPS.")
                else:
                    resolved_device_str = "cpu"
                    logger.warning(
                        "MPS was requested in config but functionality test failed. "
                        "PyTorch may not be compiled with MPS support. "
                        "Automatically falling back to CPU."
                    )

            elif device_setting == "cpu":
                resolved_device_str = "cpu"
                logger.info("CPU device explicitly requested in config. Using CPU.")

            else:
                logger.warning(
                    f"Invalid device setting '{device_setting}' in config. "
                    f"Defaulting to auto-detection."
                )
                if TORCH_AVAILABLE and _test_cuda_functionality():
                    resolved_device_str = "cuda"
                elif TORCH_AVAILABLE and _test_mps_functionality():
                    resolved_device_str = "mps"
                else:
                    resolved_device_str = "cpu"
                logger.info(f"Auto-detection resolved to: {resolved_device_str}")

            model_device = resolved_device_str
            logger.info(f"Final device selection: {model_device}")

            try:
                logger.info(
                    f"Initializing {model_class.__name__} on device '{model_device}'..."
                )
                logger.info(f"Model type: {model_type}")
                if model_type == "turbo":
                    logger.info(
                        f"Turbo model supports paralinguistic tags: {TURBO_PARALINGUISTIC_TAGS}"
                    )

                # Load the model using from_pretrained - handles HuggingFace downloads automatically
                chatterbox_model = model_class.from_pretrained(device=model_device)

                # Store model metadata
                loaded_model_type = model_type
                loaded_model_class_name = model_class.__name__

                logger.info(f"Successfully loaded {model_class.__name__} on {model_device}")
                logger.info(f"Model sample rate: {chatterbox_model.sr} Hz")
            except ImportError as e_import:
                logger.error(
                    f"Failed to load model due to import error: {e_import}",
                    exc_info=True,
                )
                chatterbox_model = None
                MODEL_LOADED = False
                return False
            except Exception as e_hf:
                logger.error(
                    f"Failed to load model using from_pretrained: {e_hf}",
                    exc_info=True,
                )
                chatterbox_model = None
                MODEL_LOADED = False
                return False

        MODEL_LOADED = True
        if chatterbox_model:
            logger.info(
                f"TTS Model loaded successfully on {model_device}. Engine sample rate: {chatterbox_model.sr} Hz."
            )
        else:
            logger.error(
                "Model loading sequence completed, but chatterbox_model is None. This indicates an unexpected issue."
            )
            MODEL_LOADED = False
            return False

        return True

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during model loading: {e}", exc_info=True
        )
        chatterbox_model = None
        MODEL_LOADED = False
        return False


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
    # Qwen3-specific parameters
    speaker: Optional[str] = None,
    instruct: Optional[str] = None,
    ref_text: Optional[str] = None,
):
    """
    Synthesizes audio from text using the loaded TTS model.

    Args:
        text: The text to synthesize.
        audio_prompt_path: Path to an audio file for voice cloning or predefined voice.
        temperature: Controls randomness in generation.
        exaggeration: Controls expressiveness (Chatterbox only).
        cfg_weight: Classifier-Free Guidance weight (Chatterbox only).
        seed: Random seed for generation. If 0, default randomness is used.
              If non-zero, a global seed is set for reproducibility.
        language: Language code for multilingual/qwen3 model.
        speaker: Speaker name for Qwen3 CustomVoice model.
        instruct: Style/emotion instruction for Qwen3 models.
        ref_text: Transcript of reference audio for Qwen3 voice cloning.

    Returns:
        A tuple containing the audio waveform and the sample rate (int),
        or (None, None) if synthesis fails.
        Audio waveform is torch.Tensor for Chatterbox, numpy.ndarray for Qwen3.
    """
    global chatterbox_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        return None, None

    try:
        # Set seed globally if a specific seed value is provided and is non-zero.
        if seed != 0:
            logger.info(f"Applying user-provided seed for generation: {seed}")
            set_seed(seed)
        else:
            logger.info(
                "Using default (potentially random) generation behavior as seed is 0."
            )

        logger.debug(
            f"Synthesizing with params: audio_prompt='{audio_prompt_path}', temp={temperature}, "
            f"exag={exaggeration}, cfg_weight={cfg_weight}, seed_applied_globally_if_nonzero={seed}, "
            f"language={language}"
        )

        if loaded_model_type == "qwen3":
            # Qwen3-TTS returns (numpy_array, sample_rate)
            audio_np, sr = chatterbox_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                ref_text=ref_text,
                temperature=temperature,
                language=language,
                speaker=speaker,
                instruct=instruct,
            )
            return audio_np, sr

        # Call the core model's generate method
        # Multilingual model requires language_id parameter
        elif loaded_model_type == "multilingual":
            wav_tensor = chatterbox_model.generate(
                text=text,
                language_id=language,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
        else:
            wav_tensor = chatterbox_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        # The ChatterboxTTS.generate method already returns a CPU tensor.
        return wav_tensor, chatterbox_model.sr

    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        return None, None


def reload_model() -> bool:
    """
    Unloads the current model, clears GPU memory, and reloads the model
    based on the current configuration. Used for hot-swapping models
    without restarting the server process.

    Returns:
        bool: True if the new model loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name

    logger.info("Initiating model hot-swap/reload sequence...")

    # 1. Unload existing model
    if chatterbox_model is not None:
        logger.info("Unloading existing TTS model from memory...")
        del chatterbox_model
        chatterbox_model = None

    # 2. Reset state flags
    MODEL_LOADED = False
    loaded_model_type = None
    loaded_model_class_name = None

    # 3. Force Python Garbage Collection
    gc.collect()
    logger.info("Python garbage collection completed.")

    # 4. Clear GPU Cache (CUDA)
    if TORCH_AVAILABLE and torch.cuda.is_available():
        logger.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()

    # 5. Clear GPU Cache (MPS - Apple Silicon)
    if TORCH_AVAILABLE and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache.")
        except AttributeError:
            # Older PyTorch versions may not have mps.empty_cache()
            logger.debug(
                "torch.mps.empty_cache() not available in this PyTorch version."
            )

    # 6. Reload model from the (now updated) configuration
    logger.info("Memory cleared. Reloading model from updated config...")
    return load_model()


# --- End File: engine.py ---

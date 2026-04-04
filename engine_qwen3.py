# File: engine_qwen3.py
# Qwen3-TTS engine adapter for MLX (Apple Silicon).
# Wraps mlx-audio's Qwen3-TTS model to match the Chatterbox server interface.

import logging
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Qwen3 preset speakers by model variant
QWEN3_SPEAKERS_CUSTOM_VOICE = {
    "Ryan": "Dynamic male (English)",
    "Aiden": "Sunny American male (English)",
    "Ethan": "Male (English)",
    "Chelsie": "Female (English)",
    "Vivian": "Bright young female (Chinese)",
    "Serena": "Warm gentle young female (Chinese)",
    "Uncle_Fu": "Low mellow male (Chinese)",
    "Dylan": "Beijing dialect male (Chinese)",
    "Eric": "Sichuan dialect male (Chinese)",
    "Ono_Anna": "Playful female (Japanese)",
    "Sohee": "Warm female (Korean)",
}

QWEN3_SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

# Model variant types
QWEN3_MODEL_VARIANTS = {
    # 1.7B models
    "qwen3-tts-custom": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "qwen3-tts-base": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "qwen3-tts-voice-design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    # 0.6B models
    "qwen3-tts-custom-lite": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
    "qwen3-tts-base-lite": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    "qwen3-tts-voice-design-lite": "mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit",
}


class Qwen3TTSAdapter:
    """
    Adapter that wraps mlx-audio's Qwen3-TTS model to match the interface
    expected by the Chatterbox TTS Server (engine.py).
    """

    def __init__(self, model, model_variant: str, sample_rate: int = 24000):
        self._model = model
        self._model_variant = model_variant
        self.sr = sample_rate
        self.device = "mlx"  # MLX uses Apple Silicon unified memory

    @classmethod
    def from_pretrained(
        cls,
        device: str = "mlx",
        model_id: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    ) -> "Qwen3TTSAdapter":
        """
        Load a Qwen3-TTS model via mlx-audio.

        Args:
            device: Ignored for MLX (always uses Apple Silicon).
            model_id: HuggingFace model ID or local path.

        Returns:
            Qwen3TTSAdapter instance.
        """
        from mlx_audio.tts.utils import load

        logger.info(f"Loading Qwen3-TTS model: {model_id}")
        model = load(model_id)
        logger.info("Qwen3-TTS model loaded successfully.")

        # Determine variant type from model_id
        variant = "base"
        model_id_lower = model_id.lower()
        if "customvoice" in model_id_lower:
            variant = "custom"
        elif "voicedesign" in model_id_lower:
            variant = "voice_design"

        return cls(model=model, model_variant=variant)

    def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        temperature: float = 0.9,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        language: str = "auto",
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        speed: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech audio from text.

        Routes to the appropriate generation mode based on model variant
        and provided parameters.

        Returns:
            Tuple of (audio_numpy_array, sample_rate)
        """
        # Map language codes to full names for Qwen3
        lang_map = {
            "en": "English", "zh": "Chinese", "ja": "Japanese",
            "ko": "Korean", "de": "German", "fr": "French",
            "ru": "Russian", "pt": "Portuguese", "es": "Spanish",
            "it": "Italian", "auto": "auto",
        }
        lang_code = lang_map.get(language, language)

        gen_kwargs = dict(
            text=text,
            temperature=temperature,
            speed=speed,
            lang_code=lang_code,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            verbose=False,
            split_pattern=r"(?<=[.!?])\s+",  # Split on sentence boundaries for reliable generation
        )

        # Route based on variant and parameters
        if audio_prompt_path and self._model_variant == "base":
            # Voice cloning mode (Base model)
            logger.info(f"Qwen3-TTS: Voice cloning with ref_audio={audio_prompt_path}")
            gen_kwargs["ref_audio"] = audio_prompt_path
            if ref_text:
                gen_kwargs["ref_text"] = ref_text
            # ICL mode uses slightly elevated repetition penalty to prevent code degeneration
            # but not too high — 1.5 causes early EOS on long prose
            gen_kwargs["repetition_penalty"] = max(repetition_penalty, 1.2)

        elif speaker and self._model_variant == "custom":
            # Custom voice mode (preset speakers)
            logger.info(f"Qwen3-TTS: Custom voice with speaker={speaker}")
            gen_kwargs["voice"] = speaker
            if instruct:
                gen_kwargs["instruct"] = instruct

        elif instruct and self._model_variant == "voice_design":
            # Voice design mode
            logger.info(f"Qwen3-TTS: Voice design with instruct={instruct}")
            gen_kwargs["instruct"] = instruct

        elif audio_prompt_path:
            # Fallback: if ref audio provided but not base model, still try
            logger.info(f"Qwen3-TTS: Attempting voice cloning with ref_audio={audio_prompt_path}")
            gen_kwargs["ref_audio"] = audio_prompt_path
            if ref_text:
                gen_kwargs["ref_text"] = ref_text
            gen_kwargs["repetition_penalty"] = max(repetition_penalty, 1.5)

        else:
            # Default: use first available speaker for custom, or no special mode
            if self._model_variant == "custom":
                gen_kwargs["voice"] = speaker or "Chelsie"
                logger.info(f"Qwen3-TTS: Custom voice defaulting to {gen_kwargs['voice']}")

        # Generate audio via the model's generator
        audio_segments = []
        sample_rate = self.sr

        for result in self._model.generate(**gen_kwargs):
            audio_np = np.array(result.audio)
            audio_segments.append(audio_np)
            sample_rate = result.sample_rate
            token_count = getattr(result, 'token_count', None)
            logger.info(
                f"Qwen3-TTS segment: {len(audio_np)} samples "
                f"({len(audio_np)/sample_rate:.2f}s), "
                f"tokens={token_count}, max_tokens={max_tokens}"
            )
            if token_count and token_count >= max_tokens - 1:
                logger.warning(
                    f"Qwen3-TTS hit max_tokens limit ({token_count}/{max_tokens})! "
                    f"Audio may be truncated. Consider smaller chunks or higher max_tokens."
                )

        if not audio_segments:
            raise RuntimeError("Qwen3-TTS generated no audio segments")

        # Concatenate all segments
        full_audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]

        logger.info(
            f"Qwen3-TTS generated {len(full_audio)} samples at {sample_rate}Hz "
            f"({len(full_audio)/sample_rate:.2f}s), "
            f"text length={len(text)} chars"
        )

        return full_audio, sample_rate

    def get_speakers(self) -> Dict[str, str]:
        """Return available speakers for CustomVoice model."""
        if self._model_variant == "custom":
            return QWEN3_SPEAKERS_CUSTOM_VOICE
        return {}

    def get_supported_languages(self) -> Dict[str, str]:
        """Return supported languages."""
        return QWEN3_SUPPORTED_LANGUAGES.copy()

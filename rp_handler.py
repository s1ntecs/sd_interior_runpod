import base64
import io
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

# --------------------------------------------------------------------------- #
#                               КОНСТАНТЫ                                     #
# --------------------------------------------------------------------------- #
MAX_SEED: int = np.iinfo(np.int32).max
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_STEPS: int = 25
DEFAULT_SEED: int = 42

LORA_DIR = "./loras"
LORA_LIST = [
    "XSArchi_110plan彩总.safetensors",
    "XSArchi_137.safetensors",
    "XSArchi_141.safetensors",
    "XSArchi_162BIESHU.safetensors",
    "XSarchitectural-38InteriorForBedroom.safetensors",
    "XSarchitectural_33WoodenluxurystyleV2.safetensors",
    "house_architecture_Exterior_SDlife_Chiasedamme.safetensors",
    "xsarchitectural-15Nightatmospherearchitecture.safetensors",
    "xsarchitectural-18Whiteexquisiteinterior.safetensors",
    "xsarchitectural-19Houseplan (1).safetensors",
    "xsarchitectural-19Houseplan.safetensors",
    "xsarchitectural-7.safetensors",
]

DEFAULT_MODEL = "checkpoints/xsarchitectural_v10interiordesignforxs.ckpt"
logger = RunPodLogger()


# --------------------------------------------------------------------------- #
#                               ЗАГРУЗКА МОДЕЛИ                               #
# --------------------------------------------------------------------------- #
def load_model_base(model_path: str) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_single_file(
        pretrained_model_link_or_path=model_path,
        torch_dtype=DTYPE,
        local_files_only=True
    )
    # важно: переносим и dtype, и устройство
    pipe = pipe.to(device=DEVICE, dtype=DTYPE)
    return pipe


PIPELINE = load_model_base(DEFAULT_MODEL)

PIPELINE.scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear",
    algorithm_type="dpmsolver++",
    solver_order=2,
    use_karras_sigmas=True
)

CURRENT_LORA: str = "None"
print("✅ Pipelines ready.")


# --------------------------------------------------------------------------- #
#                           LOADING / UNLOADING LoRA                          #
# --------------------------------------------------------------------------- #
def _switch_lora(lora_name: Optional[str]) -> Optional[str]:
    """Load new LoRA or unload if lora_name is None. Return error str or None."""
    global CURRENT_LORA

    # -------- unload current LoRA -------- #
    if lora_name is None and CURRENT_LORA != "None":
        if hasattr(PIPELINE, "unfuse_lora"):
            PIPELINE.unfuse_lora()
        if hasattr(PIPELINE, "unload_lora_weights"):
            PIPELINE.unload_lora_weights()
        CURRENT_LORA = "None"
        return None

    # ----- nothing to do / unsupported --- #
    if lora_name is None or lora_name == CURRENT_LORA:
        return None
    if lora_name not in LORA_LIST:
        return f"Unknown LoRA '{lora_name}'."

    # --------- load new LoRA ------------- #
    try:
        if CURRENT_LORA != "None":
            if hasattr(PIPELINE, "unfuse_lora"):
                PIPELINE.unfuse_lora()
            if hasattr(PIPELINE, "unload_lora_weights"):
                PIPELINE.unload_lora_weights()

        PIPELINE.load_lora_weights(f"{LORA_DIR}/{lora_name}")
        if hasattr(PIPELINE, "fuse_lora"):
            PIPELINE.fuse_lora()

        CURRENT_LORA = lora_name
        return None
    except Exception as err:  # noqa: BLE001
        return f"Failed to load LoRA '{lora_name}': {err}"


# --------------------------------------------------------------------------- #
#                                ВСПОМОГАТЕЛЬНЫЕ                              #
# --------------------------------------------------------------------------- #
def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# --------------------------------------------------------------------------- #
#                                HANDLER                                      #
# --------------------------------------------------------------------------- #
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler."""
    try:
        payload: Dict[str, Any] = job.get("input", {})
        prompt = payload.get("prompt")
        if not prompt:
            return {"error": "'prompt' is required"}

        # ----------------- handle LoRA ----------------- #
        error = _switch_lora(payload.get("lora"))
        if error:
            return {"error": error}

        # ----------------- parameters ------------------ #
        num_images = int(payload.get("num_images", 1))
        if num_images < 1 or num_images > 8:
            return {"error": "'num_images' must be between 1 and 8."}

        guidance_scale = float(payload.get("guidance_scale", 7.5))
        steps = min(int(payload.get("steps", MAX_STEPS)), MAX_STEPS)
        seed = int(payload.get("seed", DEFAULT_SEED))
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        height = int(payload.get("height", 768))
        width = int(payload.get("width", 1024))

        if height <= 0 or width <= 0:
            return {"error": "'height' and 'width' must be positive integers."}

        start = time.time()

        # ---------------- generation ------------------- #
        images: List[Image.Image] = PIPELINE(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
            num_images_per_prompt=num_images,
            height=height,
            width=width,
        ).images

        elapsed = round(time.time() - start, 2)

        return {
            "images_base64": [pil_to_b64(img) for img in images],
            "time": elapsed,
            "steps": steps,
            "seed": seed,
            "lora": CURRENT_LORA if CURRENT_LORA != "None" else None,
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "CUDA out of memory" in str(exc):
            return {"error": "CUDA out of memory — reduce 'steps' or image size."}
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc(limit=5)}


# --------------------------------------------------------------------------- #
#                               RUN WORKER                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

# download_checkpoints.py  (offline-build)

import os
import torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_download, snapshot_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEFAULT_MODEL = "checkpoints/xsarchitectural_v11.ckpt"
LORA_NAMES = [
    # (тот же список, что выше)
]

# ------------------------- загрузка весов -------------------------
def fetch_checkpoints() -> None:
    """Скачиваем SD-чекпойнт, LoRA-файлы и все внешние зависимости."""
    # 1) основной .ckpt и LoRA
    hf_hub_download(
        repo_id="sintecs/interior",
        filename="xsarchitectural_v11.ckpt",
        local_dir="checkpoints",
        local_dir_use_symlinks=False,
    )
    for fname in LORA_NAMES:
        hf_hub_download(
            repo_id="sintecs/interior",
            filename=fname,
            local_dir="loras",
            local_dir_use_symlinks=False,
        )

    # 2) text-encoder + токенайзер
    snapshot_download(
        repo_id="openai/clip-vit-large-patch14",
        local_dir_use_symlinks=False,
    )

    # 3) (опционально) safety-checker – если нужен
    snapshot_download(
        repo_id="CompVis/stable-diffusion-safety-checker",
        local_dir_use_symlinks=False,
    )


# ------------------------- пайплайн -------------------------
def get_pipeline():
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
        algorithm_type="dpmsolver++",
        solver_order=2,
        use_karras_sigmas=True,
    )

    pipe = StableDiffusionPipeline.from_single_file(
        pretrained_model_link_or_path=DEFAULT_MODEL,
        torch_dtype=torch.float16,
        scheduler=scheduler,
        local_files_only=True,          # ← строго оффлайн
        # load_safety_checker=False,    # если safety-checker не нужен
    )
    return pipe


if __name__ == "__main__":
    fetch_checkpoints()
    get_pipeline()
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from huggingface_hub import hf_hub_download

os.makedirs("./loras", exist_ok=True)
os.makedirs("./checkpoints", exist_ok=True)

DEFAULT_MODEL = "checkpoints/xsarchitectural_v11.ckpt"

lora_names = [
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
    "xsarchitectural-7.safetensors"
]


def fetch_checkpoints():
    """
    Fetches SD checkpoints from the HuggingFace model hub.
    """

    hf_hub_download(
        repo_id='sintecs/interior',
        filename='xsarchitectural_v11.ckpt',
        local_dir='./checkpoints',
        local_dir_use_symlinks=False
    )

    # DL Loras
    for fname in lora_names:
        hf_hub_download(
            repo_id="sintecs/interior",
            filename=fname,
            local_dir="./loras",
            local_dir_use_symlinks=False
        )


def get_pipeline():
    """
    Fetches the pipeline from the HuggingFace model hub.
    """
    torch_dtype = torch.float16

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        algorithm_type="dpmsolver++",
        solver_order=2,
        use_karras_sigmas=True
    )

    pipe = StableDiffusionPipeline.from_single_file(
        pretrained_model_link_or_path=DEFAULT_MODEL,
        torch_dtype=torch_dtype,
        local_files_only=True,
        scheduler=scheduler
    )

    return pipe


if __name__ == '__main__':
    fetch_checkpoints()
    get_pipeline()

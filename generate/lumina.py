import torch
from diffusers import LuminaText2ImgPipeline


class LuminaPipeline():
    def __init__(self, model="Alpha-VLLM/Lumina-Next-SFT-diffusers", device="cuda", *args, **kwargs):
        pipeline = LuminaText2ImgPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.pipeline = pipeline

    def __call__(self, prompt, *args, **kwargs):
        return self.pipeline(prompt=prompt, *args, **kwargs)


if __name__ == "__main__":
    id = "Alpha-VLLM/Lumina-Next-SFT-diffusers"
    pipeline = LuminaPipeline(id)

    while True:
        prompt = input("input prompt：")
        image = pipeline(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=4.0,
            width=1024,
            height=1024,
        ).images[0]
        filename = "_".join(prompt.split(" "))
        image.save(f"diffusers_{filename}.png")

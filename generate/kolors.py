import torch
from diffusers import KolorsPipeline


class KLPipeline():
    def __init__(self, model="Kwai-Kolors/Kolors-diffusers", device="cuda", *args, **kwargs):
        pipeline = KolorsPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
        self.pipeline = pipeline

    def __call__(self, prompt, *args, **kwargs):
        return self.pipeline(prompt=prompt, *args, **kwargs)


if __name__ == "__main__":
    id = "Kwai-Kolors/Kolors-diffusers"
    pipeline = KLPipeline(id)

    while True:
        prompt = input("input prompt：")
        image = pipeline(
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=50,
            guidance_scale=5.0,
            generator=torch.manual_seed(66),
        ).images[0]
        filename = "_".join(prompt.split(" "))
        image.save(f"diffusers_{filename}.png")

from diffusers import FluxPipeline
import torch


class FLUXPipeline():

    def __init__(self, model="black-forest-labs/FLUX.1-dev", device="cuda", *args, **kwargs):
        pipeline = FluxPipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to(device)
        self.pipeline = pipeline

    def __call__(self, prompt, *args, **kwargs):
        return self.pipeline(prompt=prompt, *args, **kwargs)


if __name__ == "__main__":
    id = "black-forest-labs/FLUX.1-dev"
    pipeline = FLUXPipeline(id)

    while True:
        prompt = input("input prompt：")
        image = pipeline(
            prompt=prompt,
            guidance_scale=3.5,
            height=1024,
            width=1024,
            num_inference_steps=50,
        ).images[0]
        filename = "_".join(prompt.split(" "))
        image.save(f"diffusers_{filename}.png")

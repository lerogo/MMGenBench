from diffusers import StableDiffusion3Pipeline
import torch


class SD3Pipeline():
    torch.set_float32_matmul_precision("high")

    def __init__(self, model="stabilityai/stable-diffusion-3-medium", device="cuda", *args, **kwargs):
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model,
            torch_dtype=torch.float16
        ).to(device)
        pipeline.set_progress_bar_config(disable=True)

        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True

        pipeline.transformer.to(memory_format=torch.channels_last)
        pipeline.vae.to(memory_format=torch.channels_last)
        pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)
        pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

        # warmup
        prompt = "a photo of a cat"
        for _ in range(3):
            _ = pipeline(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=5.0,
                generator=torch.manual_seed(1),
            )
        self.pipeline = pipeline

    def __call__(self, prompt, *args, **kwargs):
        return self.pipeline(prompt=prompt, *args, **kwargs)


class SD3PipelineNormal():
    def __init__(self, model="stabilityai/stable-diffusion-3-medium", device="cuda", *args, **kwargs):
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model,
            torch_dtype=torch.float16
        ).to(device)
        self.pipeline = pipeline

    def __call__(self, prompt, *args, **kwargs):
        return self.pipeline(prompt=prompt, *args, **kwargs)


class SD35Pipeline():
    def __init__(self, model="stabilityai/stable-diffusion-3.5-large", device="cuda", *args, **kwargs):
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16
        ).to(device)
        self.pipeline = pipeline

    def __call__(self, prompt, *args, **kwargs):
        return self.pipeline(prompt=prompt, *args, **kwargs)


if __name__ == "__main__":
    id = "stabilityai/stable-diffusion-3.5-large"
    pipeline = SD35Pipeline(id)

    while True:
        prompt = input("input prompt：")
        image = pipeline(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=5.0,
            generator=torch.manual_seed(1),
        ).images[0]
        filename = "_".join(prompt.split(" "))
        image.save(f"diffusers_{filename}.png")

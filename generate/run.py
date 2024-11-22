import base64
import datetime
import io
import os
import argparse
import torch
import torch.distributed
from tqdm import tqdm
from functools import partial
import pandas as pd
from tools import pre_caption
from sd import SD35Pipeline
from flux import FLUXPipeline
from kolors import KLPipeline
from lumina import LuminaPipeline
import warnings

modelDict = {
    "sd3-5": partial(SD35Pipeline, model="stabilityai/stable-diffusion-3.5-large"),
    "flux-1-dev": partial(FLUXPipeline, model="black-forest-labs/FLUX.1-dev"),
    "kolors": partial(KLPipeline, model="Kwai-Kolors/Kolors-diffusers"),
    "lumina": partial(LuminaPipeline, model="Alpha-VLLM/Lumina-Next-SFT-diffusers"),
}
modelArgs = {
    "sd3-5": {"guidance_scale": 3.5, "num_inference_steps": 28, "height": 1024, "width": 1024},
    "flux-1-dev": {"guidance_scale": 3.5, "num_inference_steps": 50, "height": 1024, "width": 1024},
    "kolors": {"guidance_scale": 5.0, "num_inference_steps": 50, "negative_prompt": "", "height": 1024, "width": 1024, "generator": torch.manual_seed(66)},
    "lumina": {"guidance_scale": 4.0, "num_inference_steps": 30, "height": 1024, "width": 1024},
}


def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size


def main(args):
    rank, worldsize = get_rank_and_world_size()
    if rank == 0:
        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)
    if worldsize > 1:
        torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10800))

    torch.cuda.set_device(rank % worldsize)
    args.device = torch.device(rank % worldsize)

    m = args.model
    if m not in modelDict:
        raise ValueError(f"Model {m} not found.")

    pipeline = modelDict[m](device=args.device)

    for f in args.data:
        key = f"{f.split('/')[-1].split('.xlsx')[0]}"
        if os.path.exists(os.path.join(args.work_dir, f"{key}.tsv")):
            warnings.warn(f"Skip {key} as it already exists.")
            continue

        print(f"Processing {f} with {m}...")
        if not os.path.exists(f):
            warnings.warn(f"Data file {f} not found.")
            continue
        # load data
        if worldsize > 1:
            data = pd.read_excel(f) if rank == 0 else None
            datalist = [data]
            torch.distributed.barrier()
            torch.distributed.broadcast_object_list(datalist, src=0)
            data = datalist[0]
        else:
            data = pd.read_excel(f)

        # Now there are different data on multiple GPUs
        rankRes = infer_data(pipeline, data, modelArgs[m])
        if worldsize > 1:
            rankResList = [None] * worldsize
            torch.distributed.barrier()
            torch.distributed.gather_object(rankRes, rankResList if rank == 0 else None, dst=0)
            tmpRes = []
            if rank == 0:
                for r in rankResList:
                    tmpRes.extend(r)
        else:
            tmpRes = rankRes
        tmpRes.sort(key=lambda x: x[0])

        if rank == 0:
            with open(os.path.join(args.work_dir, f"{key}.tsv"), "w") as fw:
                fw.write("index\timage\n")
                for i, img in tmpRes:
                    fw.write(f"{i}\t{img}\n")
        if worldsize > 1:
            torch.distributed.barrier()


def infer_data(pipeline, data, pipeline_args):
    rank, worldsize = torch.distributed.get_rank(), torch.distributed.get_world_size()
    res = []
    predictionList = data["prediction"].tolist()
    indexList = [i for i in range(len(predictionList))]
    if indexList != data["index"].tolist():
        warnings.warn("!!! Index not match, please check the data file.")

    indexList = torch.tensor_split(torch.tensor(indexList), worldsize)[rank].tolist()
    for i in tqdm(indexList):
        predictionList[i] = str(predictionList[i])
        prediction = predictionList[i].strip("\"")
        image = pipeline(prompt=pre_caption(prediction), **pipeline_args).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        res.append((i, img_str))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, default='flux-1-dev', required=True)
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    parser.add_argument('--device', type=str, default='cuda', help='select the device to run the model')
    args = parser.parse_args()
    main(args)

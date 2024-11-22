import argparse
import io
import json
import os
import pandas as pd
import tqdm
import unicom
import torch
import torch.nn.functional as F
from PIL import Image
import base64
import numpy as np
from metric_fid import fid_from_feats
import warnings


def main(args):
    evalimg = EvalImg(model_name=args.model, device=args.device)
    ori_data = pd.read_csv(args.ori_data, sep='\t')
    ori_data_index = [x for x in ori_data['index']]
    for data_key in args.gen_data:
        print(f"Start to evaluate {data_key}")

        output_path = os.path.join(args.work_dir, os.path.basename(data_key).replace(".tsv", ".json"))
        if os.path.exists(output_path):
            print(f"Skip {data_key}")
            continue

        gen_data = pd.read_csv(data_key, sep='\t')
        gen_data_index = [x for x in gen_data['index']]
        
        if ori_data_index != gen_data_index:
            warnings.warn(f"Error: {data_key} and {args.ori_data} have different index")
            continue

        fid, sim = evalimg(ori_data, gen_data, eval_type=args.eval)
        res = {
            "sim": {
                str(i): sim[i].item() for i in ori_data_index
            },
            "fid": fid
        }

        with open(output_path, 'w') as f:
            json.dump(res, f, indent=4)


class EvalImg(object):
    def __init__(self, model_name="ViT-L/14@336px", device="cuda"):
        model, preprocess = unicom.load(model_name, download_root="./models")
        model.eval()
        model = model.to(device)
        self.model, self.preprocess, self.device = model, preprocess, device
        self.ori_features = None

    @torch.no_grad()
    def get_feature(self, pil_image):
        img = pil_image
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        imgFea = self.model(img)
        return imgFea

    def base64_to_pil_image(self, base64_string, target_size=-1):
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        if target_size > 0:
            image.thumbnail((target_size, target_size))
        return image

    def get_similarity(self, a, b):
        sim = F.cosine_similarity(a, b, dim=1)
        return sim

    def get_fid(self, a, b):
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        fid = fid_from_feats(a, b)
        return fid

    def __call__(self, ori_data, gen_data, eval_type="sim", *args, **kwds):
        if self.ori_features is None:
            ori_features = [self.get_feature(self.base64_to_pil_image(x)) for x in tqdm.tqdm(ori_data['image'])]
            self.ori_features = ori_features
        else:
            ori_features = self.ori_features
        gen_features = [self.get_feature(self.base64_to_pil_image(x)) for x in tqdm.tqdm(gen_data['image'])]

        ori_features = torch.cat(ori_features, dim=0)
        gen_features = torch.cat(gen_features, dim=0)
        if eval_type == "sim":
            sim = self.get_similarity(ori_features, gen_features)
            return None, sim
        elif eval_type == "fid":
            fid = self.get_fid(gen_features, ori_features)
            return fid, None
        elif eval_type == "fid-sim":
            fid = self.get_fid(gen_features, ori_features)
            sim = self.get_similarity(ori_features, gen_features)
            return fid, sim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_data', type=str, required=True)
    parser.add_argument('--gen_data', type=str, nargs="+", required=True)
    parser.add_argument('--model', type=str, default="ViT-L/14@336px", help='select the model to run')
    parser.add_argument('--eval', type=str, choices=["sim", "fid", "fid-sim"], default="sim", help='select the evaluation method')
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    parser.add_argument('--device', type=str, default='cuda', help='select the device to run the model')
    args = parser.parse_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    main(args)

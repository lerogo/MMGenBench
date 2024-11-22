ori_data="../MMGenBench-data/MMGenBench-Test.tsv"
# ori_data="../MMGenBench-data/MMGenBench-Domain.tsv"

evalmertic="fid-sim"
# model="ViT-B/32"
model="ViT-L/14@336px"

readPath="output"

dataPath="../generate/${readPath}/*Test.tsv"
# dataPath="../generate/${readPath}/*Domain.tsv"
dataList=($(ls ${dataPath}))

CUDA_VISIBLE_DEVICES="0" python -u run.py --model ${model} --eval ${evalmertic} \
    --work-dir "${readPath}" \
    --ori_data ${ori_data} \
    --gen_data ${dataList[@]}

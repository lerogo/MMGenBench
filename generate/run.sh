timestamp=$(date "+%Y-%m-%d %H:%M:%S")

# sd3-5 flux-1-dev lumina kolors
modelName="flux-1-dev"
MASTER_PORT="20500"
GPUS=1

dataPath="./input/*.xlsx"
dataList=($(ls ${dataPath}))

echo "timestamp: ${timestamp}"
echo "modelName: ${modelName}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "GPUS: ${GPUS}"

echo "dataList: ${dataList[@]}"

# torchrun --nproc-per-node=${GPUS} --master-port ${MASTER_PORT} run.py --model ${modelName} --work-dir "./output_${modelName}" --data ${dataList[@]}
torchrun --nproc-per-node=${GPUS} --master-port ${MASTER_PORT} run.py --model ${modelName} --work-dir "./output" --data ${dataList[@]}

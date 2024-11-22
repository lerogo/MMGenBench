python -u run.py --prompt_data "../generate/input/InternVL2-2B_MMGenBench-Test.xlsx" \
    --gen_data "../generate/output/InternVL2-2B_MMGenBench-Test.tsv" \
    --eval_data "../evalimg/output/InternVL2-2B_MMGenBench-Test.json" \
    --data_dir "../MMGenBench-data" \
    --work-dir "./outputs" --type "test"

python -u run.py --prompt_data "../generate/input/InternVL2-2B_MMGenBench-Domain.xlsx" \
    --gen_data "../generate/output/InternVL2-2B_MMGenBench-Domain.tsv" \
    --eval_data "../evalimg/output/InternVL2-2B_MMGenBench-Domain.json" \
    --data_dir "../MMGenBench-data" \
    --work-dir "./outputs" --type "domain"

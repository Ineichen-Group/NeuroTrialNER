#!/bin/bash
#SBATCH --job-name=model_train
#SBATCH --output=%A_%a.out  # %A is job ID, %a is array index
#SBATCH --error=%A_%a.err
#SBATCH --gres=gpu:1  # Reserve 1 GPU per job
#SBATCH --array=0-2  # Three jobs for three models
#SBATCH --time=04:00:00

model_names=("dmis-lab/biobert-v1.1" "bert-base-uncased" "michiyasunaga/BioLinkBERT-base")
percentages=(100)
i_values=(1 2 3 4 5)

# Select the model based on the job array index
model_name=${model_names[$SLURM_ARRAY_TASK_ID]}

# Loop over each percentage and i value sequentially
for percentage in "${percentages[@]}"; do
    for i in "${i_values[@]}"; do
        echo "Running experiment with model=$model_name, percentage=$percentage, and i=$i"
        python train_script.py \
            --output_path "../clinical_trials_out/arr_rebuttal" \
            --model_name_or_path "$model_name" \
            --train_data_path "./data/ct_neuro_train_data_787.json" \
            --val_data_path "./data/ct_neuro_dev_data_153.json" \
            --test_data_path "./data/ct_neuro_test_data_153.json" \
            --n_epochs 15 --percentage "$percentage" --i "$i"
    done
done

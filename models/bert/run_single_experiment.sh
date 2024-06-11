#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=06:00:00

# Define the list of percentage values and i values
percentages=(80)
i_values=(3 4 5)

# Loop over percentage and i values
for percentage in "${percentages[@]}"; do
    for i in "${i_values[@]}"; do
        echo "Running experiment with percentage=$percentage and i=$i"
        python train_script.py \
            --output_path "../clinical_trials_out/arr_rebuttal/train_size_impact" \
            --model_name_or_path "dmis-lab/biobert-v1.1" \
            --train_data_path "./data/ct_neuro_train_data_787.json" \
            --val_data_path "./data/ct_neuro_dev_data_153.json" \
            --test_data_path "./data/ct_neuro_test_data_153.json" \
            --n_epochs 15 --percentage "$percentage" --i "$i"
    done
done

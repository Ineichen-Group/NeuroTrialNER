#!/bin/bash

# Define the list of percentage values and i values
percentages=(100)
i_values=(5)

# Loop over percentage and i values
for percentage in "${percentages[@]}"; do
    for i in "${i_values[@]}"; do
        echo "Running experiment with percentage=$percentage and i=$i"
        python train_script.py --output_path "../clinical_trials_out" \
            --model_name_or_path "dmis-lab/biobert-v1.1" \
            --train_data_path "./data/ct_neuro_train_data_713.json" \
            --val_data_path "./data/ct_neuro_dev_data_90.json" \
            --test_data_path "./data/ct_neuro_test_data_90.json" \
            --n_epochs 15 --percentage $percentage --i $i
    done
done
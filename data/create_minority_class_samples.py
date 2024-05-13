import pandas as pd

def sample_data(input_csv, output_csv):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Filter out rows with intervention_type = 'Procedure', 'Radiation', 'Other', and 'Behavioural'
    filtered_df = df[df['intervention_type'].isin(['Procedure', 'Radiation', 'Other', 'Behavioral'])]

    # Create a new DataFrame to store the sampled rows
    sampled_df = pd.DataFrame(columns=['nct_id', 'intervention_type'])

    # Set to store sampled nct_ids
    sampled_nct_ids_set = set()

    # Iterate over unique intervention_types
    for intervention_type in filtered_df['intervention_type'].unique():
        intervention_type_rows = filtered_df[filtered_df['intervention_type'] == intervention_type]
        # Drop duplicate nct_ids for the current intervention_type
        unique_rows = intervention_type_rows.drop_duplicates(subset=['nct_id'])
        # Sample unique nct_ids for the current intervention_type
        sampled_nct_ids = unique_rows['nct_id'].sample(n=15, random_state=42)
        # Exclude previously sampled nct_ids
        sampled_nct_ids = sampled_nct_ids[~sampled_nct_ids.isin(sampled_nct_ids_set)]
        # Add sampled nct_ids to the set
        sampled_nct_ids_set.update(sampled_nct_ids)
        # Filter rows based on the sampled nct_ids
        sampled_rows = unique_rows[unique_rows['nct_id'].isin(sampled_nct_ids)]
        # Concatenate to the sampled DataFrame
        sampled_df = pd.concat([sampled_df, sampled_rows[['nct_id', 'intervention_type']]])

    # Reset the index of the sampled DataFrame
    sampled_df.reset_index(drop=True, inplace=True)

    # Write the sampled DataFrame to a new CSV file
    sampled_df.to_csv(output_csv, index=False)

    print("New unique nct_ids: ", len(set(sampled_df['nct_id'])))

if __name__ == "__main__":
    input_csv = './data_aact_sample/aact_neuro_samples_20240513_interventional_with_type.csv'  # Change to your input CSV file path
    output_csv = './data_aact_sample/aact_neuro_samples_20240513_interventional_with_type_stratified.csv'   # Change to your output CSV file path
    sample_data(input_csv, output_csv)
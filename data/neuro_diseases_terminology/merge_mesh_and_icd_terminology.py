import pandas as pd
import datetime


def merge_disease_terms_mesh_icd(mesh_file, icd_file):
    # Read the first file
    df1 = pd.read_csv(mesh_file)
    # Add 'MeSH' prefix to column names
    df1.columns = ['MeSH ' + column if column != 'Mesh ID' else column for column in df1.columns]
    df1['MeSH Common name Key'] = df1['MeSH Common name'].str.lower()

    # Read the second file
    df2 = pd.read_csv(icd_file)
    df2.columns = ['ICD ' + column for column in df2.columns]
    df2['ICD Title Key'] = df2['ICD Title'].str.lower()

    # Perform an outer join based on 'Common name' and 'Title'
    merged_df = pd.merge(df1, df2, left_on='MeSH Common name Key', right_on='ICD Title Key', how='outer')
    merged_df = merged_df.loc[:,
                ['ICD Node URI', 'ICD Parent URI', 'Mesh ID', 'MeSH Tree Number', 'ICD Title', 'MeSH Common name',
                 'MeSH Disease Class', 'ICD Disease Class', 'MeSH Synonyms']]
    # Remove duplicate rows based on 'ICD Title' and 'MeSH Common name'
    merged_df = merged_df.drop_duplicates(subset=['ICD Title', 'MeSH Common name'], keep='first')

    return merged_df

def flatten_diseases_list(mesh_output_path, icd_file_path, flattened_output_file_path):
    # Read the file
    df = pd.read_csv(mesh_output_path)
    df2 = pd.read_csv(icd_file_path)
    titles = df2['Title'].apply(lambda x: [x]).tolist()

    # Add source column for each DataFrame
    df['Source'] = 'MeSH'
    df2['Source'] = 'ICD'

    # Split the flattened string on | delimiter and create separate rows
    flattened_rows = []
    unique_elements = set()  # Set to keep track of unique elements

    for index, row in df.iterrows():
        flattened = row['Common name'] + '|' + row['Synonyms']
        elements = flattened.split('|')
        for element in elements:
            element = [element.strip().replace("\"", ''), row['Source'], row['Disease Class']]
            element_tuple = tuple(element)  # Convert to tuple for hashability check
            if element_tuple not in unique_elements:  # Check if element is already in the set
                flattened_rows.append(element)
                unique_elements.add(element_tuple)

    for title in titles:
        flattened_rows.append([title[0], df2['Source'].iloc[0], df2['Disease Class'].iloc[0]])

    # Create a new DataFrame with the flattened rows
    flattened_df = pd.DataFrame(flattened_rows, columns=['Neurological Disease', 'Source', 'Disease Class'])

    # Combine the "Source" values for duplicate rows
    flattened_df = flattened_df.groupby('Neurological Disease').agg({'Source': set, 'Disease Class': set}).reset_index()
    # Convert the sets back to strings, separated by '|'
    flattened_df['Source'] = flattened_df['Source'].apply('|'.join)
    flattened_df['Disease Class'] = flattened_df['Disease Class'].apply('|'.join)

    # Save the flattened data to a new file
    flattened_df.to_csv(flattened_output_file_path, index=False)


if __name__ == '__main__':
    # TODO: evtl. include further terms from https://ctdbase.org/downloads/#alldiseases ???

    mesh_output_path = "./output/diseases_dictionary_mesh.csv"
    icd_file_path = "./output/icd_11_neurological_diseases.csv"
    merged_df = merge_disease_terms_mesh_icd(mesh_output_path, icd_file_path)

    # Save the merged DataFrame
    merged_df.to_csv('./output/diseases_dictionary_mesh_icd.csv', index=False)

    flattened_output_file_path = './output/diseases_dictionary_mesh_icd_flat.csv'
    flatten_diseases_list(mesh_output_path, icd_file_path, flattened_output_file_path)


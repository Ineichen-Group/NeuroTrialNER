import csv
import xml.etree.ElementTree as ET
import pandas as pd
import re

disease_variant_to_canonical = {}


def load_mesh_neurology_list(mesh_neurology_hierarchy_terms_file):
    df = pd.read_excel(mesh_neurology_hierarchy_terms_file)
    df['Combined'] = df.apply(lambda row: ' '.join(str(cell) for cell in row), axis=1)

    # Extract items in brackets
    bracket_items_df = df['Combined'].str.extractall(r'(.+?)\s*\[(.*?)\]')
    bracket_items = set(df['Combined'].str.extractall(r'\[(.*?)\]')[0].tolist())

    # Create dictionary from extracted items
    dictionary = {}
    for row in bracket_items_df.iterrows():
        value, key = row[1]
        dictionary[key] = value

    return list(bracket_items), dictionary


def parse_elements_from_mesh_xml(mesh_neurology_hierarchy_terms_file, mesh_file_path, mesh_output_path):
    tree = ET.parse(mesh_file_path)
    root = tree.getroot()
    data_list = []
    mesh_ids_list, mesh_dictionary = load_mesh_neurology_list(mesh_neurology_hierarchy_terms_file)

    # Iterate over DescriptorRecords
    for descriptor_record in root.findall('DescriptorRecord'):
        descriptor_ui = descriptor_record.find('DescriptorUI').text

        # Check if DescriptorUI starts with 'D'
        if descriptor_ui.startswith('D'):
            descriptor_name = descriptor_record.find('DescriptorName/String').text

            term_list = descriptor_record.find('ConceptList/Concept/TermList')
            synonyms = [term.find('String').text for term in term_list.findall('Term')]

            tree_number_list = descriptor_record.find('TreeNumberList')
            if tree_number_list:
                tree_numbers = [term.text for term in tree_number_list.findall('TreeNumber')]
            else:
                tree_numbers = []

            matching_prefixes = set(tree_numbers) & set(mesh_ids_list)

            if matching_prefixes:
                matching_prefixes_list = list(matching_prefixes)
                match = re.search(r'([A-Z]+\d+\.\d+)', matching_prefixes_list[0])
                if match:
                    prefix_tree = match.group(1)
                prefix_value = mesh_dictionary[prefix_tree]

            # Create a dictionary with DescriptorName as key and TermList strings as values
            if descriptor_name.upper() in disease_variant_to_canonical or (len(matching_prefixes) > 0):
                # Create a data row
                data_row = [descriptor_ui, descriptor_name, "| ".join(set(synonyms)), "| ".join(matching_prefixes),
                            prefix_value]
                data_list.append(data_row)

    with open(mesh_output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Mesh ID', 'Common name', 'Synonyms', 'Tree Number', 'Disease Class'])  # Write the header
        writer.writerows(data_list)


if __name__ == '__main__':
    mesh_file_path = "./input/mesh_desc2023.xml"
    mesh_neurology_hierarchy_terms_file = "./input/Neurology_disease-list_MeSH.xlsx"
    mesh_output_path = "./output/diseases_dictionary_mesh.csv"

    parse_elements_from_mesh_xml(mesh_neurology_hierarchy_terms_file, mesh_file_path, mesh_output_path)

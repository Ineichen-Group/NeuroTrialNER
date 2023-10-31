import requests
import pandas as pd
from multiprocessing import Pool


def process_node(node_uri, parent_uri, headers):
    node_response = requests.get(node_uri, headers=headers).json()
    node_title = node_response['title']['@value']
    result = {'Node URI': node_uri, 'Parent URI': parent_uri, 'Title': node_title}

    if 'child' in node_response:
        children = node_response['child']
        child_results = []
        for child_uri in children:
            child_result = process_node(child_uri, node_uri, headers)
            child_results.append(child_result)

        result['Children'] = child_results

    return result


def flatten_results(results):
    flattened_results = []
    for result in results:
        flattened_results.append(result)
        if 'Children' in result:
            flattened_results.extend(flatten_results(result['Children']))
    return flattened_results


def init_connection_api():
    with open('../../credentials.txt', 'r') as file:
        lines = file.readlines()

    # Initialize variables to store the values
    client_id = None
    client_secret = None
    scope = 'icdapi_access'
    grant_type = 'client_credentials'
    token_endpoint = 'https://icdaccessmanagement.who.int/connect/token'

    # Parse the lines and extract the values
    for line in lines:
        if line.startswith('client_id_ICD'):
            client_id = line.split('=', 1)[1].strip()
        elif line.startswith('client_secret_ICD'):
            client_secret = line.split('=', 1)[1].strip()

    # Check if the values were successfully extracted
    if client_id and client_secret:
        print(f'Successfully loaded secrets for ICD API.')
    else:
        print('Failed to extract values from the file.')

    # set data to post
    payload = {'client_id': client_id,
               'client_secret': client_secret,
               'scope': scope,
               'grant_type': grant_type}

    # make request
    r = requests.post(token_endpoint, data=payload, verify=False).json()
    token = r['access_token']

    # HTTP header fields to set
    headers = {'Authorization': 'Bearer ' + token,
               'Accept': 'application/json',
               'Accept-Language': 'en',
               'API-Version': 'v2'}
    return headers


def process_parallel_from_start_node(start_node_uri, headers):
    print("Processing ICD nodes from: ", start_node_uri)
    # make request
    r = requests.get(start_node_uri, headers=headers, verify=False)
    # Extract child nodes and their titles
    child_nodes = r.json()['child']

    # make request for each child node in parallel -> will expand further the hierarchy
    with Pool() as pool:
        table = pool.starmap(process_node, [(node_uri, r.json()['@id'], headers) for node_uri in child_nodes])

    flattened_table = flatten_results(table)
    df = pd.DataFrame(flattened_table)

    return df


if __name__ == "__main__":
    headers = init_connection_api()

    uri_nervous_system = 'https://id.who.int/icd/entity/1296093776'
    uri_mental_behavioural_neurodev_disorders = 'http://id.who.int/icd/entity/334423054'

    df_mental = process_parallel_from_start_node(uri_mental_behavioural_neurodev_disorders, headers)
    df_mental['Disease Class'] = "Mental, behavioural and neurodevelopmental"
    print(df_mental.shape)

    df_nervous_system = process_parallel_from_start_node(uri_nervous_system, headers)
    df_nervous_system['Disease Class'] = "Diseases of the nervous system"
    print(df_nervous_system.shape)

    df_neurological = pd.concat([df_nervous_system, df_mental])
    print("All diseases list ICD shape: ", df_neurological.shape)
    df_neurological.to_csv("./output/icd_11_neurological_diseases.csv")

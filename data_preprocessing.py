import pandas as pd

def load_dataset(file_path):
    return pd.read_csv(file_path)

def handle_missing_values(data):
    return data.fillna(method='ffill')

def encode_categorical_variables(data):
    encoded_data = data.copy()
    mappings = {}
    for column in ['Gender', 'Education', 'Job_Title', 'Location']:
        mappings[column] = dict(enumerate(data[column].astype('category').cat.categories))
        encoded_data[column] = encoded_data[column].astype('category').cat.codes
    return encoded_data, mappings

def define_features_target(data):
    X = data[['Gender', 'Education', 'Job_Title', 'Location', 'Experience']]
    y = data['Salary']
    return X, y

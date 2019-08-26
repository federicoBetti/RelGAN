import json
import pickle


def load_json(json_file_path):
    with open(json_file_path) as json_file:
        saved_dict = json.load(json_file)
    return saved_dict


def load_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as handle:
        saved_object = pickle.load(handle)
    return saved_object


def write_json(json_file_path, dict_to_save):
    with open(json_file_path, 'w') as outfile:
        json.dump(dict_to_save, outfile)


def write_pickle(pickle_file_path, *params):
    if len(list(params)) == 1:
        p = list(params)[0] # to save a single file not as a list
    else:
        p = list(params)

    with open(pickle_file_path, 'wb') as handle:
        pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)

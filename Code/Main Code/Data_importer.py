import csv
from operator import itemgetter
import os
import json
import pickle
import pandas as pd
from datetime import datetime as datetime


def location_lookup():
    locations = json.loads(open("location_settings.json").read())
    
    for path in locations:
        locations[path] = os.path.expandvars(locations[path])
        print(os.path.expandvars(locations[path]))
    return locations

def load_train_set():
	print("Loading training set")

	location = location_lookup()["train_path"]
	df = pd.read_hdf(location)

	return df

def load_test_set():
	print("Loading training set")

	location = location_lookup()["test_path"]
	df = pd.read_hdf(location)

	return df

def dump_model(model,book_bool=True):
	'''
	Purpose:
		dump model in binary representation for usage in other Python scripts
	'''
    if book_bool:
        out_path = location_lookup()["model_path_book"]
    else:
        out_path = location_lookup()["model_path_click"]
    pickle.dump(model, open(out_path, "wb")) # use wb because data is binary

def load_model(book_bool=True):
	'''
	Purpose:
		Read pickle data dump
	'''
    if book_bool:
        in_path = location_lookup()["model_path_book"]
    else:
        in_path = location_lookup()["model_path_click"]
    return pickle.load(open(in_path,'rb')) # use rb because data is binary

def write_submission(recommendations, submission_file=None):
    if submission_file is None:
        submission_path = get_paths()["submission_path"]
    else:
        path, file_name = os.path.split(get_paths()["submission_path"])
        submission_path = os.path.join(path, submission_file)
    rows = [(srch_id, prop_id)
        for srch_id, prop_id, rank_float
        in sorted(recommendations, key=itemgetter(0,2))]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    writer.writerow(("SearchId", "PropertyId"))
    writer.writerows(rows)
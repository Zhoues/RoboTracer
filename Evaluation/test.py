from datasets import load_dataset

# Load the entire dataset (all splits: location, placement)
# This returns a DatasetDict
# use the local path : /share/project/zhouenshen/hpfs/code/RoboTracer/Evaluation/TraceSpatial-Bench
dataset_dict = load_dataset("/share/project/zhouenshen/hpfs/code/RoboTracer/Evaluation/TraceSpatial-Bench")

import ipdb; ipdb.set_trace()

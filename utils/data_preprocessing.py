from typing import Optional, Any

import json
from pathlib import Path

import numpy as np
import h5py
import datasets
from huggingface_hub import snapshot_download
from dataclasses import dataclass
import sympy

import warnings

import argparse
import os
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/inference")
    
    args = parser.parse_args()
    local_dir = args.local_dir
    output_dir = "./data/inference"
    
    benchmark_dataset = []
    sample_h5file_path = Path(local_dir) / "lsr_bench_data.hdf5"
    
    # lsr_synth
    for dataset_identifier in ['matsci','chem_react','bio_pop_growth','phys_osc']:
        ds = datasets.load_dataset(local_dir)[f'lsr_synth_{dataset_identifier}']
        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e in ds:
                samples = {k:v[...].astype(np.float64) for k,v in sample_file[f'/lsr_synth/{dataset_identifier}/{e["name"]}'].items()}
                item = {
                'dataset_identifier': f'lsr_synth/{dataset_identifier}',
                'equation_idx': e['name'],
                'symbols': e['symbols'],
                'symbol_descs': e['symbol_descs'],
                'symbol_properties': e['symbol_properties'],
                'expression': e['expression'],
                'samples': samples # dict: ['train', 'test','ood_test']
                }
                benchmark_dataset.append(item)

    print(len(benchmark_dataset))
    benchmark_dataset = datasets.Dataset.from_list(benchmark_dataset)
    benchmark_dataset.to_parquet(os.path.join(output_dir, "llmsrbench.parquet"))
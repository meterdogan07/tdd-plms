from datasets import load_dataset, load_from_disk
from datasets import Dataset
from itertools import product
import logging
import os
from collections import Counter, defaultdict, deque
from typing import Dict, Set

import gcsfs
import simhash
import typer
import yaml
import datasets
from datasets import load_dataset, DatasetDict
from datasets.load import load_from_disk
from fsspec.spec import AbstractFileSystem
from tqdm import tqdm

def check_num_proc(num_proc: int = -1) -> int:
    """
    Check the number of processors. Return a safe-checked value.

    Parameters
    ----------
    num_proc : int, optional
        Number of processors to use, by default -1

    Returns
    -------
    int
        Number of processors to use

    Raises
    ------
    ValueError
        If the input exceeds the number of processors available
    """
    maximum: int = cpu_count()
    if num_proc > maximum:
        raise ValueError(
            f"{num_proc} exceeds the maximum number ({maximum}) of processors"
        )

    if num_proc == -1:
        num_proc = maximum
    else:
        print(f"Using {num_proc} processors out of {maximum} can be slow")

    return num_proc


if __name__ == "__main__":
    conf = "./deduplicate/conf/self_deduplicate_tr.yaml"

    with open(conf, "r") as f:
        conf = yaml.safe_load(f.read())
        
        print("Opened conf file")

    if conf["load_from_disk"]["path"]:
        fs: AbstractFileSystem = None
        print("fs is None")
        if conf["load_from_disk"]["gcs"]:
            fs = gcsfs.GCSFileSystem(project=conf["load_from_disk"]["gcs"])
        print("Starting to load from disk")
        ds = load_from_disk(conf["load_from_disk"]["path"], fs=fs)
    else:
        print("Starting to load from disk")
        ds = load_dataset(**conf["load_dataset"])

    print("Loaded dataset")

    indices = list(range(10000))

    ds = ds.select(indices)
    print("Selected indices")
    print(len(ds))
    ds.set_format("pandas")
    print("Starting to convert to csv")
    ds.to_csv('./mc4_downloaded/train_mc4.csv')
        
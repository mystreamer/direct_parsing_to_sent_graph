#!/usr/bin/env python3
# coding=utf-8

import os
import argparse
import torch

from model.model import Model
from data.dataset import Dataset
from utility.initialize import initialize
from config.params import Params
from utility.predict import predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    # parser.add_argument("--data_directory", type=str, default="../data")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    args = Params().load_state_dict(checkpoint["params"])
    args = args.init_data_paths()
    args.log_wandb = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    initialize(args, init_wandb=False)

    dataset = Dataset(args)

    model = Model(dataset, args).to(device)
    model.load_state_dict(checkpoint["model"])

    directory = "./inference_prediction"
    os.makedirs(directory, exist_ok=True)
    print("inference of test data", flush=True)
    # quickly fix the missing "raw test data" params in the checkpointed model
    try :
        raw_test_data = args.raw_test_data
    except AttributeError:
        print("No raw test data path could be found, did you init the data paths?")

    # breakpoint()

    res = predict(model, dataset.test, args.test_data, args.raw_test_data, args, None, directory, device, mode="test")
    print("Sentiment Tuple F1 (labeled?)", res)

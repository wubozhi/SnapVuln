import argparse
import yaml
from core_submodel.model_handler import ModelHandler
from core_submodel.model_handler_extend import ModelHandlerExtend
from core_gnnmodel.model_handler import ModelHandler_GNN
from core_gnnmodel.model_handler_extend import ModelHandlerExtend_GNN
from core_submodel.ensemble import ensemble_result
import torch
import numpy as np
from collections import OrderedDict
import os
import json
import random
import pandas as pd
import gc


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    model = ModelHandler(config)
    model.train()


def test(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    if config['out_dir'] is not None:
        config['pretrained'] = config['out_dir']
        config['out_dir'] = None
    model_handle = ModelHandlerExtend(config)
    metrics = model_handle.test()
    return metrics

def train_gnn(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    model = ModelHandler_GNN(config)
    model.train()


def test_gnn(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    if config['out_dir'] is not None:
        config['pretrained'] = config['out_dir']
        config['out_dir'] = None
    model_handle = ModelHandlerExtend_GNN(config)
    metrics = model_handle.test()
    return metrics


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--grid_search', action='store_true', help='flag: grid search')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    #### configure paths ########################
    d2a_data_dir = "/home/bozhi/snapvuln/data/d2a"
    juliet_data_dir = "/home/bozhi/snapvuln/data/sard"
    vocab_dir = "/home/bozhi/snapvuln/vocabs"
    out_dir = "/home/bozhi/snapvuln/output/models"
    result_dir = "/home/bozhi/snapvuln/output/results"
    #### Set mode (0 for D2A train, 1 for D2A test, 3 for Juliet train, 4 for Juliet test) #####
    mode = 4



    ##################################################################################
    ################################## D2A ###########################################
    ##################################################################################
    
    ##### set the number of subgraphs (k) ######
    config['max_slices_num'] = 16

    ## Train for D2A ####
    if mode == 0:
        # 1. Sub model of Buffer Overflow
        config['trainset'] = os.path.join(d2a_data_dir, 'my_bo_train.jsonl.gz')
        config['devset'] = os.path.join(d2a_data_dir, 'my_bo_valid.jsonl.gz')
        config['testset'] = os.path.join(d2a_data_dir, 'my_bo_test.jsonl.gz')
        config['saved_vocab_file'] = os.path.join(vocab_dir, "d2a_bo_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "d2a_bo/")
        config['result'] = os.path.join(result_dir, "d2a_bo")
        config['slice_type'] = 'bo_slices'
        train(config)
        # 2. Sub model of Interger Overflow
        config['trainset'] = os.path.join(d2a_data_dir, 'my_io_train.jsonl.gz')
        config['devset'] = os.path.join(d2a_data_dir, 'my_io_valid.jsonl.gz')
        config['testset'] = os.path.join(d2a_data_dir, 'my_io_test.jsonl.gz')
        config['saved_vocab_file'] = os.path.join(vocab_dir, "d2a_io_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "d2a_io/")
        config['result'] = os.path.join(result_dir, "d2a_io")
        config['slice_type'] = 'io_slices'
        train(config)
        # 3. Sub model of Null Pointer Dereference
        config['trainset'] = os.path.join(d2a_data_dir, 'my_np_train.jsonl.gz')
        config['devset'] = os.path.join(d2a_data_dir, 'my_np_valid.jsonl.gz')
        config['testset'] = os.path.join(d2a_data_dir, 'my_np_test.jsonl.gz')
        config['saved_vocab_file'] = os.path.join(vocab_dir, "d2a_np_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "d2a_np/")
        config['result'] = os.path.join(result_dir, "d2a_np")
        config['slice_type'] = 'np_slices'
        train(config)
        # 4. GNN model
        config['trainset'] = os.path.join(d2a_data_dir, 'multi_train.jsonl.gz')
        config['devset'] = os.path.join(d2a_data_dir, 'multi_valid.jsonl.gz')
        config['testset'] = os.path.join(d2a_data_dir, 'multi_test.jsonl.gz')
        config['saved_vocab_file'] = os.path.join(vocab_dir, "d2a_all_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "d2a_all/")
        config['result'] = os.path.join(result_dir, "d2a_all")
        train_gnn(config)

    ## test for D2A ####
    elif mode == 1:
        # 0. set the path of test data #####
        config['testset'] = os.path.join(d2a_data_dir, 'my_test.jsonl.gz')

        # 1. Sub model of Buffer Overflow
        config['saved_vocab_file'] = os.path.join(vocab_dir, "d2a_bo_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "d2a_bo/")
        config['result'] = os.path.join(result_dir, "d2a_bo")
        config['slice_type'] = 'bo_slices'
        test(config)
        # 2. Sub model of Interger Overflow
        config['saved_vocab_file'] = os.path.join(vocab_dir, "d2a_io_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "d2a_io/")
        config['result'] = os.path.join(result_dir, "d2a_io")
        config['slice_type'] = 'io_slices'
        test(config)
        # 3. Sub model of Null Pointer Dereference
        config['saved_vocab_file'] = os.path.join(vocab_dir, "d2a_np_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "d2a_np/")
        config['result'] = os.path.join(result_dir, "d2a_np")
        config['slice_type'] = 'np_slices'
        test(config)

        # 4. GNN model
        config['testset'] = os.path.join(d2a_data_dir, 'multi_test.jsonl.gz')
        config['saved_vocab_file'] = os.path.join(vocab_dir, "d2a_all_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "d2a_all/")
        config['result'] = os.path.join(result_dir, "d2a_all")
        test_gnn(config)

        # 5. Ensemble result
        result_files = [os.path.join(result_dir, file) for file in os.listdir(result_dir) if "d2a_" in file]
        print(result_files)
        metrics = ensemble_result(result_files)
        print(metrics)



    ##################################################################################
    ################################## Juliet ########################################
    ##################################################################################

    ##### set the number of subgraphs (k) ######
    config['max_slices_num'] = 4

    ## Train for Juliet ####
    if mode == 3:
        # 1. Sub model of Buffer Overflow
        config['trainset'] = os.path.join(juliet_data_dir, 'my_bo_train.jsonl.gz')
        config['devset'] = os.path.join(juliet_data_dir, 'my_bo_valid.jsonl.gz')
        config['testset'] = os.path.join(juliet_data_dir, 'my_bo_test.jsonl.gz')
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_bo_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_bo/")
        config['result'] = os.path.join(result_dir, "juliet_bo")
        config['slice_type'] = 'bo_slices'
        train(config)
        # 2. Sub model of Interger Overflow
        config['trainset'] = os.path.join(juliet_data_dir, 'my_io_train.jsonl.gz')
        config['devset'] = os.path.join(juliet_data_dir, 'my_io_valid.jsonl.gz')
        config['testset'] = os.path.join(juliet_data_dir, 'my_io_test.jsonl.gz')
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_io_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_io/")
        config['result'] = os.path.join(result_dir, "juliet_io")
        config['slice_type'] = 'io_slices'
        train(config)
        # 3. Sub model of Null Pointer Dereference
        config['trainset'] = os.path.join(juliet_data_dir, 'my_np_train.jsonl.gz')
        config['devset'] = os.path.join(juliet_data_dir, 'my_np_valid.jsonl.gz')
        config['testset'] = os.path.join(juliet_data_dir, 'my_np_test.jsonl.gz')
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_np_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_np/")
        config['result'] = os.path.join(result_dir, "juliet_np")
        config['slice_type'] = 'np_slices'
        train(config)
        # 4. Sub model of Memory leak
        config['trainset'] = os.path.join(juliet_data_dir, 'my_ml_train.jsonl.gz')
        config['devset'] = os.path.join(juliet_data_dir, 'my_ml_valid.jsonl.gz')
        config['testset'] = os.path.join(juliet_data_dir, 'my_ml_test.jsonl.gz')
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_ml_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_ml/")
        config['result'] = os.path.join(result_dir, "juliet_ml")
        config['slice_type'] = 'ml_slices'
        train(config)
        # 5. Sub model of Use After Free
        config['trainset'] = os.path.join(juliet_data_dir, 'my_uf_train.jsonl.gz')
        config['devset'] = os.path.join(juliet_data_dir, 'my_uf_valid.jsonl.gz')
        config['testset'] = os.path.join(juliet_data_dir, 'my_uf_test.jsonl.gz')
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_uf_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_uf/")
        config['result'] = os.path.join(result_dir, "juliet_uf")
        config['slice_type'] = 'uaf_slices'
        train(config)
        # 6. Sub model of Double Free
        config['trainset'] = os.path.join(juliet_data_dir, 'my_df_train.jsonl.gz')
        config['devset'] = os.path.join(juliet_data_dir, 'my_df_valid.jsonl.gz')
        config['testset'] = os.path.join(juliet_data_dir, 'my_df_test.jsonl.gz')
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_df_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_df/")
        config['result'] = os.path.join(result_dir, "juliet_df")
        config['slice_type'] = 'df_slices'
        train(config)
        # 7. GNN model
        config['trainset'] = os.path.join(juliet_data_dir, 'multi_train.jsonl.gz')
        config['devset'] = os.path.join(juliet_data_dir, 'multi_valid.jsonl.gz')
        config['testset'] = os.path.join(juliet_data_dir, 'multi_test.jsonl.gz')
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_all_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_all/")
        config['result'] = os.path.join(result_dir, "juliet_all")
        train_gnn(config)

    ## test for Juliet ####
    elif mode == 4:
        # 0. set the path of test data #####
        config['testset'] = os.path.join(juliet_data_dir, 'my_test.jsonl.gz')

        # 1. Sub model of Buffer Overflow
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_bo_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_bo/")
        config['result'] = os.path.join(result_dir, "juliet_bo")
        config['slice_type'] = 'bo_slices'
        test(config)
        # 2. Sub model of Interger Overflow
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_io_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_io/")
        config['result'] = os.path.join(result_dir, "juliet_io")
        config['slice_type'] = 'io_slices'
        test(config)
        # 3. Sub model of Null Pointer Dereference
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_np_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_np/")
        config['result'] = os.path.join(result_dir, "juliet_np")
        config['slice_type'] = 'np_slices'
        test(config)
        # 4. Sub model of Memory leak
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_ml_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_ml/")
        config['result'] = os.path.join(result_dir, "juliet_ml")
        config['slice_type'] = 'ml_slices'
        test(config)
        # 5. Sub model of Use After Free
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_uf_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_uf/")
        config['result'] = os.path.join(result_dir, "juliet_uf")
        config['slice_type'] = 'uaf_slices'
        test(config)
        # 6. Sub model of Double Free
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_df_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_df/")
        config['result'] = os.path.join(result_dir, "juliet_df")
        config['slice_type'] = 'uaf_slices'
        test(config)

        # 7. GNN model
        config['testset'] = os.path.join(juliet_data_dir, 'multi_test.jsonl.gz')
        config['saved_vocab_file'] = os.path.join(vocab_dir, "juliet_all_vocab.pkl")
        config['out_dir'] = os.path.join(out_dir, "juliet_all/")
        config['result'] = os.path.join(result_dir, "juliet_all")
        test_gnn(config)

        # 5. Ensemble result
        result_files = [os.path.join(result_dir, file) for file in os.listdir(result_dir) if "juliet_" in file]
        metrics = ensemble_result(result_files)
        print(metrics)

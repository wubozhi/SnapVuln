# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""
import torch
import json
import numpy as np
from scipy.sparse import *
from . import padding_utils
import gzip
from tqdm import tqdm
import re
# from numpy import random
import multiprocessing


def vectorize_input(batch, training=True, device=None, mode='train'):
    if not batch:
        return None
    slices_mask = torch.LongTensor(batch.slices_mask)
    targets = torch.FloatTensor(batch.targets)
    srcs = torch.LongTensor(batch.srcs)
    src_lens = torch.LongTensor(batch.src_lens)
    with torch.set_grad_enabled(training):
        example = {'batch_size': batch.batch_size,
                   'code_graphs': batch.code_graph,
                   'slices_mask':slices_mask.to(device) if device else slices_mask,
                   'targets': targets.to(device) if device else targets,
                   'srcs': srcs.to(device) if device else srcs,
                   'src_lens': src_lens.to(device) if device else src_lens,
                   'files': batch.filenames
                   }
        return example


def prepare_datasets(config):
    if config['trainset'] is not None:
        train_set, train_src_len = read_all_Datasets(config['trainset'], config, 'train', isLower=True)
        print('# of training examples: {}'.format(len(train_set)))
        print('Training source node length: {}'.format(train_src_len))
    else:
        train_set = None

    if config['devset'] is not None:
        dev_set, dev_src_len = read_all_Datasets(config['devset'], config, 'dev', isLower=True)
        print('# of dev examples: {}'.format(len(dev_set)))
        print('Dev source node length: {}'.format(dev_src_len))
    else:
        dev_set = None
    return {'train': train_set, 'dev': dev_set}


def read_all_Datasets(inpath, config, mode='dev', isLower=True):
    raw_instances = []
    code_graph_len = []
    lines = []
    raw_vul, raw_non_vul = 0, 0
    if type(inpath) is list:
        for subfile in inpath:
            with gzip.GzipFile(subfile, 'r') as f:
                lines.extend(list(f))
    else:
        with gzip.GzipFile(inpath, 'r') as f:
            lines.extend(list(f))
    results = parallel_process(lines, single_instance_process, args=(config, isLower,))
    print("line: %s, graphs: %s"%(len(lines),len(results)))
    for sent1 in results:
        if mode == 'train':
            if sent1.get_node_length() > 1500 or sent1.get_max_token_in_node() > 500:                           ####noted
                continue
        if sent1.target == 1:
            raw_vul += 1
        else:
            raw_non_vul += 1
        code_graph_len.append(sent1.get_node_length())
        raw_instances.append(sent1)

    np.random.shuffle(raw_instances)
    sample_data = raw_instances
    code_graph_len_stats = {'min': np.min(code_graph_len), 'max': np.max(code_graph_len), 'mean': np.mean(code_graph_len),
                            'raw_count': len(raw_instances), 'raw_vul': raw_vul, 'raw_non_vul': raw_non_vul}
    return sample_data, code_graph_len_stats


def parallel_process(array, single_instance_process, args=(), n_cores=None):
    if n_cores == 1:
        return [single_instance_process(x, *args) for x in tqdm(array)]
    with tqdm(total=len(array)) as pbar:
        def update(*args):
            pbar.update()
        if n_cores is None:
            n_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=n_cores) as pool:
            jobs = [
                pool.apply_async(single_instance_process, (x, *args), callback=update) for x in array
            ]
            results = [job.get() for job in jobs]
        return results


def single_instance_process(line, config, isLower):
    instance = json.loads(line)
    sent1 = Graph(instance, config, isLower=isLower)
    # print("Graph:%s"%len(sent1))
    return sent1


class Graph(object):
    def __init__(self, instance, config, isLower=False):
        self.config = config
        slices = instance[self.config['slice_type']]
        self.slices_graph, self.slices_mask, self.max_token_in_node = self.bulid_subgraph(slices, isLower)
        self.function = instance['file_txt']
        self.func_tokens = instance['file_tokens']
        self.target = instance['target']
        self.file = instance['file']

    def bulid_subgraph(self, slices, isLower):
        subgraphs = []
        slices_mask = []
        pad_subgraph = {'nodes': [{'id': 0, 'content': 'null', 'type': 'STATE'}, {'id': 1, 'content': 'null', 'type': 'STATE'}], 'edges': [['IS_AST_PARENT', 0, 1]]}
        max_token_in_node = 0

        shuffled_index = [i for i in range(len(slices))]
        np.random.shuffle(shuffled_index)
        for i in shuffled_index[:self.config['max_slices_num']]:
            s = slices[i]
            filter_code_nodes, edges, max_token = self.build_code_graph(s, isLower)
            subgraphs.append({'nodes': filter_code_nodes, 'edges': edges})
            slices_mask.append(1)
            if max_token > max_token_in_node:
                max_token_in_node = max_token

        pad_length = self.config['max_slices_num'] - len(slices_mask)
        # subgraphs += [subgraphs[0] for i in range(pad_length)]
        subgraphs += [pad_subgraph for i in range(pad_length)]
        slices_mask += [1]
        slices_mask += [0] * pad_length

        return subgraphs, slices_mask, max_token_in_node


    def build_code_graph(self, code_graph, isLower):
        filter_code_nodes = []
        filter_edges = []
        max_token_in_node = 0
        if self.config['IsCFGDFG']:
            nodes, edges = self.get_cfgdfg(code_graph)
        else:
            nodes, edges = code_graph['nodes'], code_graph['edges']
        for node in nodes:
            if isLower:
                if 'name' in node.keys():
                    node_content = node['name'].lower()
                else:
                    node_content = node['code'].lower()
            else:
                if 'name' in node.keys():
                    node_content = node['name']
                else:
                    node_content = node['code']

            if len(node_content.split(' ')) > max_token_in_node:
                max_token_in_node = len(node_content.split(' '))
            filter_code_nodes.append({'id': node['ID'], 'content': node_content, 'type': node['type']})
        for edge in edges:
            if edge[0] in ['IS_AST_PARENT', 'FLOWS_TO', 'DEF', 'REACHES', 'USE', 'CONTROLS']:
                filter_edges.append([edge[0], edge[1], edge[2]])
        return filter_code_nodes, filter_edges, max_token_in_node

    def get_node_length(self):
        nodes = []
        for g in self.slices_graph:
            nodes.extend(g['nodes'])
        return len(nodes)

    def get_token_length(self):
        return len(self.func_tokens)

    def get_max_token_in_node(self):
        return self.max_token_in_node

    def get_cfgdfg(self, code_graph):
        edges = code_graph['edges']
        nodes = code_graph['nodes']
        cfg_nodes, new_edges, new_nodes = [], [], []
        for edge in edges:
            type, src, dest = edge[0], edge[1], edge[2]
            if type == 'FLOWS_TO' or type == 'USE' or type == 'DEF' or type == 'CALL_GRAPH':
                cfg_nodes.append(src)
                cfg_nodes.append(dest)
                new_edges.append(edge)
        for node in nodes:
            if node['ID'] in cfg_nodes:
                new_nodes.append(node)
        map = dict()
        for index, node in enumerate(new_nodes):
            map[node['ID']] = index
            node['ID'] = index
        for edge in new_edges:
            edge[1] = map[edge[1]]
            edge[2] = map[edge[2]]
        return new_nodes, new_edges


class DataStream(object):
    def __init__(self, all_instances, word_vocab, edge_vocab, config=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1):
        self.config = config
        if batch_size == -1: batch_size = config['batch_size']
        if isSort:
            all_instances = sorted(all_instances, key=lambda instance: (instance.get_node_length()))
        else:
            np.random.shuffle(all_instances)
            np.random.shuffle(all_instances)
        self.num_instances = len(all_instances)

        # distribute srcs into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for (batch_start, batch_end) in tqdm(batch_spans):
            cur_instances = all_instances[batch_start: batch_end]
            cur_batch = Batch(cur_instances, config, word_vocab, edge_vocab)
            if len(cur_batch.targets) == 1:
                continue
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[i]


class Batch(object):
    def __init__(self, instances, config, word_vocab, edge_vocab):
        self.instances = instances
        self.batch_size = len(instances)
        # Create word representation and length
        self.filenames = []
        self.targets = []
        self.srcs = []
        self.src_lens = []
        self.slices_mask = []
        batch_code_graph = []
        self.word_vocab = word_vocab
        for sent1 in instances:
            for subgraph in sent1.slices_graph:
                batch_code_graph.append(subgraph)
            self.targets.append(sent1.target)
            self.srcs.append(self.w2idx(sent1.func_tokens))
            self.src_lens.append(len(sent1.func_tokens))
            self.filenames.append(sent1.file)
            self.slices_mask.append(sent1.slices_mask)

        # print(len(batch_code_graph))
        self.targets = np.array(self.targets, dtype=np.int32)
        # print("slices_mask")
        # print(self.slices_mask)
        self.slices_mask = np.array(self.slices_mask, dtype=np.int32)
        self.srcs = padding_utils.pad_2d_vals_no_size(self.srcs)
        self.src_lens = np.array(self.src_lens, dtype=np.int32)
        batch_code_graphs = cons_batch_graph(batch_code_graph, word_vocab)
        self.code_graph = vectorize_batch_graph(batch_code_graphs, edge_vocab, config)

    def w2idx(self, words_list):
        ids = []
        for word in words_list:
            ids.append(self.word_vocab.getIndex(word))
        return ids


def cons_batch_graph(graphs, word_vocab):
    num_nodes = max([len(g['nodes']) for g in graphs])
    num_edges = max([len(g['edges']) for g in graphs])
    batch_edges = []
    batch_node2edge, batch_edge2node = [], []
    batch_node_labels, batch_node_num = [], []
    batch_node_word_index = []
    batch_max_node_tokens = []
    for g in graphs:
        graph_node_word_index, max_node_tokens = cons_node_features(g['nodes'], word_vocab)
        edges, node2edge, edge2node = cons_edge_features(g['edges'], num_edges, num_nodes)
        batch_edges.append(edges)
        batch_node2edge.append(node2edge)
        batch_edge2node.append(edge2node)
        batch_node_num.append(len(g['nodes']))
        batch_node_word_index.append(graph_node_word_index)
        batch_max_node_tokens.append(max_node_tokens)
    batch_graphs = {'max_num_edges': num_edges,
                    'edge_features': batch_edges,
                    'node2edge': batch_node2edge,
                    'edge2node': batch_edge2node,
                    'node_num': batch_node_num,
                    'max_num_nodes': num_nodes,
                    'node_word_index': batch_node_word_index,
                    'max_node_tokens': batch_max_node_tokens
                    }
    return batch_graphs


def cons_edge_features(edges, num_edges, num_nodes):
    node2edge = lil_matrix(np.zeros((num_edges, num_nodes)), dtype=np.float32)
    edge2node = lil_matrix(np.zeros((num_nodes, num_edges)), dtype=np.float32)
    edge_index = 0
    edge2index = {}
    for edge, src_node, dest_node in edges:
        # if src_node == dest_node:                       # Ignore self-loops for now
        #     continue
        edge2index[edge_index] = edge
        node2edge[edge_index, dest_node] = 1
        edge2node[src_node, edge_index] = 1
        edge_index += 1
    return edge2index, node2edge, edge2node


def cons_node_features(nodes, word_vocab):
    graph_node_word_index = []
    max_node_tokens = 0
    for node_id, node in enumerate(nodes):
        node_word_index = []
        code = node['content']
        splitted_code = re.split('\\s+', code)
        if max_node_tokens < len(splitted_code):
            max_node_tokens = len(splitted_code)
        for word in splitted_code:
            idx = word_vocab.getIndex(word)
            node_word_index.append(idx)
        graph_node_word_index.append(node_word_index)
    return graph_node_word_index, max_node_tokens


def vectorize_batch_graph(graph, edge_vocab, config):
    edge_features = []
    for edges in graph['edge_features']:
        edges_v = []
        for idx in range(len(edges)):
            edges_v.append(edge_vocab.getIndex(edges[idx]))
        for _ in range(graph['max_num_edges'] - len(edges_v)):
            edges_v.append(edge_vocab.PAD)
        edge_features.append(edges_v)
    edge_features = torch.LongTensor(np.array(edge_features))
    node_num_masks = torch.FloatTensor(np.zeros((len(graph['node_num']), graph['max_num_nodes'])))
    for i in range(len(graph['node_num'])):
        node_num_masks[i, :graph['node_num'][i]] = 1
    node_word_index = np.zeros((sum(graph['node_num']), np.max(graph['max_node_tokens'])), dtype=np.int32)
    node_word_lengths = []
    index = 0
    for graph_index, each_graph in enumerate(graph['node_word_index']):
        for node_index, node in enumerate(each_graph):
            node_word_index[index, 0: len(node)] = node
            node_word_lengths.append(len(node))
            index += 1
    node_word_lengths = np.array(node_word_lengths, dtype=np.int32)
    node_word_index = torch.LongTensor(node_word_index)
    node_word_lengths = torch.LongTensor(node_word_lengths)
    gv = {'edge_features': edge_features.to(config['device']) if config['device'] else edge_features,
          'node2edge': graph['node2edge'],
          'edge2node': graph['edge2node'],
          'node_num_masks': node_num_masks.to(config['device']) if config['device'] else node_num_masks,
          'node_word_index': node_word_index.to(config['device']) if config['device'] else node_word_index,
          'node_word_lengths': node_word_lengths.to(config['device']) if config['device'] else node_word_lengths,
          'node_num': graph['node_num'],
          'max_node_num_batch': graph['max_num_nodes'],
          }
    return gv
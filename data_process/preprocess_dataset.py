import os
import gzip
import codecs
import json
from sklearn.model_selection import train_test_split
import random
from scripts.process_data import preprocess
from tqdm import tqdm
random.seed(2022)

def load_jsonl(file_path):
	instances = []
	with open(file_path, "r") as f:
		lines = f.readlines()
		for line in tqdm(lines):
			instance = json.loads(line.strip())
			instances.append(instance)
	return instances

def load_jsonl_gz(file_path):
	instances = []
	with gzip.GzipFile(file_path, 'r') as f:
		lines = list(f)
		for line in tqdm(lines):
			instance = json.loads(line.strip())
			instances.append(instance['file'])
	return instances

def save_jsonl_gz(filename, data):
	with gzip.GzipFile(filename, 'w') as out_file:
		writer = codecs.getwriter('utf-8')
		for element in tqdm(data):
			writer(out_file).write(json.dumps(element))
			writer(out_file).write('\n')

def split_d2a(output_folder):
	train_dataset, valid_dataset, test_dataset = [], [], []
	samples_vuln = load_jsonl(os.path.join(output_folder,"vuln_my.jsonl"))
	samples_nonvuln = load_jsonl(os.path.join(output_folder,"nonvuln_my.jsonl"))

	print("vul:%d"%len(samples_vuln))
	print("nonvul:%d"%len(samples_nonvuln))

	integer_overflow = [i for i in samples_vuln if i['vul_type'] == 'integer_overflow']
	buffer_overflow = [i for i in samples_vuln if i['vul_type'] == 'buffer_overflow']
	null_pointers = [i for i in samples_vuln if i['vul_type'] == 'null_pointers']

	train_io_dataset, test_io_dataset = train_test_split(integer_overflow, test_size=0.2, random_state=2020)
	test_io_dataset, valid_io_dataset = train_test_split(test_io_dataset, test_size=0.5, random_state=2020)
	train_bo_dataset, test_bo_dataset = train_test_split(buffer_overflow, test_size=0.2, random_state=2020)
	test_bo_dataset, valid_bo_dataset = train_test_split(test_bo_dataset, test_size=0.5, random_state=2020)
	train_np_dataset, test_np_dataset = train_test_split(null_pointers, test_size=0.2, random_state=2020)
	test_np_dataset, valid_np_dataset = train_test_split(test_np_dataset, test_size=0.5, random_state=2020)

	train_vul_dataset = train_io_dataset + train_bo_dataset + train_np_dataset
	valid_vul_dataset = valid_io_dataset + valid_bo_dataset + valid_np_dataset
	test_vul_dataset = test_io_dataset + test_bo_dataset + test_np_dataset

	train_nonvul_dataset, test_nonvul_dataset = train_test_split(samples_nonvuln, test_size=0.2, random_state=2020)
	test_nonvul_dataset, valid_nonvul_dataset = train_test_split(test_nonvul_dataset, test_size=0.5, random_state=2020)

	test_ion_dataset = random.sample(test_nonvul_dataset, len(test_io_dataset))
	test_bon_dataset = random.sample(test_nonvul_dataset, len(test_bo_dataset))
	test_npn_dataset = random.sample(test_nonvul_dataset, len(test_np_dataset))

	train_dataset.extend(train_nonvul_dataset)
	train_dataset.extend(train_vul_dataset)
	valid_dataset.extend(valid_nonvul_dataset)
	valid_dataset.extend(valid_vul_dataset)
	test_dataset.extend(test_nonvul_dataset)
	test_dataset.extend(test_vul_dataset)
	random.shuffle(train_dataset)
	random.shuffle(valid_dataset)
	random.shuffle(test_dataset)

	print("train: %d, valid: %d, test: %d"%(len(train_dataset),len(valid_dataset),len(test_dataset)))
	save_jsonl_gz(os.path.join(output_folder,"my_train.jsonl.gz"), train_dataset)
	print('%d samples in trainset' % len(train_dataset))
	save_jsonl_gz(os.path.join(output_folder,'my_valid.jsonl.gz'), valid_dataset)
	print('%d samples in validset' % len(valid_dataset))
	save_jsonl_gz(os.path.join(output_folder,'my_test.jsonl.gz'), test_dataset)
	print('%d samples in testset' % len(test_dataset))

	bo_train_dataset = [{'bo_slices':i['bo_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in train_dataset if i['bo_slices'] != []]
	io_train_dataset = [{'io_slices':i['io_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in train_dataset if i['io_slices'] != []]
	np_train_dataset = [{'np_slices':i['np_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in train_dataset if i['np_slices'] != []]
	save_jsonl_gz(os.path.join(output_folder,"my_bo_train.jsonl.gz"), bo_train_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_io_train.jsonl.gz"), io_train_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_np_train.jsonl.gz"), np_train_dataset)

	bo_test_dataset = [{'bo_slices':i['bo_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in test_dataset if i['bo_slices'] != []]
	io_test_dataset = [{'io_slices':i['io_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in test_dataset if i['io_slices'] != []]
	np_test_dataset = [{'np_slices':i['np_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in test_dataset if i['np_slices'] != []]
	save_jsonl_gz(os.path.join(output_folder,"my_bo_test.jsonl.gz"), bo_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_io_test.jsonl.gz"), io_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_np_test.jsonl.gz"), np_test_dataset)

	bo_valid_dataset = [{'bo_slices':i['bo_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in valid_dataset if i['bo_slices'] != []]
	io_valid_dataset = [{'io_slices':i['io_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in valid_dataset if i['io_slices'] != []]
	np_valid_dataset = [{'np_slices':i['np_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in valid_dataset if i['np_slices'] != []]
	save_jsonl_gz(os.path.join(output_folder,"my_bo_valid.jsonl.gz"), bo_valid_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_io_valid.jsonl.gz"), io_valid_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_np_valid.jsonl.gz"), np_valid_dataset)

	io_test_dataset, bo_test_dataset, np_test_dataset = [], [], []
	io_test_dataset.extend(test_io_dataset)
	io_test_dataset.extend(test_ion_dataset)
	random.shuffle(io_test_dataset)

	bo_test_dataset.extend(test_bo_dataset)
	bo_test_dataset.extend(test_bon_dataset)
	random.shuffle(bo_test_dataset)

	np_test_dataset.extend(test_np_dataset)
	np_test_dataset.extend(test_npn_dataset)
	random.shuffle(np_test_dataset)

	print("io_test_dataset: %d, bo_test_dataset: %d, np_test_dataset: %d"%(len(io_test_dataset),len(bo_test_dataset),len(np_test_dataset)))
	save_jsonl_gz(os.path.join(output_folder,"my_io_test_dataset.jsonl.gz"), io_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,'my_bo_test_dataset.jsonl.gz'), bo_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,'my_np_test_dataset.jsonl.gz'), np_test_dataset)

	##### get files #######
	train_path = "/home/bozhi2/work_vuln/data/json/d2a/test/my_train.jsonl.gz"
	valid_path = "/home/bozhi2/work_vuln/data/json/d2a/test/my_valid.jsonl.gz"
	test_path = "/home/bozhi2/work_vuln/data/json/d2a/test/my_test.jsonl.gz"
	train_files = load_jsonl_gz(train_path)
	valid_files = load_jsonl_gz(valid_path)
	test_files = load_jsonl_gz(test_path)

	train_path = "/home/bozhi2/work_vuln/data/json/d2a/test/my_io_test_dataset.jsonl.gz"
	valid_path = "/home/bozhi2/work_vuln/data/json/d2a/test/my_bo_test_dataset.jsonl.gz"
	test_path = "/home/bozhi2/work_vuln/data/json/d2a/test/my_np_test_dataset.jsonl.gz"
	io_test_files = load_jsonl_gz(train_path)
	bo_test_files = load_jsonl_gz(valid_path)
	np_test_files = load_jsonl_gz(test_path)

	### get multi sample ##########
	multi_vuln = load_jsonl(os.path.join(output_folder,"vuln_multi.jsonl"))
	multi_nonvuln = load_jsonl(os.path.join(output_folder,"nonvuln_multi.jsonl"))
	multi = multi_vuln + multi_nonvuln
	multi_train = [i for i in multi if i['file'] in train_files]
	multi_valid = [i for i in multi if i['file'] in valid_files]
	multi_test = [i for i in multi if i['file'] in test_files]
	random.shuffle(multi_train)
	random.shuffle(multi_valid)
	random.shuffle(multi_test)
	save_jsonl_gz(os.path.join(output_folder,"multi_train.jsonl.gz"), multi_train)
	save_jsonl_gz(os.path.join(output_folder,'multi_valid.jsonl.gz'), multi_valid)
	save_jsonl_gz(os.path.join(output_folder,'multi_test.jsonl.gz'), multi_test)	

	multi_io = [i for i in multi if i['file'] in io_test_files]
	multi_bo = [i for i in multi if i['file'] in bo_test_files]
	multi_np = [i for i in multi if i['file'] in np_test_files]
	random.shuffle(multi_io)
	random.shuffle(multi_bo)
	random.shuffle(multi_np)
	save_jsonl_gz(os.path.join(output_folder,"multi_io_test_dataset.jsonl.gz"), multi_io)
	save_jsonl_gz(os.path.join(output_folder,'multi_bo_test_dataset.jsonl.gz'), multi_bo)
	save_jsonl_gz(os.path.join(output_folder,'multi_np_test_dataset.jsonl.gz'), multi_np)	



def split_sard(output_folder):
	train_dataset, valid_dataset, test_dataset = [], [], []
	samples_vuln = load_jsonl(os.path.join(output_folder,"vuln_my.jsonl"))
	samples_nonvuln = load_jsonl(os.path.join(output_folder,"nonvuln_my.jsonl"))

	print("vul:%d"%len(samples_vuln))
	print("nonvul:%d"%len(samples_nonvuln))

	integer_overflow = [i for i in samples_vuln if i['vul_type'] == 'integer_overflow']
	buffer_overflow = [i for i in samples_vuln if i['vul_type'] == 'buffer_overflow']
	null_pointers = [i for i in samples_vuln if i['vul_type'] == 'null_pointers']
	memory_leak = [i for i in samples_vuln if i['vul_type'] == 'memory_leak']
	double_free = [i for i in samples_vuln if i['vul_type'] == 'double_free']
	use_after_free = [i for i in samples_vuln if i['vul_type'] == 'use_after_free']

	train_io_dataset, test_io_dataset = train_test_split(integer_overflow, test_size=0.2, random_state=2020)
	test_io_dataset, valid_io_dataset = train_test_split(test_io_dataset, test_size=0.5, random_state=2020)

	train_bo_dataset, test_bo_dataset = train_test_split(buffer_overflow, test_size=0.2, random_state=2020)
	test_bo_dataset, valid_bo_dataset = train_test_split(test_bo_dataset, test_size=0.5, random_state=2020)

	train_np_dataset, test_np_dataset = train_test_split(null_pointers, test_size=0.2, random_state=2020)
	test_np_dataset, valid_np_dataset = train_test_split(test_np_dataset, test_size=0.5, random_state=2020)

	train_ml_dataset, test_ml_dataset = train_test_split(memory_leak, test_size=0.2, random_state=2020)
	test_ml_dataset, valid_ml_dataset = train_test_split(test_ml_dataset, test_size=0.5, random_state=2020)

	train_df_dataset, test_df_dataset = train_test_split(double_free, test_size=0.2, random_state=2020)
	test_df_dataset, valid_df_dataset = train_test_split(test_df_dataset, test_size=0.5, random_state=2020)

	train_ua_dataset, test_ua_dataset = train_test_split(use_after_free, test_size=0.2, random_state=2020)
	test_ua_dataset, valid_ua_dataset = train_test_split(test_ua_dataset, test_size=0.5, random_state=2020)

	train_vul_dataset = train_io_dataset + train_bo_dataset + train_np_dataset + train_ml_dataset + train_df_dataset + train_ua_dataset
	valid_vul_dataset = valid_io_dataset + valid_bo_dataset + valid_np_dataset + valid_ml_dataset + valid_df_dataset + valid_ua_dataset
	test_vul_dataset = test_io_dataset + test_bo_dataset + test_np_dataset + test_ml_dataset + test_df_dataset + test_ua_dataset

	train_nonvul_dataset, test_nonvul_dataset = train_test_split(samples_nonvuln, test_size=0.2, random_state=2020)
	test_nonvul_dataset, valid_nonvul_dataset = train_test_split(test_nonvul_dataset, test_size=0.5, random_state=2020)

	test_ion_dataset = random.sample(test_nonvul_dataset, len(test_io_dataset))
	test_bon_dataset = random.sample(test_nonvul_dataset, len(test_bo_dataset))
	test_npn_dataset = random.sample(test_nonvul_dataset, len(test_np_dataset))
	test_mln_dataset = random.sample(test_nonvul_dataset, len(test_ml_dataset))
	test_dfn_dataset = random.sample(test_nonvul_dataset, len(test_df_dataset))
	test_uan_dataset = random.sample(test_nonvul_dataset, len(test_ua_dataset))

	train_dataset.extend(train_nonvul_dataset)
	train_dataset.extend(train_vul_dataset)
	valid_dataset.extend(valid_nonvul_dataset)
	valid_dataset.extend(valid_vul_dataset)
	test_dataset.extend(test_nonvul_dataset)
	test_dataset.extend(test_vul_dataset)
	random.shuffle(train_dataset)
	random.shuffle(valid_dataset)
	random.shuffle(test_dataset)

	print("train: %d, valid: %d, test: %d"%(len(train_dataset),len(valid_dataset),len(test_dataset)))
	save_jsonl_gz(os.path.join(output_folder,"my_train.jsonl.gz"), train_dataset)
	print('%d samples in trainset' % len(train_dataset))
	save_jsonl_gz(os.path.join(output_folder,'my_valid.jsonl.gz'), valid_dataset)
	print('%d samples in validset' % len(valid_dataset))
	save_jsonl_gz(os.path.join(output_folder,'my_test.jsonl.gz'), test_dataset)
	print('%d samples in testset' % len(test_dataset))


	bo_train_dataset = [{'bo_slices':i['bo_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in train_dataset if i['bo_slices'] != []]
	io_train_dataset = [{'io_slices':i['io_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in train_dataset if i['io_slices'] != []]
	np_train_dataset = [{'np_slices':i['np_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in train_dataset if i['np_slices'] != []]
	ml_train_dataset = [{'ml_slices':i['ml_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in train_dataset if i['ml_slices'] != []]
	df_train_dataset = [{'df_slices':i['df_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in train_dataset if i['uaf_slices'] != []]
	uf_train_dataset = [{'uaf_slices':i['uaf_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in train_dataset if i['uaf_slices'] != []]
	save_jsonl_gz(os.path.join(output_folder,"my_bo_train.jsonl.gz"), bo_train_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_io_train.jsonl.gz"), io_train_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_np_train.jsonl.gz"), np_train_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_ml_train.jsonl.gz"), ml_train_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_df_train.jsonl.gz"), df_train_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_uf_train.jsonl.gz"), uf_train_dataset)

	bo_test_dataset = [{'bo_slices':i['bo_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in test_dataset if i['bo_slices'] != []]
	io_test_dataset = [{'io_slices':i['io_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in test_dataset if i['io_slices'] != []]
	np_test_dataset = [{'np_slices':i['np_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in test_dataset if i['np_slices'] != []]
	ml_test_dataset = [{'ml_slices':i['ml_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in test_dataset if i['ml_slices'] != []]
	df_test_dataset = [{'df_slices':i['df_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in test_dataset if i['uaf_slices'] != []]
	uf_test_dataset = [{'uaf_slices':i['uaf_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in test_dataset if i['uaf_slices'] != []]
	save_jsonl_gz(os.path.join(output_folder,"my_bo_test.jsonl.gz"), bo_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_io_test.jsonl.gz"), io_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_np_test.jsonl.gz"), np_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_ml_test.jsonl.gz"), ml_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_df_test.jsonl.gz"), df_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_uf_test.jsonl.gz"), uf_test_dataset)

	bo_valid_dataset = [{'bo_slices':i['bo_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in valid_dataset if i['bo_slices'] != []]
	io_valid_dataset = [{'io_slices':i['io_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in valid_dataset if i['io_slices'] != []]
	np_valid_dataset = [{'np_slices':i['np_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in valid_dataset if i['np_slices'] != []]
	ml_valid_dataset = [{'ml_slices':i['ml_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in valid_dataset if i['ml_slices'] != []]
	df_valid_dataset = [{'df_slices':i['df_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in valid_dataset if i['uaf_slices'] != []]
	uf_valid_dataset = [{'uaf_slices':i['uaf_slices'], 'file_txt':i['file_txt'], 'file_tokens':i['file_tokens'], 'file':i['file'],'vul_type':i['vul_type'],'target': i['target']} for i in valid_dataset if i['uaf_slices'] != []]
	save_jsonl_gz(os.path.join(output_folder,"my_bo_valid.jsonl.gz"), bo_valid_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_io_valid.jsonl.gz"), io_valid_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_np_valid.jsonl.gz"), np_valid_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_ml_valid.jsonl.gz"), ml_valid_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_df_valid.jsonl.gz"), df_valid_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_uf_valid.jsonl.gz"), uf_valid_dataset)


	io_test_dataset, bo_test_dataset, np_test_dataset, ml_test_dataset, df_test_dataset, ua_test_dataset = [], [], [], [], [], []
	io_test_dataset.extend(test_io_dataset)
	io_test_dataset.extend(test_ion_dataset)
	random.shuffle(io_test_dataset)

	bo_test_dataset.extend(test_bo_dataset)
	bo_test_dataset.extend(test_bon_dataset)
	random.shuffle(bo_test_dataset)

	np_test_dataset.extend(test_np_dataset)
	np_test_dataset.extend(test_npn_dataset)
	random.shuffle(np_test_dataset)

	ml_test_dataset.extend(test_ml_dataset)
	ml_test_dataset.extend(test_mln_dataset)
	random.shuffle(np_test_dataset)

	df_test_dataset.extend(test_df_dataset)
	df_test_dataset.extend(test_dfn_dataset)
	random.shuffle(df_test_dataset)

	ua_test_dataset.extend(test_ua_dataset)
	ua_test_dataset.extend(test_uan_dataset)
	random.shuffle(ua_test_dataset)	

	print("io_test_dataset: %d, bo_test_dataset: %d, np_test_dataset: %d"%(len(io_test_dataset),len(bo_test_dataset),len(np_test_dataset)))
	print("ml_test_dataset: %d, df_test_dataset: %d, ua_test_dataset: %d"%(len(ml_test_dataset),len(df_test_dataset),len(ua_test_dataset)))

	save_jsonl_gz(os.path.join(output_folder,"my_io_test_dataset.jsonl.gz"), io_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,'my_bo_test_dataset.jsonl.gz'), bo_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,'my_np_test_dataset.jsonl.gz'), np_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,"my_ml_test_dataset.jsonl.gz"), ml_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,'my_df_test_dataset.jsonl.gz'), df_test_dataset)
	save_jsonl_gz(os.path.join(output_folder,'my_ua_test_dataset.jsonl.gz'), ua_test_dataset)

	##### get files #######

	train_files = [i['file'] for i in train_dataset]
	valid_files = [i['file'] for i in valid_dataset]
	test_files = [i['file'] for i in test_dataset]

	io_test_files = [i['file'] for i in io_test_dataset]
	bo_test_files = [i['file'] for i in bo_test_dataset]
	np_test_files = [i['file'] for i in np_test_dataset]
	ml_test_files = [i['file'] for i in ml_test_dataset]
	df_test_files = [i['file'] for i in df_test_dataset]
	ua_test_files = [i['file'] for i in ua_test_dataset]

	### get multi sample ##########
	multi_vuln = load_jsonl(os.path.join(output_folder,"vuln_multi.jsonl"))
	multi_nonvuln = load_jsonl(os.path.join(output_folder,"nonvuln_multi.jsonl"))
	multi = multi_vuln + multi_nonvuln
	multi_train = [i for i in multi if i['file'] in train_files]
	multi_valid = [i for i in multi if i['file'] in valid_files]
	multi_test = [i for i in multi if i['file'] in test_files]
	random.shuffle(multi_train)
	random.shuffle(multi_valid)
	random.shuffle(multi_test)
	save_jsonl_gz(os.path.join(output_folder,"multi_train.jsonl.gz"), multi_train)
	save_jsonl_gz(os.path.join(output_folder,'multi_valid.jsonl.gz'), multi_valid)
	save_jsonl_gz(os.path.join(output_folder,'multi_test.jsonl.gz'), multi_test)	

	multi_io = [i for i in multi if i['file'] in io_test_files]
	multi_bo = [i for i in multi if i['file'] in bo_test_files]
	multi_np = [i for i in multi if i['file'] in np_test_files]
	multi_ml = [i for i in multi if i['file'] in ml_test_files]
	multi_df = [i for i in multi if i['file'] in df_test_files]
	multi_ua = [i for i in multi if i['file'] in ua_test_files]
	random.shuffle(multi_io)
	random.shuffle(multi_bo)
	random.shuffle(multi_np)
	random.shuffle(multi_ml)
	random.shuffle(multi_df)
	random.shuffle(multi_ua)
	save_jsonl_gz(os.path.join(output_folder,"multi_io_test_dataset.jsonl.gz"), multi_io)
	save_jsonl_gz(os.path.join(output_folder,'multi_bo_test_dataset.jsonl.gz'), multi_bo)
	save_jsonl_gz(os.path.join(output_folder,'multi_np_test_dataset.jsonl.gz'), multi_np)	
	save_jsonl_gz(os.path.join(output_folder,"multi_ml_test_dataset.jsonl.gz"), multi_ml)
	save_jsonl_gz(os.path.join(output_folder,'multi_df_test_dataset.jsonl.gz'), multi_df)
	save_jsonl_gz(os.path.join(output_folder,'multi_ua_test_dataset.jsonl.gz'), multi_ua)	


if __name__ == "__main__":
	### set output paths ######
	sard_output_folder = "/home/bozhi2/work_vuln/data/json/sard"
	d2a_output_folder = "/home/bozhi2/work_vuln/data/json/d2a"
	### set operation flag: 0 for juliet data process, 1 for juliet data split, 3 for d2a data process, 4 for d2a data split #######
	flag = 0

	####################################   sard  ########################################################################
	if flag == 0:
		multi_func_folder = "/home/bozhi2/work_vuln/data/sard/func/multi/vuln"
		multi_dot_folder = "/home/bozhi2/work_vuln/data/sard/dot/multi/vuln/combine"
		single_func_folder = "/home/bozhi2/work_vuln/data/sard/func/single/vuln"
		single_dot_folder = "/home/bozhi2/work_vuln/data/sard/dot/single/vuln/dots"
		preprocess(multi_func_folder,multi_dot_folder,single_func_folder,single_dot_folder,sard_output_folder,'vuln')

		multi_func_folder = "/home/bozhi2/work_vuln/data/sard/func/multi/nonvuln"
		multi_dot_folder = "/home/bozhi2/work_vuln/data/sard/dot/multi/nonvuln/combine"
		single_func_folder = "/home/bozhi2/work_vuln/data/sard/func/single/nonvuln"
		single_dot_folder = "/home/bozhi2/work_vuln/data/sard/dot/single/nonvuln/dots"
		preprocess(multi_func_folder,multi_dot_folder,single_func_folder,single_dot_folder,sard_output_folder,'nonvuln')

	elif flag == 1:
		split_sard(sard_output_folder)


	####################################   d2a   ########################################################################
	elif flag == 3:
		multi_func_folder = "/home/bozhi2/work_vuln/data/d2a/func/vuln/multi"
		multi_dot_folder = "/home/bozhi2/work_vuln/data/d2a/dot/vuln/combine"
		single_func_folder = "/home/bozhi2/work_vuln/data/d2a/func/vuln/single"
		single_dot_folder = "/home/bozhi2/work_vuln/data/d2a/dot/vuln/single/dots"
		preprocess(multi_func_folder,multi_dot_folder,single_func_folder,single_dot_folder,d2a_output_folder,'vuln')

		multi_func_folder = "/home/bozhi2/work_vuln/data/d2a/func/nonvuln/multi"
		multi_dot_folder = "/home/bozhi2/work_vuln/data/d2a/dot/nonvuln/combine"
		single_func_folder = "/home/bozhi2/work_vuln/data/d2a/func/nonvuln/single"
		single_dot_folder = "/home/bozhi2/work_vuln/data/d2a/dot/nonvuln/single/dots"
		preprocess(multi_func_folder,multi_dot_folder,single_func_folder,single_dot_folder,d2a_output_folder,'nonvuln')

	elif flag == 4:
		split_d2a(d2a_output_folder)
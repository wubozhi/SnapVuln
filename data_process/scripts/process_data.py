import os
import json
from tqdm import tqdm
import glob
import multiprocessing
from scripts.slice_rules import * 
from scripts.slice_api import *


def read_raw_function(file_path):
	with open(file_path, 'r', encoding='utf-8') as f:
		txt = f.read()
	return txt


def convert_nodes_edges(dot, mode='default'):
	new_nodes, new_edges = [], []
	if mode == 'default':
		nodes, edges = get_nodes_and_edges(dot, mode)
	elif mode == 'string':
		nodes = dot['nodes']
		edges = dot['edges']
	else:
		return None, None

	node_ids = sorted([int(node['ID']) for node in nodes])
	for node in nodes:
		node['ID'] = node_ids.index(int(node['ID']))
		new_nodes.append(node)

	for edge in edges:
		new_edges.append([edge['type'], node_ids.index(int(edge['front'])), node_ids.index(int(edge['rear']))])

	return new_nodes, new_edges


def parallel_process(array, single_instance_process, args=(), n_cores=None):
	with tqdm(total=len(array)) as pbar:
		def update(*args):
			pbar.update()
		if n_cores is None:
			n_cores = multiprocessing.cpu_count()
		with multiprocessing.Pool(processes=n_cores) as pool:
			jobs = [
					pool.apply_async(single_instance_process, (x, *args), callback=update) for x in array
			]
			results = [job.get() for job in jobs if job.get() != None]
		return results


def get_slices(dotfile):
	bo_slices, ml_slices, io_slices, np_slices, uaf_slices, df_slices = [], [], [], [], []

	nodes, edges = get_nodes_and_edges(dotfile)
	slice_set = bo_slice(nodes, edges)
	for _slice in slice_set:
		ns, es = convert_nodes_edges(_slice, 'string')
		# flag = 1
		# for s in bo_slices:
		# 	tmp = [i for i in ns if i not in s['nodes']]
		# 	if len(tmp)==0:
		# 		print("The slice belongs to another slice.")
		# 		flag = 0
		# if flag:
		content = {'nodes': ns, 'edges': es}
		bo_slices.append(content)

	slice_set = ml_slice(nodes, edges)
	for _slice in slice_set:
		ns, es = convert_nodes_edges(_slice, 'string')
		# flag = 1
		# for s in ml_slices:
		# 	tmp = [i for i in ns if i not in s['nodes']]
		# 	if len(tmp)==0:
		# 		print("The slice belongs to another slice.")
		# 		flag = 0
		# if flag:
		content = {'nodes': ns, 'edges': es}
		ml_slices.append(content)

	slice_set = io_slice(nodes, edges)
	io_slices = []
	for _slice in slice_set:
		ns, es = convert_nodes_edges(_slice, 'string')
		# flag = 1
		# for s in io_slices:
		# 	tmp = [i for i in ns if i not in s['nodes']]
		# 	if len(tmp)==0:
		# 		print("The slice belongs to another slice.")
		# 		flag = 0
		# if flag:
		content = {'nodes': ns, 'edges': es}
		io_slices.append(content)

	slice_set = np_slice(nodes, edges)
	np_slices = []
	for _slice in slice_set:
		ns, es = convert_nodes_edges(_slice, 'string')
		# flag = 1
		# for s in np_slices:
		# 	tmp = [i for i in ns if i not in s['nodes']]
		# 	if len(tmp)==0:
		# 		print("The slice belongs to another slice.")
		# 		flag = 0
		# if flag:
		content = {'nodes': ns, 'edges': es}
		np_slices.append(content)

	slice_set = uaf_slice(nodes, edges)
	uaf_slices = []
	for _slice in slice_set:
		ns, es = convert_nodes_edges(_slice, 'string')
		# flag = 1
		# for s in uaf_slices:
		# 	tmp = [i for i in ns if i not in s['nodes']]
		# 	if len(tmp)==0:
		# 		print("The slice belongs to another slice.")
		# 		flag = 0
		# if flag:
		content = {'nodes': ns, 'edges': es}
		uaf_slices.append(content)

	slice_set = df_slice(nodes, edges)
	df_slices = []
	for _slice in slice_set:
		ns, es = convert_nodes_edges(_slice, 'string')
		# flag = 1
		# for s in df_slices:
		# 	tmp = [i for i in ns if i not in s['nodes']]
		# 	if len(tmp)==0:
		# 		print("The slice belongs to another slice.")
		# 		flag = 0
		# if flag:
		content = {'nodes': ns, 'edges': es}
		df_slices.append(content)

	if len(bo_slices) + len(ml_slices) + len(io_slices) + len(np_slices) + len(uaf_slices) + len(df_slices) == 0:
		return None, None, None, None, None

	return bo_slices, ml_slices, io_slices, np_slices, uaf_slices, df_slices


def single_instance_process(file,multi_func_folder,multi_dot_folder,single_func_folder,single_dot_folder,output_folder,label):
	bug_func = file.replace("-multi_function.c", "-bug_function.c")
	file_dot = file.replace(".c",".dot")
	bug_dot = bug_func.replace(".c","*.c.dot")
	if bug_func not in os.listdir(single_func_folder):
		print("No bug function.")
		return None
	else:
		bug_func = os.path.join(single_func_folder,bug_func)

	if file_dot not in os.listdir(multi_dot_folder):
		print("No multi-function dot.")
		return None
	else:
		file_dot = os.path.join(multi_dot_folder, file_dot)

	single_dots = glob.glob(os.path.join(single_dot_folder,bug_dot))
	if len(single_dots) == 0:
		print("No bug dot.")
		return None
	else:
		bug_dot = single_dots[0]

	if label == "vuln":
		target = 1 
		if "BUFFER_OVERRUN" in file or "Buffer_Overflow" in file:
			vul_type = "buffer_overflow"
		elif "INTEGER_OVERFLOW" in file or "Integer_Overflow" in file:
			vul_type = "integer_overflow"
		elif "NULLPTR_DEREFERENCE" in file or "NULL_Pointer_Dereference" in file:
			vul_type = "null_pointers"
		elif "Memory_Leak" in file:
			vul_type = "memory_leak"
		elif "Double_Free" in file:
			vul_type = "double_free"
		elif "Use_After_Free" in file:
			vul_type = "use_after_free"
		else:
			vul_type = "unknown"
	else:
		target = 0
		vul_type = "nonvuln"

	bug_txt = read_raw_function(bug_func)
	# if len(bug_txt.split('\n')) >= 1500:
	# 	print("file too big.")
	# 	return None
	bug_tokens = tokenizer(bug_txt)
	bug_dot_nodes, bug_dot_edges = convert_nodes_edges(bug_dot)

	file = os.path.join(multi_func_folder,file)
	file_txt = read_raw_function(file)
	# if len(file_txt.split('\n')) >= 1500:
	# 	print("file too big.")
	# 	return None
	file_tokens = tokenizer(file_txt)
	file_dot_nodes, file_dot_edges = convert_nodes_edges(file_dot)

	bo_slices, ml_slices, io_slices, np_slices, uaf_slices, df_slices = get_slices(file_dot)
	if bo_slices == None:
		print("No slices")
		return None


	my_sample = {'bo_slices':bo_slices, 'ml_slices':ml_slices, 'io_slices':io_slices, 'np_slices':np_slices, 'uaf_slices':uaf_slices, 'df_slices':df_slices,
				 'file_txt':file_txt, 'file_tokens':file_tokens, 'file':file,'vul_type':vul_type,'target': int(target)}

	multi_sample = {'multi_graph': {'nodes': file_dot_nodes, 'edges': file_dot_edges},
					'file_txt':file_txt, 'file_tokens':file_tokens,'file':file,'vul_type':vul_type,'target': int(target)}

	single_sample = {'single_graph':{'nodes': bug_dot_nodes, 'edges': bug_dot_edges}, 
					 'bug_txt':bug_txt, 'bug_tokens':bug_tokens, 'bug_func':bug_func,'file':file,'vul_type':vul_type,'target': int(target)}	

	myfile = os.path.join(output_folder,label+"_my.jsonl")
	multi_file = os.path.join(output_folder,label+"_multi.jsonl")
	single_file = os.path.join(output_folder,label+"_single.jsonl")

	with open(myfile, "a") as f:
		f.write(json.dumps(my_sample)+'\n')
	with open(multi_file, "a") as f:
		f.write(json.dumps(multi_sample)+'\n')
	with open(single_file, "a") as f:
		f.write(json.dumps(single_sample)+'\n')


def preprocess(multi_func_folder,multi_dot_folder,single_func_folder,single_dot_folder,output_folder,label):
	files = os.listdir(multi_func_folder)
	parallel_process(files, single_instance_process, (multi_func_folder,multi_dot_folder,single_func_folder,single_dot_folder,output_folder,label))


if __name__ == '__main__':
	pass



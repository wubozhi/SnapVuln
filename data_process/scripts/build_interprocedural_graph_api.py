# - coding = utf-8-#
import re
import subprocess
import os
import shutil
import time
import sys
if sys.version > '3':
	import queue as Queue
else:
	import Queue

def get_nodes_and_edges(dotfile, mode='default'):
	node_chunks, edge_chunks = [], []
	regex1 = r'(  \d+ \[.*?\]\;\n)'
	regex2 = r'(  \d+ \-\> \d+ .*?\]\;\n)'
	if mode == 'default':
		with open(dotfile,'r') as f:
			fs = f.read()
	elif mode == 'string':
		fs = dotfile
	else:
		return None, None
	node_chunks = re.findall(regex1, fs, re.S)
	edge_chunks = re.findall(regex2, fs, re.S)

	nodes, edges = [], []
	for chunk in node_chunks:
		node = {}
		id_regex = r"nodeid:(\d+)"
		fid_regex = r'functionId="(.*?)" '
		name_regex = r'name="(.*?)" '
		type_regex = r'type="(.*?)" '
		code_regex = r'code="(.*?)" '
		loc_regex = r'location:(\d+)'
		childnum_regex = r'childNum:(\d+)'
		_id = re.findall(id_regex, chunk)
		fid = re.findall(fid_regex, chunk)
		name = re.findall(name_regex, chunk)
		tp = re.findall(type_regex, chunk)
		code = re.findall(code_regex, chunk)
		loc = re.findall(loc_regex, chunk)
		childnum = re.findall(childnum_regex, chunk)
		if _id != []:
			node['nodeid'] = _id[0]
		if fid != []:
			node['functionId']=fid[0]
		if name != []:
			node['name']=name[0]
		if tp != []:
			node['type'] = tp[0]
		if code != []:
			node['code'] = code[0]
		if loc != []:
			node['location'] = loc[0]
		if childnum != []:
			node['childNum'] = childnum[0]
		nodes.append(node)

	for chunk in edge_chunks:
		edge_regex = r'name\=\"\(\((\d+)\) \: \((\d+)\) \: (.+?)\)\" \]\;\n'
		edge_info = re.findall(edge_regex, chunk, re.S)[0]
		edges.append({'front':edge_info[0],'rear':edge_info[1],'type':edge_info[2]})

	return nodes, edges


def get_graph(nodes, edges, g_type):
	if g_type == 'ast':
		edge_type_list = ['IS_AST_PARENT','IS_FUNCTION_OF_AST','CALL_GRAPH']
	elif g_type == 'pdg':
		edge_type_list = ['CONTROLS', 'REACHES','IS_FUNCTION_OF_CFG','CALL_GRAPH']
	elif g_type == 'ddg':
		edge_type_list = ['REACHES','IS_FUNCTION_OF_CFG','CALL_GRAPH']
	elif g_type == 'cdg':
		edge_type_list = ['CONTROLS','IS_FUNCTION_OF_CFG','CALL_GRAPH']
	elif g_type == 'cfg':
		edge_type_list = ['FLOWS_TO','IS_FUNCTION_OF_CFG','CALL_GRAPH']
	elif g_type == 'dfg':
		edge_type_list = ['USE','DEF']
	else:
		raise Exception("No such graph type: %s"%g_type)

	tgt_nodes = []
	tgt_edges = []

	tmp_nodes = []
	for edge in edges:
		if edge['type'] in edge_type_list:
			tgt_edges.append(edge)
			front = edge['front']
			rear = edge['rear']
			if front not in tmp_nodes:
				tmp_nodes.append(front)
			if rear not in tmp_nodes:
				tmp_nodes.append(rear)
	for node in nodes:
		if node['nodeid'] in tmp_nodes:
			tgt_nodes.append(node)

	return tgt_nodes, tgt_edges


def get_callee(nodes):
	callees = [node for node in nodes if node['type'] == "Callee"]
	return callees


def get_identifiers(nodes, edges):
	identifiers = []
	dfg_nodes, dfg_edges = get_graph(nodes, edges, 'dfg')
	def_nodes_id = [edge['front'] for edge in dfg_edges if edge['type'] == 'DEF']
	identifiers = [node for node in nodes if node['nodeid'] in def_nodes_id]
	return identifiers


def get_pointers(nodes, edges):
	pointers = []

	dfg_nodes, dfg_edges = get_graph(nodes, edges, 'dfg')
	def_node_ids = [edge['front'] for edge in dfg_edges if edge['type'] == 'DEF']

	tmp = [node for node in nodes if node['type'] == "IdentifierDeclStatement" or node['type'] == "Parameter"]
	for node in tmp:
		if node['nodeid'] not in def_node_ids:
			continue

		code = node['code']
		if code.find(' = ') != -1:
			code = code.split(' = ')[0]

		if code.find('*') != -1:
			pointers.append(node)

	return pointers



def get_arrays(nodes, edges):
	arrays = []

	dfg_nodes, dfg_edges = get_graph(nodes, edges, 'dfg')
	def_node_ids = [edge['front'] for edge in dfg_edges if edge['type'] == 'DEF']

	tmp = [node for node in nodes if node['type'] == "IdentifierDeclStatement" or node['type'] == "Parameter"]
	for node in tmp:
		if node['nodeid'] not in def_node_ids:
			continue

		code = node['code']
		if code.find(' = ') != -1:
			code = code.split(' = ')[0]

		if code.find(' [ ') != -1:
			arrays.append(node)

	return arrays


def get_operators(nodes, edges):
	operators = []

	expstmt_nodes = [node for node in nodes if node['type'] == "ExpressionStatement" or node['type'] == 'IdentifierDeclStatement']
	for exp in expstmt_nodes:
		code = exp['code']
		if code.find(' = ') > -1:
			code_op = code.split(' = ')[-1]
			if not "+" in code_op and not "*" in code_op:
				continue
			else:
				operators.append(exp)
		else:
			if not "+" in code and not "*" in code:
				continue
			else:
				operators.append(exp)
	return operators


def get_correspond_node(node, nodes, edges, g_type):
	ast_nodes, ast_edges = get_graph(nodes, edges, 'ast')
	root_nodeid_ast = [edge['front'] for edge in ast_edges if edge['type']=="IS_FUNCTION_OF_AST"][0]
	if g_type == 'cfg':
		_nodes, _ = get_graph(nodes, edges, 'cfg')
	elif g_type == 'dfg':
		_nodes, _ = get_graph(nodes, edges, 'dfg')
	elif g_type == 'pdg':
		_nodes, _ = get_graph(nodes, edges, 'pdg')
	elif g_type == 'ddg':
		_nodes, _ = get_graph(nodes, edges, 'ddg')
	elif g_type == 'cdg':
		_nodes, _ = get_graph(nodes, edges, 'cdg')
	else:
		print("no such graph: %s"%g_type)
		return None


	node_id = node['nodeid']
	correspond_nodes_id = [n['nodeid'] for n in _nodes if 'code' in n.keys()]

	tmp_id = node_id
	flag = True
	while flag:
		tmp_ids = [edge['front'] for edge in ast_edges if int(edge['rear']) == int(tmp_id)]
		if len(tmp_ids) != 1:
			print("Node %s not in AST (%d) ."%(node, len(tmp_ids)))
			return None

		tmp_id = tmp_ids[0]
		if tmp_id in correspond_nodes_id:
			flag = False
			break
		elif tmp_id == root_nodeid_ast:
			print("No correspond node in %s"%g_type)
			return None

	_node = [node for node in _nodes if int(node['nodeid']) == int(tmp_id)][0]
	return _node


def get_forward_nodes(node, nodes, edges):
	forward_nodes_id = []
	node_id = node['nodeid']

	for edge in edges:
		if edge['front'] == node_id:
			forward_nodes_id.append(edge['rear'])

	forward_nodes = [node for node in nodes if node['nodeid'] in forward_nodes_id]
	return forward_nodes

def get_backward_nodes(node, nodes, edges):
	backward_nodes_id = []
	node_id = node['nodeid']

	for edge in edges:
		if edge['rear'] == node_id:
			backward_nodes_id.append(edge['front'])
			
	backward_nodes = [node for node in nodes if node['nodeid'] in backward_nodes_id]
	return backward_nodes


def forward_slice(criterion, nodes, edges):
	result = []
	visited = set()
	q = Queue.Queue()
	q.put(criterion)
	while not q.empty():
		u = q.get()
		result.append(u)
		g = get_forward_nodes(u, nodes, edges)
		for v in g:
			if v['nodeid'] not in visited:
				visited.add(v['nodeid'])
				q.put(v)
	return result

def backward_slice(criterion, nodes, edges):
	result = []
	visited = set()
	q = Queue.Queue()
	q.put(criterion)
	while not q.empty():
		u = q.get()
		result.append(u)
		g = get_backward_nodes(u, nodes, edges)
		for v in g:
			if v['nodeid'] not in visited:
				visited.add(v['nodeid'])
				q.put(v)
	return result

def bid_slice(criterion, nodes, edges):
	result = []
	rb = forward_slice(criterion, nodes, edges)
	rf = backward_slice(criterion, nodes, edges)
	for r in rb:
		if r not in result:
			result.append(r)
	for r in rf:
		if r not in result:
			result.append(r)
	return result

def construct_subgraph(slice_nodes, edges):
	slice_edges = []
	slice_nodes_id = [slice_node['nodeid'] for slice_node in slice_nodes]
	slice_edges = [edge for edge in edges if edge['front'] in slice_nodes_id and edge['rear'] in slice_nodes_id]
	return slice_nodes, slice_edges

def slice_graph_to_dot(slice_nodes, slice_edges, dot_file):
	with open(dot_file, 'w') as f:
		content = 'digraph G {\n'

		node_content = ''
		for node in slice_nodes:
			node_content = node_content + '  '+node['nodeid']+' [ label="'
			for key, value in node.items():
				node_content = node_content +str(key)+':'+str(value)+'\n'
			node_content = node_content + '" '

			for key, value in node.items():
				node_content = node_content + str(key) + '="' + str(value) + '" '
			node_content = node_content + "];\n"

		edge_content = ''
		for edge in slice_edges:
			edge_content = edge_content + '  ' + str(edge['front']) + ' -> ' + str(edge['rear']) + ' [ label="' + str(edge['type']) + '" name="(('+ str(edge['front']) + ') : ('+str(edge['rear'])+') : '+ str(edge['type'])+')" ];\n'

		content = content + node_content + edge_content + '}'
		f.write(content)

def slice_graph_to_content(slice_nodes, slice_edges):
	content = 'digraph G {\n'

	node_content = ''
	for node in slice_nodes:
		node_content = node_content + '  '+node['nodeid']+' [ label="'
		for key, value in node.items():
			node_content = node_content +str(key)+':'+str(value)+'\n'
		node_content = node_content + '" '

		for key, value in node.items():
			node_content = node_content + str(key) + '="' + str(value) + '" '
		node_content = node_content + "];\n"

	edge_content = ''
	for edge in slice_edges:
		edge_content = edge_content + '  ' + str(edge['front']) + ' -> ' + str(edge['rear']) + ' [ label="' + str(edge['type']) + '" name="(('+ str(edge['front']) + ') : ('+str(edge['rear'])+') : '+ str(edge['type'])+')" ];\n'

	content = content + node_content + edge_content + '}'

	return content



# - coding = utf-8-#
import re
import subprocess
import os
import shutil
import time
import copy
from pygments.lexers.c_cpp import CLexer
import sys
if sys.version > '3':
	import queue as Queue
else:
	import Queue


NODE_TYPE_LIST = {'Symbol': 'SYS', 'PostfixExpression': 'POSTFIXEXPR', 'ElseStatement': 'ELSESTATE', 'ForInit': 'FORINIT',
						'PrimaryExpression': 'PRIEXPR', 'UnaryOp': 'UNAOP', 'CallExpression': 'CALLEXPR', 'IncDecOp': 'INCDECOP',
						'BitAndExpression': 'BITANDEXPR', 'SizeofOperand': 'SIZEOFOPERA', 'Label': 'LABEL', 'ParameterList': 'PARAMLIST',
						'ExpressionHolder': 'EXPRHOLDER', 'ReturnStatement': 'RETURNSTATE', 'Statement': 'STATE', 'AndExpression': 'ANDEXPR',
						'CompoundStatement': 'COMPSTATE', 'Parameter': 'PARAM', 'CastExpression': 'CASTEXPR', 'EqualityExpression': 'EQEXPR',
						'ArgumentList': 'ARGULIST', 'ContinueStatement': 'CONTINUESTATE', 'MultiplicativeExpression': 'MULTIEXPR',
						'Callee': 'CALLEE', 'SizeofExpr': 'SIZEOFEXPR', 'SwitchStatement': 'SWITCHSTATE', 'ForStatement': 'FORSTATE',
						'AssignmentExpr': 'ASSIGNEXPR', 'Sizeof': 'SIZEOF', 'BreakStatement': 'BREAKSTATE', 'GotoStatement': 'GOGOSTATE', 'BlockStarter':'BLOCKSTARTER',
						'UnaryOperator':'UNARYOPER', 'DummyReturnType': 'DUMRETTYPE',  'CastTarget': 'CASTTAR',   'ShiftExpression':'SHIFTEXPR', 'Condition':'COND',
						'ParameterType':'PARAMTYPE', 'InitializerList':'INITIALIST', 'DoStatement':'DOSTATE', 'BlockCloser':'BLKCLOSER', 'IdentifierDecl': 'IDENDECL',
						'OrExpression': 'OREXPR',  'MemberAccess':'MEMACCESS', 'JumpStatement':'JUMPSTATE', 'ReturnType':'RETURNTYPE', 'BinaryExpression':'BINARYEXPR',
						'Identifier':'IDEN', 'Expression':'EXPR', 'IncDec':'IncDec', 'FunctionDef':'FUNCDEF', 'AdditiveExpression':'ADDIEXPR','ExpressionStatement':'EXPRSTATE',
						'ConditionalExpression':'CONDEXPR', 'UnaryExpression':'UNARYEXPR', 'InclusiveOrExpression': 'INCLUOREXPR', 'WhileStatement':'WHILESTATE',
						'ExclusiveOrExpression': 'EXCLUOREXPR', 'Argument':'ARGUMENT', 'IdentifierDeclType':'IDENDECLTYPE', 'RelationalExpression': 'RELATIONEXPRE',
						'PtrMemberAccess': 'PTRMEMACCESS', 'ArrayIndexing':'ARRINDEX', 'IdentifierDeclStatement': 'IDENDECLSTATE', 'ClassDefStatement': 'CLASSDEFSTATE',
						'ExpressionHolderStatement': 'EXPRHOLDERSTATE', 'IfStatement': 'IFSTATE', 'CFGEntryNode': 'CFGENTRYNODE', 'CFGExitNode': 'CFGEXITNODE',
						'CFGErrorNode': 'CFGERRORNODE', 'InfiniteForNode': 'INFINITEFORNODE', 'Function':'Function'}


def tokenizer(fun):
	tokens = []
	for i, j, t in CLexer().get_tokens_unprocessed(fun):
		t = camel_case_split(t)
		for subword in t:
			subword = subword.lower()
			if subword is None or str(subword).strip() == "":
				pass
			elif "Token.Comment" in str(j):
				pass
			elif "Token.Name" in str(j):
				subword = str(subword).split('_')
				tokens.extend(subword)
			else:
				tokens.append(subword)
	return tokens


def camel_case_split(identifier):
	matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
	return [m.group(0) for m in matches]


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

		if tp != []:
			node['type'] = NODE_TYPE_LIST[tp[0]]
			if node['type']	== 'Function':
				if childnum != []:
					node['childNum'] = childnum[0]
					node['ID'] = _id[0]
					node['name']=' '.join(tokenizer(name[0])).lower()
					if loc != []:
						node['location'] = loc[0]
				else:
					node['ID'] = _id[0]
					node['name']=' '.join(tokenizer(name[0])).lower()
					if loc != []:
						node['location'] = loc[0]
			else:
				if code != []:
					if childnum != []:
						node['childNum'] = childnum[0]
						node['ID'] = _id[0]
						node['code']=' '.join(tokenizer(code[0])).lower()
						if loc != []:
							node['location'] = loc[0]
					else:
						node['ID'] = _id[0]
						node['code']=' '.join(tokenizer(code[0])).lower()
						if loc != []:
							node['location'] = loc[0]
				else:
					if childnum != []:
						node['childNum'] = childnum[0]
						node['ID'] = _id[0]
						node['code']=''
						if loc != []:
							node['location'] = loc[0]
					else:
						node['ID'] = _id[0]
						node['code']=''
						if loc != []:
							node['location'] = loc[0]
			nodes.append(node)
			continue

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
		if node['ID'] in tmp_nodes:
			tgt_nodes.append(node)

	return tgt_nodes, tgt_edges


def get_callee(nodes):
	callees = [node['code'] for node in nodes if node['type'] == "CALLEE"]
	return callees


def get_identifiers(nodes, edges):
	identifiers = [node['code'] for node in nodes if node['type'] == 'IDEN']
	return identifiers

def get_operators(nodes, edges):
	operators = []

	expstmt_nodes = [node for node in nodes if node['type'] == "EXPRSTATE" or node['type'] == 'IDENDECLSTATE']
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


def get_pointers(nodes, edges):
	pointers = []

	identifiers = [node['code'] for node in nodes if node['type'] == 'IDEN']
	tmp = [node for node in nodes if node['type'] == "IDENDECLSTATE" or node['type'] == "PARAM"]
	for node in tmp:
		code = node['code']
		if code.find(' = ') != -1:
			code = code.split(' = ')[0]

		if code.find('*') != -1:
			pointer = [i for i in identifiers if i in code]
			pointers.extend(pointer)
	pointers = list(set(pointers))
	return pointers


def get_arrays(nodes, edges):
	arrays = []

	identifiers = [node['code'] for node in nodes if node['type'] == 'IDEN']
	tmp = [node for node in nodes if node['type'] == "IDENDECLSTATE" or node['type'] == "PARAM"]
	for node in tmp:
		code = node['code']
		if code.find(' = ') != -1:
			code = code.split(' = ')[0]

		if code.find(' [ ') != -1:
			array = [i for i in identifiers if i in code.split("[")[0]]
			arrays.extend(array)
	arrays = list(set(arrays))
	return arrays


def get_operator_nodes(nodes, edges):
	operators = []

	expstmt_nodes = [node for node in nodes if node['type'] == "EXPRSTATE" or node['type'] == 'IDENDECLSTATE']
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


	node_id = node['ID']
	correspond_nodes_id = [n['ID'] for n in _nodes if 'code' in n.keys()]

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

	_node = [node for node in _nodes if int(node['ID']) == int(tmp_id)][0]
	return _node


def get_forward_nodes(node, nodes, edges):
	forward_nodes_id = []
	node_id = node['ID']

	for edge in edges:
		if edge['front'] == node_id:
			forward_nodes_id.append(edge['rear'])

	forward_nodes = [node for node in nodes if node['ID'] in forward_nodes_id]
	return forward_nodes

def get_backward_nodes(node, nodes, edges):
	backward_nodes_id = []
	node_id = node['ID']

	for edge in edges:
		if edge['rear'] == node_id:
			backward_nodes_id.append(edge['front'])
			
	backward_nodes = [node for node in nodes if node['ID'] in backward_nodes_id]
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
			if v['ID'] not in visited:
				visited.add(v['ID'])
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
			if v['ID'] not in visited:
				visited.add(v['ID'])
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

def construct_subgraph(_nodes, _edges):
	nodes = copy.deepcopy(_nodes)
	edges = copy.deepcopy(_edges)
	slice_nodes = []
	slice_edges = []
	slice_nodes_id = [slice_node['ID'] for slice_node in nodes]
	for edge in edges:
		if edge['front'] in slice_nodes_id and edge['rear'] in slice_nodes_id:
			slice_edges.append(edge)
			if nodes[slice_nodes_id.index(edge['front'])] not in slice_nodes:
				slice_nodes.append(nodes[slice_nodes_id.index(edge['front'])])
			if nodes[slice_nodes_id.index(edge['rear'])] not in slice_nodes:
				slice_nodes.append(nodes[slice_nodes_id.index(edge['rear'])])
	return slice_nodes, slice_edges


def slice_graph_to_content(slice_nodes, slice_edges):
	content = 'digraph G {\n'

	node_content = ''
	for node in slice_nodes:
		node_content = node_content + '  '+node['ID']+' [ label="'
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


if __name__ == '__main__':
	pass


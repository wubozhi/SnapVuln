import os
from tqdm import tqdm
import glob
import os
import glob
from tqdm import tqdm
import multiprocessing as mp
from scripts.build_interprocedural_graph_api import * 


def construct_graph(func,dot_folder,out_folder):
	call_dots = glob.glob(os.path.join(dot_folder,func.replace(".c","")+"*.c.dot"))
	if  len(call_dots) == 0:
		print("No call dots.")
		return
	try:
		names, ids, callees_ast, callees_pdg, callees_cfg = get_func_id(call_dots)
		call_graph_edges = get_call_edges(names, ids, callees_ast, callees_pdg, callees_cfg)
	except Exception as e:
		print("%s No call graphs. "%func)
		return

	if not os.path.exists(out_folder):
		os.mkdir(out_folder)

	with open(os.path.join(out_folder,func.replace(".c",".dot")),'w') as fd:
		for i, dotfile in enumerate(call_dots):
			with open(dotfile, 'r') as f:
				content = f.read()
			if i != 0:
				content = content.strip().lstrip('digraph G {')
			content = content.rstrip().rstrip('}')
			fd.write(content)
		fd.write(call_graph_edges+'}')


def get_func_id(dotfiles):
	names = []
	ids = []
	callees_ast = []
	callees_pdg = []
	callees_cfg = []
	for dotfile in dotfiles:
		nodes, edges = get_nodes_and_edges(dotfile)
		for node in nodes:
			if node['type'] == "Function":
				names.append(node['name'])
				ids.append(node['nodeid'])
			if node['type'] == "Callee":
				correspond_node_pdg = get_correspond_node(node, nodes, edges, 'pdg')
				correspond_node_cfg = get_correspond_node(node, nodes, edges, 'cfg')
				callees_ast.append(node)
				callees_pdg.append(correspond_node_pdg)
				callees_cfg.append(correspond_node_cfg)				
	return names, ids, callees_ast, callees_pdg, callees_cfg


def get_call_edges(names, ids, callees_ast, callees_pdg, callees_cfg):
	call_graph_edges = ''

	for i, call in enumerate(callees_ast):
		call_func = call['code']
		call_id = call['nodeid']
		if call_func in names:
			### Add AST edge ####
			start = call_id
			end = ids[names.index(call_func)]
			edge = '  %s -> %s [ label="IS_AST_PARENT" name="((%s) : (%s) : IS_AST_PARENT)" ];'%(start, end, start, end)
			call_graph_edges = call_graph_edges + edge + '\n'

			### Add PDG edge ####
			if callees_pdg[i] != None:
				start = callees_pdg[i]['nodeid']
				edge = '  %s -> %s [ label="CONTROLS" name="((%s) : (%s) : CONTROLS)" ];'%(start, end, start, end)
				call_graph_edges = call_graph_edges + edge + '\n'

			### Add CFG edge ####
			if callees_cfg[i] != None:
				start = callees_cfg[i]['nodeid']
				edge = '  %s -> %s [ label="FLOWS_TO" name="((%s) : (%s) : FLOWS_TO)" ];'%(start, end, start, end)
				call_graph_edges = call_graph_edges + edge + '\n'			
				
	return call_graph_edges


if __name__ == "__main__":
	args = []

	############# d2a ##################################################
	# need to employ Joern to generate .dot file (Code Property Graph) 
	# from source code in func_folder, and store .dot file in dot_folder
	nonvuln_func_folder = "./data/d2a/func/nonvuln/multi"
	nonvuln_dot_folder = "./data/d2a/dot/nonvuln/multi/dots"
	vuln_func_folder = "./data/d2a/func/vuln/multi"
	vuln_dot_folder = "./data/d2a/dot/vuln/multi/dots"

	# Configure the output folder that is used to store the inter-procedural graph
	nonvuln_out_folder = "./data/d2a/dot/nonvuln/combine"
	vuln_out_folder = "./data/d2a/dot/vuln/combine"

	## nonvuln 
	function_files = os.listdir(nonvuln_func_folder)
	for func in function_files:
		args.append((func,nonvuln_dot_folder, nonvuln_out_folder))

	with mp.Pool(80) as p:
		p.starmap(construct_graph,args)

	## vuln 
	function_files = os.listdir(vuln_func_folder)
	for func in function_files:
		args.append((func,vuln_dot_folder,vuln_out_folder))

	with mp.Pool(80) as p:
		p.starmap(construct_graph,args)

	############## sard ##################################################
	# need to employ Joern to generate .dot file (Code Property Graph) 
	# from source code in func_folder, and store .dot file in dot_folder
	nonvuln_func_folder = "./data/sard/func/multi/nonvuln"
	nonvuln_dot_folder = "./data/sard/dot/multi/nonvuln/dots"
	vuln_func_folder = "./data/sard/func/multi/vuln"
	vuln_dot_folder = "./data/sard/dot/multi/vuln/dots"

	# Configure the output folder that is used to store the inter-procedural graph
	nonvuln_out_folder = "./data/sard/dot/multi/nonvuln/combine"
	vuln_out_folder = "./data/sard/dot/multi/vuln/combine"

	## nonvuln
	function_files = os.listdir(nonvuln_func_folder)
	for func in tqdm(function_files):
		args.append((func,nonvuln_dot_folder,nonvuln_out_folder))

	with mp.Pool(80) as p:
		p.starmap(construct_graph,args)

	## vuln
	function_files = os.listdir(vuln_func_folder)
	for func in tqdm(function_files):
		args.append((func,vuln_dot_folder,vuln_out_folder))

	with mp.Pool(80) as p:
		p.starmap(construct_graph,args)

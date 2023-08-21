from scripts.slice_api import * 
import os

def bo_slice(nodes, edges):
	slices = []

	pointer_arrays = get_pointers(nodes, edges) + get_arrays(nodes, edges)
	callees = get_callee(nodes)
	if len(pointer_arrays) == 0 or len(callees) ==0:
		print("No pointer or arrays or callees.")
		return slices

	temp = []
	for node in nodes:
		if node['type'] == "EXPRSTATE":
			for item in pointer_arrays:
				if item in node['code']:
					temp.append(node)
					break
	criterions = []
	for node in temp:
		for item in callees:
			if item in node['code']:
				criterions.append(node)
				break				
	pdg_nodes, pdg_edges = get_graph(nodes, edges, 'pdg')
	for criterion in criterions:
		slice_nodes = backward_slice(criterion, pdg_nodes, pdg_edges)
		slice_nodes, slice_edges = construct_subgraph(slice_nodes, pdg_edges)
		if len(slice_edges) == 0 or len(slice_nodes) == 1:
			continue

		slice_content = {'nodes':slice_nodes,'edges':slice_edges}
		slices.append(slice_content)
	return slices


def ml_slice(nodes, edges):
	slices = []

	pointer = get_pointers(nodes, edges)
	callees = get_callee(nodes)
	if len(pointer) == 0 or len(callees) ==0:
		print("No pointer or callees.")
		return slices

	temp = []
	for node in nodes:
		if node['type'] == "EXPRSTATE" and "=" in node['code']:
			for item in pointer:
				if item in node['code']:
					temp.append(node)
					break
	criterions = []
	for node in temp:
		for item in callees:
			if item in node['code']:
				criterions.append(node)
				break	
	pdg_nodes, pdg_edges = get_graph(nodes, edges, 'pdg')
	for criterion in criterions:
		slice_nodes = forward_slice(criterion, pdg_nodes, pdg_edges)
		slice_nodes, slice_edges = construct_subgraph(slice_nodes, pdg_edges)
		if len(slice_edges) == 0 or len(slice_nodes) == 1:
			continue

		slice_content = {'nodes':slice_nodes,'edges':slice_edges}
		slices.append(slice_content)
	return slices


def io_slice(nodes, edges):
	slices = []

	criterions = get_operator_nodes(nodes, edges)
	if len(criterions) == 0:
		print("No operators.")
		return slices

	pdg_nodes, pdg_edges = get_graph(nodes, edges, 'pdg')
	for criterion in criterions:
		slice_nodes = bid_slice(criterion, pdg_nodes, pdg_edges)
		slice_nodes, slice_edges = construct_subgraph(slice_nodes, pdg_edges)
		if len(slice_edges) == 0 or len(slice_nodes) == 1:
			continue

		slice_content = {'nodes':slice_nodes,'edges':slice_edges}
		slices.append(slice_content)
	return slices


def np_slice(nodes, edges):
	slices = []

	def get_near_data_depend_node_location(ddg_nodes):
		location = [int(node['location']) for node in ddg_nodes if "location" in node.keys()]
		location.sort()
		if len(location)>2:
			loc = location[2]
		elif len(location) >1:
			loc = location[1]
		else:
			loc = location[0]
		return loc

	pointer = get_pointers(nodes, edges)
	callees = get_callee(nodes)
	if len(pointer) == 0 or len(callees) ==0:
		print("No pointer or callees.")
		return slices

	temp = []
	for node in nodes:
		if node['type'] == "EXPRSTATE" and "=" in node['code']:
			for item in pointer:
				if item in node['code']:
					temp.append(node)
					break
	criterions = []
	for node in temp:
		for item in callees:
			if item in node['code']:
				criterions.append(node)
				break

	ddg_nodes, ddg_edges = get_graph(nodes, edges, 'ddg')
	cfg_nodes, cfg_edges = get_graph(nodes, edges, 'cfg')

	for criterion in criterions:
		ddg_slice_nodes = forward_slice(criterion, ddg_nodes, ddg_edges)
		if len(ddg_slice_nodes) == 1:
			print("No null pointer error.")
			continue
		near_location = get_near_data_depend_node_location(ddg_slice_nodes)

		cfg_slice_nodes = forward_slice(criterion, cfg_nodes, cfg_edges)
		slice_nodes = [node for node in cfg_slice_nodes if "location" in node.keys() and int(node['location']) < near_location+1]

		slice_nodes, slice_edges = construct_subgraph(slice_nodes, cfg_edges)
		if len(slice_edges) == 0 or len(slice_nodes) == 1:
			continue

		slice_content = {'nodes':slice_nodes,'edges':slice_edges}
		slices.append(slice_content)
	return slices


def uaf_slice(nodes, edges):
	slices = []

	pointer = get_pointers(nodes, edges)
	callees = get_callee(nodes)
	if len(pointer) == 0 or len(callees) ==0:
		print("No pointer or callees.")
		return slices

	temp = []
	for node in nodes:
		if node['type'] == "EXPRSTATE" and "=" in node['code']:
			for item in pointer:
				if item in node['code']:
					temp.append(node)
					break
	criterions = []
	for node in temp:
		for item in callees:
			if item in node['code']:
				criterions.append(node)
				break			

	cfg_nodes, cfg_edges = get_graph(nodes, edges, 'cfg')
	for criterion in criterions:
		slice_nodes = forward_slice(criterion, cfg_nodes, cfg_edges)
		slice_nodes, slice_edges = construct_subgraph(slice_nodes, cfg_edges)
		if len(slice_edges) == 0 or len(slice_nodes) == 1:
			continue

		slice_content = {'nodes':slice_nodes,'edges':slice_edges}
		slices.append(slice_content)
	return slices



def df_slice(nodes, edges):
	slices = []

	pointer = get_pointers(nodes, edges)
	callees = get_callee(nodes)
	if len(pointer) == 0 or len(callees) ==0:
		print("No pointer or callees.")
		return slices

	temp = []
	for node in nodes:
		if node['type'] == "EXPRSTATE" and "=" in node['code']:
			for item in pointer:
				if item in node['code']:
					temp.append(node)
					break
	criterions = []
	for node in temp:
		for item in callees:
			if item in node['code']:
				criterions.append(node)
				break			

	ddg_nodes, ddg_edges = get_graph(nodes, edges, 'ddg')
	for criterion in criterions:
		slice_nodes = forward_slice(criterion, ddg_nodes, ddg_edges)
		slice_nodes, slice_edges = construct_subgraph(slice_nodes, ddg_edges)
		if len(slice_edges) == 0 or len(slice_nodes) == 1:
			continue

		slice_content = {'nodes':slice_nodes,'edges':slice_edges}
		slices.append(slice_content)
	return slices


def api_slice(nodes, edges):
	slices = []
	callees = get_callee(nodes)
	if len(callees) ==0:
		print("No callees.")
		return slices

	criterions = []
	for node in nodes:
		if node['type'] == "EXPRSTATE":
			for item in callees:
				if item in node['code']:
					criterions.append(node)
					break
	pdg_nodes, pdg_edges = get_graph(nodes, edges, 'pdg')
	for criterion in criterions:
		slice_nodes = bid_slice(criterion, pdg_nodes, pdg_edges)
		slice_nodes, slice_edges = construct_subgraph(slice_nodes, pdg_edges)
		if len(slice_edges) == 0 or len(slice_nodes) == 1:
			print("no slice nodes or edges")
			continue

		slice_content = {'nodes':slice_nodes,'edges':slice_edges}
		slices.append(slice_content)
	return slices

def api_slice_in_ddg(nodes, edges):
	slices = []
	callees = get_callee(nodes)
	if len(callees) ==0:
		print("No callees.")
		return slices

	criterions = []
	for node in nodes:
		if node['type'] == "EXPRSTATE":
			for item in callees:
				if item in node['code']:
					criterions.append(node)
					break
	pdg_nodes, pdg_edges = get_graph(nodes, edges, 'ddg')
	for criterion in criterions:
		slice_nodes = bid_slice(criterion, pdg_nodes, pdg_edges)
		slice_nodes, slice_edges = construct_subgraph(slice_nodes, pdg_edges)
		if len(slice_edges) == 0 or len(slice_nodes) == 1:
			print("no slice nodes or edges")
			continue

		slice_content = {'nodes':slice_nodes,'edges':slice_edges}
		slices.append(slice_content)
	return slices

def operator_slice(nodes, edges):
	slices = []

	criterions = get_operator_nodes(nodes, edges)
	if len(criterions) == 0:
		print("No operators.")
		return slices

	pdg_nodes, pdg_edges = get_graph(nodes, edges, 'pdg')
	for criterion in criterions:
		slice_nodes = bid_slice(criterion, pdg_nodes, pdg_edges)
		slice_nodes, slice_edges = construct_subgraph(slice_nodes, pdg_edges)
		if len(slice_edges) == 0 or len(slice_nodes) == 1:
			continue

		slice_content = {'nodes':slice_nodes,'edges':slice_edges}
		slices.append(slice_content)
	return slices

def pointer_slice(nodes, edges):
	slices = []

	pointers = get_pointers(nodes, edges)
	if len(pointers) == 0:
		print("No pointer.")
		return slices

	criterions = []
	for node in nodes:
		if node['type'] == "EXPRSTATE":
			for item in pointers:
				if item in node['code']:
					criterions.append(node)
					break			
	pdg_nodes, pdg_edges = get_graph(nodes, edges, 'pdg')
	for criterion in criterions:
		slice_nodes = bid_slice(criterion, pdg_nodes, pdg_edges)
		slice_nodes, slice_edges = construct_subgraph(slice_nodes, pdg_edges)
		if len(slice_edges) == 0 or len(slice_nodes) == 1:
			continue

		slice_content = {'nodes':slice_nodes,'edges':slice_edges}
		slices.append(slice_content)

	return slices

def array_slice(nodes, edges):
	slices = []

	arrays = get_arrays(nodes, edges)
	if len(arrays) == 0:
		print("No array.")
		return slices

	criterions = []
	for node in nodes:
		if node['type'] == "EXPRSTATE":
			for item in arrays:
				if item in node['code']:
					criterions.append(node)
					break			
	pdg_nodes, pdg_edges = get_graph(nodes, edges, 'pdg')
	for criterion in criterions:
		slice_nodes = bid_slice(criterion, pdg_nodes, pdg_edges)
		slice_nodes, slice_edges = construct_subgraph(slice_nodes, pdg_edges)
		if len(slice_edges) == 0 or len(slice_nodes) == 1:
			continue

		slice_content = {'nodes':slice_nodes,'edges':slice_edges}
		slices.append(slice_content)
		
	return slices


if __name__ == '__main__':
	dotfile = '1.dot'
	slices = get_slices_test(dotfile)
	print(len(slices))

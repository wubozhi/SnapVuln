import os
from scripts.bash_exe import execute_command
# from bash_exe import execute_command
def get_funcs_in_file(file_path):
    func_dict_list = []

    cmd2 = 'ctags --fields=+ne-t -o - --sort=no --excmd=number %s' % file_path
    res = execute_command(cmd2, ".")
    if not res:
        with open(file_path, "r", errors="ignore") as rfile:
            file_str = rfile.read()
        if not file_str:
            return func_dict_list
        else:
            print("Empty Ctags Result [%s]" % (file_path))
            return func_dict_list
            
    lines = res.splitlines()

    for line in lines:
        fields = line.split()
        if 'f' in fields:
            func_dict = {}
            start_num = get_num(fields, 'line:')
            end_num = get_num(fields, 'end:')
            func_dict['func'] = extract_function(file_path, start_num, end_num)
            func_dict['start_line'] = start_num
            func_dict['end_line'] = end_num
            func_dict['name'] = fields[0]
            # sometimes the ctag results are incomplete, we have to make do here
            if start_num is None or end_num is None:
                continue
            func_dict_list.append(func_dict)

    return func_dict_list


def get_num(fields, tag):
    try:
        for item in fields:
            if tag in item:
                tag_list = item.split(":")
                return int(tag_list[-1])
    except:
        print(fields, tag)


def extract_function(file_path, start_num, end_num):
    with open(file_path, "r", errors="ignore") as rfile:
        lines = rfile.readlines()
        return "".join(lines[start_num - 1:end_num])

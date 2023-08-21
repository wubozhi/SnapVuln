from subprocess import Popen, PIPE
import os

def execute_command(command,cwd):
	try:
		p = Popen(command,stdout=PIPE,stderr=PIPE,cwd=cwd,shell=True)
		content, _ = p.communicate()
		out = content.decode("utf8","ignore")
		return out
	except Exception as e:
		print(command)
		return ''

def execute_command_err(command,cwd):
	try:
		p = Popen(command,stdout=PIPE,stderr=PIPE,cwd=cwd,shell=True)
		content, err = p.communicate()
		out = err.decode("utf8","ignore")
		return out
	except Exception as e:
		print(command)
		return ''
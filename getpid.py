import os, tempfile, sys, datetime, logging, threading, time, shutil, requests, json, math
import os.path
from os import path
from shutil import copyfile
from distutils.dir_util import copy_tree
requests.packages.urllib3.disable_warnings()
def readcmd(cmd):
    ftmp = tempfile.NamedTemporaryFile(suffix='.out', prefix='tmp', delete=False)
    fpath = ftmp.name
    if os.name=="nt":
        fpath = fpath.replace("/","\\") # forwin
    ftmp.close()
    os.system(cmd + " > " + fpath)
    data = ""
    with open(fpath, 'r') as file:
        data = file.read()
        file.close()
    os.remove(fpath)
    return data


print(readcmd('netstat -tnlp | grep python3'))

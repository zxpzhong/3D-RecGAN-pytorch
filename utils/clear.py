# clear empty log
import os
import sys
import shutil
deleted = {}

models = 'saved/models/'
dirs = os.listdir(models)
for dir in dirs:
    logs = os.listdir(models+dir)
    for log in logs:
        files = os.listdir(models+dir+'/'+log)
        flag = 0
        for file in files:
            if file.endswith('pth'):
                flag = 1
                break
        if flag == 0:
            shutil.rmtree(models+dir+'/'+log)
            deleted[log] = 0

models = 'saved/log/'
dirs = os.listdir(models)
for dir in dirs:
    logs = os.listdir(models+dir)
    for log in logs:
        if log in deleted:
            shutil.rmtree(models+dir+'/'+log)

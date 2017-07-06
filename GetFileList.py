#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
def GetFiles(pwd):
	fileList = []
	for path, dirs, files in os.walk(pwd):
		for file in files:
			if os.path.splitext(file)[1].lower() == '.pos':
				fileList.append(os.path.join(path,file))
	return fileList
def GetFile_ABS(pwd):
	fileList = []
	for path, dirs, files in os.walk(pwd):
		for file in files:
			if os.path.splitext(file)[1].lower() == '.abs':
				fileList.append(os.path.join(path,file))
	return fileList
def GetFile_REL(pwd):
	fileList = []
	for path, dirs, files in os.walk(pwd):
		for file in files:
			if os.path.splitext(file)[1].lower() == '.body':
				fileList.append(os.path.join(path,file))
	return fileList

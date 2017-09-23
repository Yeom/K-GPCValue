#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
def GetFiles(pwd):
	fileList = []
	for path, dirs, files in os.walk(pwd):
		for file in files:
			if os.path.splitext(file)[1].lower() == '.txt':
				fileList.append(os.path.join(path,file))
	return fileList


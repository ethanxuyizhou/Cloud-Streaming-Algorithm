#!/usr/bin/env python
import sys
import re

def tokenizeDoc(curr_doc):
	return re.findall('\\w+', curr_doc)

for line in sys.stdin:
	class_labels = (line.split('\t')[1]).split(',')
	words = tokenizeDoc(line.split('\t')[2])
	
	for clas in class_labels:
		print("Y=*" + '\t'+"1")
		print("Y=" + clas + '\t' + "1")
		for word in words: print("Y=" + clas + ",W=" + word + '\t' + "1")
		print("Y=" + clas + ",W=*" + '\t' + str(len(words)))





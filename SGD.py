#!/usr/bin/env python
from __future__ import division
import sys
import math
import re

classes = dict()
classes['Person'] = 0
classes['Place'] = 1
classes['Species'] = 2
classes['Work'] = 3
classes['other'] = 4

vocab_size = sys.argv[1]
init_lr = sys.argv[2]
reg_coeff = sys.argv[3]
max_iters = sys.argv[4]
train_size = sys.argv[5]


stopwords = {'a', 'an', 'and','as', 'at','be','by','for','from','has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were','will', 'with'}

vocab_size = int(vocab_size)
init_lr = float(init_lr)
reg_coeff = float(reg_coeff)
max_iters = int(max_iters)
train_size = int(train_size)

testData = sys.argv[6]


def sigmoid(score):
	overflow = 20.0
	if score > overflow:
		score = overflow
	elif score < -overflow:
		score = -overflow
	exp = math.exp(score)
	return exp / (1 + exp)


def tokenizeDoc(curr_doc):
	return re.findall('\\w+', curr_doc)


def helper(b, x):
	total = 0
	for element in x:
		total += b[element]
	return total

def computer(listWords):
	result = []
	for word in listWords:
		if (word in stopwords): continue
		val = hash(word)%vocab_size
		result += [val]

	return result


k = 0
A = [[0]*vocab_size, [0]*vocab_size, [0]*vocab_size, [0]*vocab_size, [0]*vocab_size]
B = [[0]*vocab_size, [0]*vocab_size, [0]*vocab_size, [0]*vocab_size, [0]*vocab_size]


counter = 0

lr = 0

for ite in range(max_iters):

	lr = init_lr/((1+ite)**2)
	counter = 0
	update = (1-2*lr*reg_coeff)

	s = 0

	for line in sys.stdin:

		counter += 1
		if (counter > train_size): break

		k += 1
		trainClasses = (line.split('\t')[1]).split(',')
		imm = tokenizeDoc(line.split('\t')[2])

		trainWords = computer(imm)
		storeBX = [0]*5

		for i in range (5):
			storeBX[i] = helper(B[i], trainWords)

		for clas in classes:
			y = 0
			if (clas in trainClasses): y = 1
			val = classes[clas]

			p = sigmoid(storeBX[val])

			if (y==1): s += math.log(p)
			else: s += math.log(1-p)
			
			for j in trainWords:
				B[val][j] *= update**(k-A[val][j])
				B[val][j] += lr*(y-p)
				A[val][j] = k

	print(s)


update = 1-2*lr*reg_coeff

for i in range (5):
	for j in range (len(B[i])):
		B[i][j] *= update**(k-A[i][j])



with open(testData) as fp:

	line = fp.readline()
	while line:

		curr = 0
		for i in range (len(line)):
			if (line[i].isspace()): 
				curr = i
				break

		imm = tokenizeDoc(line[curr:])

		testWords = computer(imm)
		
		outStr = ''

		for clas in classes:
			actualClas =str(clas)
			outval = sigmoid(helper(B[classes[clas]], testWords))
			stri = str(outval)

			outStr += actualClas + '\t' + stri +','

		print(outStr[:-1])
	
		line = fp.readline()



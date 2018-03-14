#!/usr/bin/env python
from __future__ import division
from guineapig import *
import re
import sys
import math

#always subclass Planner
class NB(Planner):
	# params is a dictionary of params given on the command line. 
	# e.g. trainFile = params['trainFile']

	params = GPig.getArgvParams()
	train = ReadLines(params.get('trainFile', 'train')) | Map(by = lambda line:line.strip().split("\t"))
	test = ReadLines (params.get('testFile', 'test')) | Map(by = lambda line:(line.strip().split()[0], line.strip().split()[1:]))
	# main operations for testing

	# main operations for training
	mainR = Map(train, by = lambda a : (a[1].split(','), a[2].split()))
	main1= FlatMap(mainR, by = lambda (classes, words):(map(lambda clas: (clas, words), classes)))
	###

	getClasses = Map(main1, by = lambda (clas, words) : clas)
	distinctClasses = Distinct(getClasses)

	y = Map(main1, by = lambda a : 1)
	resultY = Group(y, by = lambda x:x, retaining = lambda x:x, reducingTo = ReduceToCount())

	yCla = Map(main1, by = lambda (clas, words):(clas))
 	resultCla = Group(yCla, by = lambda x:x, retaining = lambda x:x, reducingTo = ReduceToCount())

 	numClasses = Group(distinctClasses, by = lambda x : 1, retaining = lambda x : 1, reducingTo = ReduceToCount())

 	resultClaY = Join(Jin(resultY, by = lambda x : 1), Jin(resultCla, by = lambda x: 1))

 	mid = Augment(resultClaY, sideview = numClasses, loadedBy = lambda v : GPig.onlyRowOf(v))

 	mappedResult = Map(mid, by = lambda (((clas, countCla), (any1, countAny)), (a, value)) : (any1, math.log(countAny + 1) - math.log(countCla + value)))  #######need smoothing


	requestById = Map(test, by = lambda (x, y) : x)
	joinWithYCla = Join(Jin(requestById, by=lambda x : 1), Jin(mappedResult, by = lambda x : 1))

	#######################

	#total requests

	mainRequest = FlatMap(test, by = lambda (docid, words) : (map(lambda word : (docid, word), words)))
	distinctRequest = Map(mainRequest, by = lambda (docid, word) : word)
	distinct = Distinct(distinctRequest)
	distinctive = Group(distinct, by = lambda (word) : 1, retaining = lambda (word) : 1, reducingTo = ReduceToCount())
	mapDistinct = Map(distinctive, by = lambda (a, count) : count)

	requestClass = Group(mainRequest, by = lambda (docid, word) : docid, retaining = lambda (docid, word) : word, reducingTo = ReduceToCount())


	requestClassCount = Augment(requestClass, sideview = mapDistinct, loadedBy = lambda v: GPig.onlyRowOf(v))

	resultyClasW = Group(main1, by = lambda (x,y):x, retaining = lambda (x,y):len(y), reducingTo = ReduceToSum())

	classCount = Join(Jin(requestClassCount, by = lambda ((docid, count), offset) : 1), Jin(resultyClasW, by = lambda (clas, count) : 1))


	mappedClassCount = Map(classCount, by = lambda (((docid,count),offset), (clas2,count2)) : (docid, clas2, -count * (math.log(offset + count2))))


	yClasWord = FlatMap(main1, by = lambda (clas, words): (map(lambda word : (clas, word), words)))

	resultYW = Group(yClasWord, by = lambda (x,y):(x,y), retaining = lambda (x,y) : 1, reducingTo = ReduceToCount())


	joinYW = Join(Jin(resultyClasW, by = lambda (x, y):x), Jin(resultYW, by = lambda ((x, y), z): x))


	inter = Augment(joinYW, sideview = mapDistinct, loadedBy = lambda v : GPig.onlyRowOf(v))
	mappedYW = Map(inter, by = lambda (((clas1, ln), ((clas2, word2), ln2)), offset) : (clas1, word2, math.log(ln2 + 1) - math.log(ln + offset) + math.log(offset + ln)))      #######need smoothing


	la = Group(mainRequest, by = lambda (docid, word) : (docid, word), reducingTo = ReduceToCount())

	lb = Group(la, by = lambda ((docid, word), count) : word, retaining = lambda ((docid, word), count) : (docid, count))



	joinRequestYW = Join(Jin(lb, by = lambda (word, b):word), Jin(mappedYW, by = lambda (clas, word, value) : word))




	step = FlatMap(joinRequestYW, by = lambda (((word), b), (clas, word2, value)) : (map(lambda (docid, count) : (clas, word, docid, count*value), b)))

	finalYW = Group(step, by = lambda (clas, word, docid, count) : (docid, clas), retaining = lambda (clas, word, docid, count) : count, reducingTo = ReduceToSum())


	#joinRequestYW = Join(Jin(mainRequest, by = lambda (docid, word) : word), Jin(mappedYW, by = lambda (clas, word, value) : word))



	#finalYW = Group(joinRequestYW, by = lambda ((docid,word), (clas, word2, value)) : (docid, clas), retaining = lambda ((docid, word), (clas, word2, value)) : value, reducingTo = ReduceToSum())



	joinWithRequestClass = Join(Jin(finalYW, by = lambda ((docid, clas), count) : (docid, clas)), Jin(mappedClassCount, by = lambda (docid, clas, count) : (docid, clas)))

	furtherMap = Map(joinWithRequestClass, by = lambda (((docid1, clas1), count1), (docid2, clas2, count2)): (docid1, clas1, count1 + count2))

	joinTogether = Join(Jin(joinWithYCla, by = lambda (docid, (clas, count)) : (docid, clas)), Jin(furtherMap, by = lambda (docid, clas, value) : (docid, clas)))
	reducedTogether = Map(joinTogether, by = lambda ((docid, (clas, count)), (docid2, clas2, value)) : (docid, clas2, count + value)) #add the two log values together

	result = Group(reducedTogether, by = lambda (docid, clas, value) : docid, retaining = lambda (docid, clas, value) : (clas, value), reducingTo = ReduceTo(lambda : ('', -1 - sys.maxint), lambda accum, newTup : ((accum[0], max(accum[1], newTup[1])) if (accum[1] >= newTup[1]) else (newTup[0], max(accum[1], newTup[1])))))

	output = Map(result, by = lambda (docid, (clas, prob)) : (docid, clas, prob))



# always end like this
if __name__ == "__main__":
    NB().main(sys.argv)

# supporting routines can go here



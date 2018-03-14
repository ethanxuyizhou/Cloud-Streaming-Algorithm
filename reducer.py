#!/usr/bin/env python
import sys

previousKey = ""
sumForPrevious = 0

for line in sys.stdin:
	currKey = line.split('\t')[0]
	currVal = int(line.split('\t')[1])
	if (currKey == previousKey):
		sumForPrevious += currVal
	else:
		if (previousKey != ""): print(previousKey + '\t' + str(sumForPrevious))
		previousKey = currKey
		sumForPrevious = currVal

print(previousKey + '\t' + str(sumForPrevious))






lines=0
wordCount=0
mostWordsInLine = 0
#fileHandler=train.txt

result = [len(line.split()) for line in r]
lines = fileHandler.readlines()
result = [len(line.split()) for line in lines]
print(*('{} -- {}'.format(*item) for item in zip(lines, results)), sep='\n')


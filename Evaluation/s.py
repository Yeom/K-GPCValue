import string

with open('ans.txt') as f:
	for line in f:
		s = line[0].upper()+line[1:]
		print s.strip()

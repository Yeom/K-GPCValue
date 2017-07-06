#!/bin/sh 
perl stemmer.pl Combined_SemEval2010 > ans.txt
python s.py > result.txt
perl performance.pl result.txt

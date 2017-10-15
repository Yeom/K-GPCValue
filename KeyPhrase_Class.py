# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import argparse
import GetFileList as f
import KeyPhrase_Function as KF
import string
import re
import codecs
import operator
import time
from operator import itemgetter
import numpy as np
from scipy import stats
from konlpy.tag import Komoran
print 'Loading Part of Speech Tagger'
komoran = Komoran()
print 'Load End'
def preprocess_document(document):
    result = ''
    document = re.sub('\n',' ',document)
    document = re.sub('[ ]+',' ',document)
    tagged = komoran.pos(document)
    for word,tag in tagged:
        result += word+'_'+tag+' '
    return result.strip()
class Keyphrase():
    def __init__(self):
        pass 
    def __call__(self, *args):
        call_flag = False
        pwd = ''
        window_size = 0
        sentence_ = ''

        if 'pwd' in args[0].keys():
            pwd = args[0]['pwd']
            call_flag = True
        if 'window_size' in args[0].keys():
            window_size = args[0]['window_size']
        if 'sentence' in args[0].keys():
            sentence_ = args[0]['sentence']

        self.Final_Info = {}
        self.cValue_Info = {}
        self.word_Dic = {}
        self.document = []
        self.posTags = []
        self.position = {}
        self.upperDic = {}
        doc_idx = 0

        if call_flag:
            with codecs.open(pwd,'r',encoding='utf8') as doc:
                self.doc_Lines = preprocess_document(doc.read())
        else:
            self.doc_Lines = preprocess_document(sentence_)

        self.doc_Line = self.doc_Lines
        tokens = self.doc_Lines.split()
        self.doc_len = len(tokens)
        for token in tokens:
            word = token.split('_')[0]
            try : pos = token.split('_')[1]
            except : continue
            for i in xrange(len(word)):
                if word[i].isupper():
                    self.upperDic[word.lower()] = self.upperDic.get(word.lower(), 0.0) + 1.0
                    break
            word = word.strip().lower()
            self.word_Dic[word] = self.word_Dic.get(word,0) + 1.0
            self.document.append(word)
            self.posTags.append(pos)
                        
            flag1 = KF.isGoodPOS(pos)
            flag2 = KF.isSpecialToken(word)
            if(flag1 == False or flag2 == False):
                doc_idx = doc_idx +1
                continue
            doc_idx = doc_idx + 1
            if self.position.get(word) == None :
                position_list = []
                self.position[word] = position_list
                    
            self.position[word].append(doc_idx)
        p = re.compile('_[A-Z\.\!\@\#\$\%\^\&\*\?]+')
        dot = re.compile('[\.\!\@\#\$\%\^\&\*\?]')
        self.doc_Lines = p.sub('',self.doc_Lines)
        self.doc_Lines = dot.sub('',self.doc_Lines)
        self.doc_Lines = self.doc_Lines.lower()

    def Basic_GraphModel(self,window_size):
        self.srGraph,self.new_srGraph = KF.buildGraph(self.document, self.position,int(window_size), self.posTags) # Building Document Graph
        self.srGraph = KF.WeightGraph(self.word_Dic, self.srGraph, self.new_srGraph)#Adjust weighting scheme to graph edge
        self.srScore, self.srScore_avg = KF.BasicRankAlgorithm(self.srGraph, self.position) # Ranking Vertex Scores
        self.candidate_Clause = KF.extractPattern(self.document,self.posTags, self.srScore,self.word_Dic) # Extracting Keyphrase Candidates from a given document
        self.sortedKeyPhrase, self.keyP = KF.Scoring(self.srScore,self.candidate_Clause) # Scoinrg Keyphrase Candidates ( Sum / Modified Harmonic )

#Get KeyPhrases End now we have to make info for c-value
    def CValueModel(self):
        keyList = self.sortedKeyPhrase.keys()
        for key in keyList:
            if len(key.split()) >= 1:
                self.cValue_Info[key] = self.cValue_Info.get(key,0.0) + (self.sortedKeyPhrase[key] * (self.doc_Lines.count(key)))    #Phrase-C-Value
        self.FinalKeyPhrase = KF.srScoreXcvalue(self.cValue_Info,self.document)

###################################################################################
class FileOperations:
    def open_file_combined(self):
        self.file_ = open('./Result.txt','w')
    def close_file_combined(self):
        self.file_.close()
    def write_to_file_combined(self,file_name, Final_, num_key):
        self.file_.write(file_name+' : ')
        for idx in xrange(len(Final_)):
            key , val = Final_[idx]
            self.file_.write(key)
            if idx != num_key-1:
                self.file_.write(',')
            else:
                break
        self.file_.write('\n')

###################################################################################
def main(args):
    _module_ = Keyphrase()
    with open(args.data) as f_:
        for line in f_:
            data_path = line.strip()
            fileList = f.GetFiles(data_path)
            fileHandler = FileOperations()
            fileHandler.open_file_combined()
            for idx in xrange(len(fileList)):
                file_name = fileList[idx].split('/')[-1].split('.')[0]
                param = {}
                param['pwd'] = fileList[idx]
                param['window_size'] = args.window
                _module_(param)
                _module_.Basic_GraphModel(args.window)
                _module_.CValueModel()
                KeyPhrase = _module_.FinalKeyPhrase
                fileHandler.write_to_file_combined(file_name, KeyPhrase, int(args.key))
            fileHandler.close_file_combined()
#Run the model : python KeyPhrase_Class.py --data filepath.txt --key #(default 15) --window #(default 10)
def module(_module_,sentence, window_size=5, key=5):
    fileHandler = FileOperations()
    fileHandler.open_file_combined()
    param = {}
    Keyword = []
    param['sentence'] = sentence
    param['window_size'] = window_size
    _module_(param)
    _module_.Basic_GraphModel(window_size)
    _module_.CValueModel()
    KeyPhrase = _module_.FinalKeyPhrase
    
    for idx in xrange(key):
        key , val = KeyPhrase[idx]
        Keyword.append(key)
    return Keyword
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",help="data path file")
    parser.add_argument("--window",help="window size")
    parser.add_argument("--key",help="number of keyphrase")
    args = parser.parse_args()
    main(args)

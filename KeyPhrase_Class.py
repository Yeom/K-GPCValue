# -*- coding: utf-8 -*-
import argparse
import GetFileList as f
import KeyPhrase_Function as KF
import string
import re
import operator
import time
from operator import itemgetter
import numpy as np
from scipy import stats

class Keyphrase():
    def __init__(self,pwd,window_size = 10):

        with open(pwd) as doc:
            self.Final_Info = {}
            self.cValue_Info = {}
            self.key_path = pwd.split('/')[-1].split('.')[0]
            self.word_Dic = {}
            self.document = []
            self.posTags = []
            self.position = {}
            self.upperDic = {}
            doc_idx = 0
            self.doc_Lines = doc.read()
            self.doc_Line = self.doc_Lines
            tokens = self.doc_Lines.split()
            self.doc_len = len(tokens)
            for token in tokens:
                word = token.split('_')[0]
                try : pos = token.split('_')[1]
                except : print word
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

    def Basic_GraphModel(self,window_size,file_name,dataset,alpha):
        self.srGraph,self.new_srGraph = KF.buildGraph(self.document, self.position,int(window_size), self.posTags) # Building Document Graph
        self.srGraph = KF.WeightGraph(self.word_Dic, self.srGraph, self.new_srGraph)#Adjust weighting scheme to graph edge
        self.srScore, self.srScore_avg = KF.BasicRankAlgorithm(self.srGraph, self.position,dataset) # Ranking Vertex Scores
        self.candidate_Clause = KF.extractPattern(self.document,self.posTags, self.srScore,self.word_Dic) # Extracting Keyphrase Candidates from a given document
        self.candidate_Position = KF.extractPattern_Position(self.candidate_Clause,self.doc_Lines) # Extracting Keyphrase Candidate's first occurence position from a given document 
        self.sortedKeyPhrase, self.keyP = KF.Scoring(self.srScore,self.candidate_Clause, float(alpha)/10.0) # Scoinrg Keyphrase Candidates ( Sum / Modified Harmonic )

#Get KeyPhrases End now we have to make info for c-value
    def CValueModel(self,dataset,beta,position_Distribution):
        ###################Re-extract from another corpora######################
        interval = 0
        if dataset == "Inspec":
            interval = int(self.doc_len/12.5) #Inspec
        elif dataset == "SemEval2010":
            interval = int(self.doc_len/959) #SemEval
        ########################################################################
        keyList = self.sortedKeyPhrase.keys()
        for key in keyList:
            if len(key.split()) >= 1:
                try : self.cValue_Info[key] = self.cValue_Info.get(key,0.0) + (self.sortedKeyPhrase[key] * ((position_Distribution[int(self.candidate_Position[key][0]) / interval]) * self.doc_Lines.count(key))) # Position-Phrase-C-Value
                except : self.cValue_Info[key] = self.cValue_Info.get(key,0.0) + (self.sortedKeyPhrase[key] * ((position_Distribution[-1]) * self.doc_Lines.count(key)))

#                self.cValue_Info[key] = self.cValue_Info.get(key,0.0) + (self.sortedKeyPhrase[key] * (self.doc_Lines.count(key)))    #Phrase-C-Value
        self.FinalKeyPhrase = KF.srScoreXcvalue(self.cValue_Info,self.document,self.posTags, self.upperDic, self.candidate_Position, float(beta)/10.0)
                

def performance(dataset,windowsize,alpha,beta):
#    A_path = './Result/Experiment/Dev_set/'+dataset+'/Combine/combine_'+windowsize+'-'+str(alpha)+'-'+str(beta)
#    G_path = './Result/Experiment/Dev_set/'+dataset+'/Graph/graph_'+windowsize+'-'+str(alpha)+'-'+str(beta)
#    K_path = './Result/Experiment/Dev_set/Inspec/Inspec_Keyset'

    A_path = './Result/Experiment/Test_set/'+dataset+'/Combine/combine_'+windowsize+'-'+str(alpha)+'-'+str(beta)
    G_path = './Result/Experiment/Test_set/'+dataset+'/Graph/graph_'+windowsize+'-'+str(alpha)+'-'+str(beta)
    K_path = './Result/Experiment/Test_set/Inspec/Inspec_Keyset'

    Key_ = {}
    Ans_ = {}
    Gra_ = {}
    #data read
    with open(K_path) as f:
        for line in f:
            line = line.strip()
            file_name = line.split(' : ')[0].strip().lower()
            file_key = line.split(' : ')[-1].split(',')
            Key_[file_name] = file_key
    with open(A_path) as f:
        for line in f: 
            line = line.strip()
            file_name = line.split(' : ')[0].strip().lower()
            file_key = line.split(' : ')[-1].split(',')
            Ans_[file_name] = file_key
    with open(G_path) as f:
        for line in f: 
            line = line.strip()
            file_name = line.split(' : ')[0].strip().lower()
            file_key = line.split(' : ')[-1].split(',')
            Gra_[file_name] = file_key

    key_count = 0
    ans_count = 0
    gra_count = 0
    key_file_list = Key_.keys()
    ans_file_list = Ans_.keys()
    gra_file_list = Gra_.keys()
    key_ = [5,10,15]
    for idx in range(len(key_)):
        k_count = key_[idx]

        match = 0
        key_count = 0
        gra_count = 0
        for file_name in gra_file_list:
            gra_file_key = Gra_[file_name]
            key_file_key = Key_[file_name]
            if len(gra_file_key) < k_count:
                k_count = len(gra_file_key)       
            gra_count += k_count
            key_count += len(key_file_key)
        
            for i in range(k_count):
                gra_key = gra_file_key[i]
                for key in key_file_key:
                    if key.strip() == gra_key.strip():
                        match += 1
            k_count = key_[idx]
        
        precision = float(match)/gra_count*100
        recall = float(match)/key_count*100
        f1_score = (2*precision*recall)/(precision+recall)
        print "Single Model ( Graph ) %d keys"%(k_count)
        print "Match\t"+str(match) + "\tSystem\t" + str(gra_count) + "\tAnswer\t"+ str(key_count)
        print "Precision\t%.2f\tRecall\t%.2f\tF1-Score\t%.2f"%(precision,recall,f1_score)        


        match = 0
        key_count = 0
        ans_count = 0
        for file_name in ans_file_list:
            ans_file_key = Ans_[file_name]
            key_file_key = Key_[file_name]
            if len(ans_file_key) < k_count:
                k_count = len(ans_file_key)
            ans_count += k_count
            key_count += len(key_file_key)
        
            for i in range(k_count):
                ans_key = ans_file_key[i]
                for key in key_file_key:
                    if key.strip() == ans_key.strip():
                        match += 1
            k_count = key_[idx] 
        precision = float(match)/ans_count*100
        recall = float(match)/key_count*100
        f1_score = (2*precision*recall)/(precision+recall)
        print "Combined Model ( Graph + C-Value ) %d keys"%(k_count)
        print "Match\t"+str(match) + "\tSystem\t" + str(ans_count) + "\tAnswer\t"+ str(key_count)
        print "Precision\t%.2f\tRecall\t%.2f\tF1-Score\t%.2f"%(precision,recall,f1_score)

###################################################################################
class FileOperations:
    def open_file_combined(self,dataset,windowsize,alpha,beta):
#        self.file_ = open('./Result/Experiment/Dev_set/'+dataset+'/Combine/combine_'+str(windowsize)+'-'+str(alpha)+'-'+str(beta),'w')
#        self.file_ = open('./Result/Experiment/Test_set/'+dataset+'/Combine/combine_'+str(windowsize)+'-'+str(alpha)+'-'+str(beta),'w')
        self.file_ = open('./Evaluation//Combined_'+dataset,'w')
    def close_file_combined(self):
        self.file_.close()
    def open_file_graph(self,dataset,windowsize,alpha,beta):
#        self.file__ = open('./Result/Experiment/Dev_set/'+dataset+'/Graph/graph_'+str(windowsize)+'-'+str(alpha)+'-'+str(beta),'w')
#        self.file__ = open('./Result/Experiment/Test_set/'+dataset+'/Graph/graph_'+str(windowsize)+'-'+str(alpha)+'-'+str(beta),'w')
        self.file__ = open('./Evaluation//Graph_'+dataset,'w')
    def close_file_graph(self):
        self.file__.close()
    def write_to_file_combined(self,file_name, Final_):
        num_key = 15
        self.file_.write(file_name+' : ')
        for idx in xrange(len(Final_)):
            key , val = Final_[idx]
            self.file_.write(key)
            if idx != num_key-1:
                self.file_.write(',')
            else:
                break
        self.file_.write('\n')
    def write_to_file_graph(self,file_name, Final_):
        num_key = 15
        self.file__.write(file_name+' : ')
        for idx in xrange(len(Final_)):
            key , val = Final_[idx]
            self.file__.write(key)
            if idx != num_key-1:
                self.file__.write(',')
            else:
                break
        self.file__.write('\n')
###################################################################################
def main(args):
    if args.test:
        performance(args.test, args.window, args.alpha, args.beta)
    else:
        position_Distribution = []
        sum_ = 133530.0
        with open('./position') as f__:
            for line in f__:
                position_Distribution.append(float(line.strip())/sum_)

        with open(args.data) as f_:
            for line in f_:
                data_path = line.strip()
                try : dataset = data_path.split('/')[7]
                except :
                    print data_path
                    continue
                fileList = f.GetFiles(data_path)
                fileHandler = FileOperations()
                fileHandler_ = FileOperations()
                fileHandler.open_file_combined(dataset,args.window,args.alpha,args.beta)
                fileHandler_.open_file_graph(dataset,args.window,args.alpha,args.beta)
    
                for idx in xrange(len(fileList)):
                    file_name = fileList[idx].split('/')[-1].split('.')[0]
                    AKE1 = Keyphrase(fileList[idx], args.window)
                    AKE1.Basic_GraphModel(args.window,file_name,dataset, args.alpha)
                    AKE1.CValueModel(dataset, args.beta,position_Distribution)
                    KeyPhrase = AKE1.FinalKeyPhrase
                    KeyGraph = AKE1.keyP

                    fileHandler.write_to_file_combined(file_name, KeyPhrase)
                    fileHandler_.write_to_file_graph(file_name,KeyGraph)
                fileHandler.close_file_combined()
                fileHandler_.close_file_graph()
#Run the model : python KeyPhrase_Class.py --data filepath.txt --key #(default 15) --window #(default 10)
#Text the model : python KeyPhrase_Class.py --test SemEval2010(Dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",help="data path file")
    parser.add_argument("--window",help="input the window size")
    parser.add_argument("--alpha",help="Ranking Alpha")
    parser.add_argument("--beta",help="Ranking Beta")
    parser.add_argument("--test",help="if you test the data")
    args = parser.parse_args()
    main(args)

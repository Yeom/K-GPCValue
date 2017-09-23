# -*- coding: utf-8 -*-
import sys
reload(sys)
import time
import numpy as np
from scipy import stats
import operator
import re
import string
import math
import random

def isGoodPOS(pos):
    if pos == "NNG" or pos == "NNP" or pos == "NNB" or pos == "VV" or pos == "VA" or pos == "VX" :
        return True
    else:
        return False

def isSpecialToken(word):
    p = re.compile('[#]+')
    m = p.match(word)
    if word == '%' or m != None or word == '=' or len(word) == 1:
        return False
    else:
        return True
def buildGraph(document,position,windowSize, posTags):
    srGraph = {}
    new_srGraph = {}
    idx = 0
    for word in document:
        if word in position.keys() :#
            adjacency_vertex = {}
            index = idx - 1
            count = windowSize
            while index >= 0 and count > 0:
                neighbor = document[index]
                if neighbor in position.keys():
                    adjacency_vertex[neighbor] = adjacency_vertex.get(neighbor,0.0) + 1.0
                index = index - 1
                count = count - 1
            
            index = idx +1
            count = windowSize
            
            while index < len(document) and count > 0:
                neighbor = document[index]
                
                if neighbor in position.keys() :
                    adjacency_vertex[neighbor] = adjacency_vertex.get(neighbor,0) + 1.0
# 
                index = index + 1
                count = count - 1

            
            if word in srGraph:
                Dic1 = srGraph[word]
                Dic2 = adjacency_vertex
                Dic = sum_two_dics(Dic1, Dic2)
                srGraph[word] = Dic
                new_srGraph[word] = Dic
            else:
                srGraph[word] = adjacency_vertex #이렇게하면 그래프 단어 겹치면 하나로 다시 통합되서 들어가게됨.
                new_srGraph[word] = adjacency_vertex
        idx = idx + 1
    return srGraph, new_srGraph

def sum_two_dics(dic1,dic2):
    dic = dic1
    key_list = dic2.keys()
    for key in key_list:
        if key in dic:
            temp = dic[key]
            dic[key] = temp + dic2[key]
        else:
            dic[key] = dic2[key]
    return dic

def normalize(srGraph, new_srGraph):
    vtx_list = srGraph.keys()
    for vtx in vtx_list:
        sum_ = 0.0
        adjacency_vtx_list = srGraph[vtx].keys()
        for adjacency_vtx in adjacency_vtx_list:
            sum_ = sum_ + srGraph[vtx][adjacency_vtx]
        for adjacency_vtx in adjacency_vtx_list:
            if sum_ != 0.0:#1.0-?
                new_srGraph[vtx][adjacency_vtx] =srGraph[vtx][adjacency_vtx]/sum_
            else:
                new_srGraph[vtx][adjacency_vtx] = 0.0
    return new_srGraph


def WeightGraph(frequency_Dic,srGraph,new_srGraph):
    KeyList = {}

    max_val = 0.0
    weight_sum = 0.0
    weight_count = 0.0
    srGraphTemp = srGraph
    vtx_list = srGraphTemp.keys()
    for vtx in vtx_list:
        adjacency_vtx_list = srGraphTemp[vtx].keys()
        for adjacency_vtx in adjacency_vtx_list:
            attr = ((srGraphTemp[vtx][adjacency_vtx]) / (frequency_Dic[vtx] * frequency_Dic[adjacency_vtx]))
#Adjust Weight
            srGraph[vtx][adjacency_vtx] = attr
            weight_sum = weight_sum + attr
            weight_count = weight_count + 1
    if weight_count > 0:
        weight_avg = weight_sum / weight_count
    else :
        weight_avg = weight_sum / 1.0
    biased_weight = 0.0

    for vtx in vtx_list:
        adjacency_vtx_list = srGraph[vtx].keys()
        for adjacency_vtx in adjacency_vtx_list:
            biased_weight = 1 + (srGraph[vtx][adjacency_vtx]-weight_avg)
            srGraph[vtx][adjacency_vtx] = biased_weight

    norm_srGraph = normalize(srGraph,new_srGraph)
    return norm_srGraph
   
def nodeCalculate(srGraph):
    nodeCount = 0
    vtx_list = srGraph.keys()
    for vtx in vtx_list:
        adj_vtx_list = srGraph[vtx].keys()
        for adj_vtx in adj_vtx_list:
            if srGraph[vtx][adj_vtx] != 0:
                nodeCount = nodeCount+1
    return nodeCount / 2


def BasicRankAlgorithm(srGraph,position):
    #print alpha
    srScore = {}
    pos_prob = {}
    vtx_list = srGraph.keys()
    nodeCount = nodeCalculate(srGraph)
    #Initialize Score
    for vtx in vtx_list:
        if vtx in position.keys():
            srScore[vtx] = 1.0
    #iteration
    breakFlag = False

    for i in range(20):
        srScoreDiff = srScore
        srScoreTemp = {}
        #vertices in Graph
        for vtx in vtx_list:
            score = 0.0
            adjacency_vtx_list = srGraph[vtx].keys()
            #adjacency vertex
            for adjacency_vtx in adjacency_vtx_list:
                score = score + srGraph[adjacency_vtx][vtx] * srScore[adjacency_vtx]
            score = score * (1.0 - 0.15)
            score = score + (0.15/(1.0*nodeCount))
            srScoreTemp[vtx] = score
            if abs(srScoreTemp[vtx] - srScoreDiff[vtx]) < 0.0001:
                breakFlag = True        
        srScore = srScoreTemp
        if breakFlag == True:
            break
    avg_ = 0.0
    for _ in srScore.keys():
        avg_ += srScore[_]
    avg_ /= len(srScore.keys())
    return srScore, avg_

def extractPattern(document,posTags,srScore,word_Dic):
    candidate_Clause = {}
    pattern = []
    patternPos = []
    idx = 0
    for pos in posTags:
        if (isGoodPOS(pos) == True and isSpecialToken(document[idx]) ):
            pattern.append(document[idx])
            patternPos.append(posTags[idx])       
        elif len(pattern) > 4:
            pattern = []
            patternPos = []
        elif len(pattern) != 0 and (isGoodPOS(pos) == False or  isSpecialToken(document[idx]) == False):

            s = " ".join(pattern).strip()
            
            if patternPos[-1] != 'JJ'and patternPos[-1] != 'JJS'and patternPos[-1] != 'JJR'and patternPos[-1] != "FW"and patternPos[-1] != "VBN" and patternPos[-1] != "CC" and patternPos[-1] != "IN" and patternPos[0] != "CC" and patternPos[0] != "IN":
                if s != "and":
                    candidate_Clause[s] = candidate_Clause.get(s,0) + 1
            pattern = []
            patternPos = []
            
        idx = idx + 1
    if len(pattern) != 0 :
        s = " ".join(pattern).strip()
        if len(pattern) > 4:
            pattern = []
            patternPos = []

        elif patternPos[-1] != 'JJ'and patternPos[-1] != 'JJS'and patternPos[-1] != 'JJR'and patternPos[-1] != "FW"and patternPos[-1] != "VBN" and patternPos[-1] != "CC" and patternPos[-1] != "IN":
            candidate_Clause[s] = candidate_Clause.get(s,0) + 1

        pattern = []
        patternPos = []
    
    return candidate_Clause

def Scoring(srScore,candidate_Clause):
    topKey = {}
    keyword_list = srScore.keys()
    cand_list = candidate_Clause.keys()
    for candidate in cand_list:
        if candidate_Clause[candidate] != None:
            topKey[candidate] = harmonicMean(srScore,candidate)
    sortedKeyPhrase = sorted(topKey.iteritems(),key = operator.itemgetter(1), reverse = True)
    return topKey,sortedKeyPhrase

def harmonicMean(srScore,candidate):
    d = 0.0
    token_list = candidate.split()
    for token in token_list:
        try : d = d + 1 / srScore[token]
        except : print token, token_list
    len_ = len(token_list)
    try : score = 1 / (d / len_)
    except : score = 0.0
    return score
def srScoreXcvalue(cVInfo,document):

    keyList = cVInfo.keys()
    cValue = {}
    for candidate in keyList:
        if len(candidate.split()) > 0:
            C_value = calCvalue(candidate,cVInfo)
            cValue[candidate] = 1.0*C_value

    sortCvalue = sorted(cValue.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sortCvalue

def calCvalue(candidate, cand_List):
    keyList = cand_List.keys()
    count = 0.0
    f_b = 0.0
    f_a = cand_List[candidate]
    for key in keyList:
        if key != candidate and isSubToken(candidate, key) == True:
            count = count + 1.0
            f_b = f_b + cand_List[key]
    if count == 0.0:
        return f_a
    else :
        return (f_a - (f_b/count))
def isSubToken(candidate, key):
    k_ = key.split(' ')
    c_ = candidate.split(' ')
    len_c_ = len(c_)
    range_ = len(k_) - len_c_ + 1
    if range_ > 0:
        for i in xrange(range_):
            if " ".join(k_[i:i+len_c_]) == candidate:
                return True
        return False
    else:
        return False


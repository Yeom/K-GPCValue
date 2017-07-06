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
def stopWordList(word):
    stopList = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each","effort", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill","mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere",  "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"]
    if word in stopList:
        return False
    return True

def isGoodPOS(pos):
    if pos == "NN" or pos == "NNS" or pos == "NNP" or pos == "NNPS" or pos == "JJ" or pos == "JJR"or pos == "JJS" or pos == "FW" or pos == "VBN":
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


def BasicRankAlgorithm(srGraph,position, dataset):
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
        if (isGoodPOS(pos) == True and isSpecialToken(document[idx]) and stopWordList(document[idx])):
            pattern.append(document[idx])
            patternPos.append(posTags[idx])       
        elif len(pattern) > 4:
            pattern = []
            patternPos = []
        elif len(pattern) != 0 and (isGoodPOS(pos) == False or stopWordList(document[idx]) == False or isSpecialToken(document[idx]) == False):

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

def extractPattern_Position(candidate_Clause,doc_Lines):
    ans_pos = {}
    document = doc_Lines
    document = re.sub("[\r]", '', document)
    document = re.sub("[\t]", '', document)
    document = re.sub("[\n]", '', document)
    document = re.sub("[ ]+", ' ',document)
    doc = document.split(' ')
    for candidate in candidate_Clause.keys():
        find_  = False
        cand = candidate.split(' ')
        len_cand = len(cand)
        len_doc  = len(doc) - len_cand + 1
        if len_doc > 0:
            for i in xrange(len_doc):
                if " ".join(doc[i:i+len_cand]) == candidate:
                    ans_pos[candidate] = ans_pos.get(candidate, []) + [i]
                    find_ = True
    return ans_pos
def Scoring(srScore,candidate_Clause,alpha):
    topKey = {}
    keyword_list = srScore.keys()
    cand_list = candidate_Clause.keys()
    for candidate in cand_list:
        if candidate_Clause[candidate] != None:
            len_ = len(candidate.split())
#            topKey[candidate] = getSumScore(srScore,candidate)
#            topKey[candidate] = harmonicMean(srScore,candidate)

            para_ =  (math.log(len(candidate.split()),2))
            if para_ == 0.0:
                para_ = alpha
            topKey[candidate] = harmonicMean(srScore,candidate) * para_


#            topKey[candidate] = arithmeticMean(srScore,candidate)
#            topKey[candidate] = geometricMean(srScore,candidate)
    sortedKeyPhrase = sorted(topKey.iteritems(),key = operator.itemgetter(1), reverse = True)
    return topKey,sortedKeyPhrase
def arithmeticMean(srScore,candidate):
    d = 0.0
    token_list = candidate.split()
    for token in token_list:
        d = d + srScore[token]
    score = d/len(token_list)
    return score
def geometricMean(srScore,candidate):
    token_list = candidate.split()
    len_ = len(token_list)
    d = 1.0
    for token in token_list:
        d = d * srScore[token]
    score = (d ** (1.0/float(len_)))
    return score
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
def getSumScore(srScore,candidate):
    token_list = candidate.split()
    d = 0.0
    for token in token_list:
        d = d + srScore[token]       
    return d

def srScoreXcvalue(cVInfo,document,posTags, upperDic, candidate_Position,beta):

    keyList = cVInfo.keys()
    cValue = {}
#    avg_score = 0.0
    for candidate in keyList:
        if len(candidate.split()) > 0:
            para_ =  (math.log(len(candidate.split()),2))
            if para_ == 0.0:
                para_ = beta

            C_value = calCvalue(candidate,cVInfo) * para_
            cValue[candidate] = 1.0*C_value
#            avg_score += C_value
    
#    avg_score /= len(keyList)
#    print avg_score
#    for candidate in keyList:
#        candidate_element = candidate.split(' ')
#        for element in candidate_element:
#            if missed_keyword(element):
#                cValue[candidate] = cValue.get(candidate, 0.0 ) + avg_score
#            if '-' in element:
#                cValue[candidate] = cValue.get(candidate, 0.0 ) + avg_score
#            if element in upperDic:
#                cValue[candidate] = cValue.get(candidate, 0.0 ) + avg_score

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
def wordCount(key, document):
    k_ = document.split(' ')
    c_ = key.split(' ')
    len_c_ = len(c_)
    range_ = len(k_) - len_c_ + 1
    cnt = 0
    if range_ > 0:
        for i in xrange(range_):
            if " ".join(k_[i:i+len_c_]) == key:
                cnt += 1
    return cnt
def missed_keyword(kwd):
    missed_ = ['system','distribut','search','model','algorithm','quer','network','comput','inform','mechan','perform','auction','agent','market','retriev','theor']
    for miss in missed_:
        _ = re.compile(miss)
        if bool(_.match(kwd)):
            return True
    return False









"""
Experimented Extracting Candidate Pattern


def extractPattern_RE(document):
#    document = 'Scalable_JJ Grid_NNP Service_NNP Discovery_NNP Based_VBD on_IN UDDI_NNP *_SYM *_SYM Authors_NNPS are_VBP listed_VBN in_IN alphabetical_JJ order_NN ._.'
    _FILTER_ = '(( |[\w\-]+_NN|[\w\-]+_NNS|[\w\-]+_NNP|[\w\-]+_NNPS|[\w\-]+_JJ|[\w\-]+_VBN|[\w\-]+_NN [\w\-]+_IN|[\w\-]+_NNS [\w\-]+_IN)* ([\w\-]+_NN|[\w\-]+_NNS|[\w\-]+_NNP|[\w\-]+_NNPS|[\w\-]+_VBG))'
    _ERASEPOS_ = '_[A-Z]+'
    candidate_ = re.findall(_FILTER_,document)
    candidate_Clause = {}
    for _ in candidate_:
        for candidate in _:
    #        print candidate
            e = re.sub(_ERASEPOS_,'',candidate)
            e = e.lower().strip()
            b = e.split(' ')
            f_ = False
            for a in b:
                if len(a) == 1:
                    f_ = True
                    break
            if f_:
                continue
            candidate_Clause[e] = candidate_Clause.get(e, 0.0) + 1.0
#    sorted_candidate_Clause = sorted(candidate_Clause.iteritems(), key = operator.itemgetter(1), reverse = True)
    return candidate_Clause

def eraseSpecialToken(word):
    p = re.compile('[^0-9a-zA-Z\-\/ ]+')
    p_ = re.compile('[ ]+')
    word = p.sub('',word).strip()
    word = p_.sub(' ',word).strip()
    return word
def extractPattern_Ngram(document, srScore, avg):
    candidate_Ngram_ = []
    for n in xrange(5):
        output = []
        if n == 0:
            continue
        for i in xrange(len(document)- n + 1):
            output.append(document[i:i+n])
        candidate_Ngram_ += [" ".join(x) for x in output]
    for idx in xrange(len(candidate_Ngram_)):
        _ = candidate_Ngram_[idx]
        candidate_Ngram_[idx] = eraseSpecialToken(_)
    #중복제거
    candidate_Ngram = list(set(candidate_Ngram_))
    tmp_candidate_Ngram = []
    for candidate in candidate_Ngram:
        insert_Flag = False
        element = candidate.strip().split(' ')
        for e in element:
            if stopWordList(e) == True:
                pass
            else:
                insert_Flag = False
                break
            if e in srScore and srScore[e] > avg:
                insert_Flag = True
            else:
                insert_Flag = False
                break
        if insert_Flag:
            tmp_candidate_Ngram.append(candidate)
    _ = {k : 0 for k in tmp_candidate_Ngram}
    return _
"""

"""
Experimented Scoring Method

def getMulScore(srScore,candidate):
    token_list = candidate.split()
    d = 1.0
    for token in token_list:
        d = d * srScore[token]
    return d
"""

"""
Experimented N-Value

def isContextPos(posTag):
    if (posTag == "NN" or    posTag == "NNS" or    posTag == "NNP" or    posTag == "NNPS" or    posTag == "JJ" or    posTag == "JJS" or    posTag == "JJR" or    posTag == "VB" or    posTag == "VBD" or    posTag == "VBZ" or    posTag == "VBG" or    posTag == "VBN" or    posTag == "VBP" or    posTag == "DT"):
        return True
    else:
        return False
def extractTermContextWord(candidate,document,posTags):
    keyword = candidate.split()
    cand_context_term = {}
    for idx in range(0,len(document)-len(keyword)+1):
        flag = False
        for index in range(idx,idx+len(keyword)):
            if document[index] == keyword[index - idx]:
                flag = True
            else:
                flag = False
            if flag == False:
                break;
        if flag == True and isContextPos(posTags[idx-1]):
            cand_context_term[document[idx-1]] = cand_context_term.get(document[idx-1],0) + 1.0
#        if flag == True and isContextPos(posTags[idx-2]):
#            cand_context_term[document[idx-2]] = cand_context_term.get(document[idx-2],0) + 1.0
    return cand_context_term
def calNvalue(candidate,document,posTags,TCW):
    sumV = 0.0
    TCW_Info = extractTermContextWord(candidate,document,posTags)
    KK = TCW_Info.keys()
    for cw in KK:
        sumV = sumV + float(TCW_Info[cw])*TCW.get(cw,0)
    return sumV
"""

"""
def extractClause_Position(doc_Lines, candidate_Clause):
    candidate_Clause = list(set(candidate_Clause))
    clause_position = {}
    for candidate in candidate_Clause:
        clause_position[candidate] = [m.start()+1 for m in re.finditer(candidate, doc_Lines)]
    return clause_position

def isInTrainDic(train_dic,phrase1,phrase2):
    word1 = phrase1.split(' ')
    word2 = phrase2.split(' ')
    words = []
    flag = True
    for word in word1:
        if word in train_dic:
            flag = True
        else:
            flag = False
            break
    if flag == False:
        return flag

    for word in word2:
        if word in train_dic:
            flag = True
        else:
            flag = False
            break
    return flag
def ReadTrainKey():
    train_dic = {}
    with open('train_keyword_set.txt') as f:
        for line in f:
            train_dic[line.strip()] = train_dic.get(line.strip(),0)+1
    return train_dic
"""

"""
def Candidate_buildGraph(candidate_Clause, windowSize):
    srGraph = {}
    new_srGraph = {}
    idx = 0
    for candidate in candidate_Clause:
        adjacency_vertex = {}
        index = idx - 1
        count = windowSize
        while index >=0 and count > 0:
            neighbor = candidate_Clause[index]
            adjacency_vertex[neighbor] = adjacency_vertex.get(neighbor, 0.0) + 1.0
            index = index - 1
            count = count - 1

        index = idx+1
        count = windowSize

        while index < len(candidate_Clause) and count > 0:
            neighbor = candidate_Clause[index]
            adjacency_vertex[neighbor] = adjacency_vertex.get(neighbor, 0.0) + 1.0
            index = index + 1
            count = count - 1

        if candidate in srGraph:
            Dic1 = srGraph[candidate]
            Dic2 = adjacency_vertex
            Dic = sum_two_dics(Dic1, Dic2)
            srGraph[candidate] = Dic
            new_srGraph[candidate] = Dic
        else:
            srGraph[candidate] = adjacency_vertex
            new_srGraph[candidate] = adjacency_vertex
        idx = idx + 1
    return srGraph, new_srGraph
def extractPattern_First(document,posTags):
    candidate_Clause = []
    pattern = []
    patternPos = []
    idx = 0
    for pos in posTags:
        if isGoodPOS(pos) == True and isSpecialToken(document[idx]) and stopWordList(document[idx]) :
            pattern.append(document[idx])
            patternPos.append(posTags[idx])
        
        elif len(pattern) != 0 and (isGoodPOS(pos) == False or stopWordList(document[idx]) == False or isSpecialToken(document[idx]) == False):
            s = " ".join(pattern).strip()
            if patternPos[-1] != 'JJ'and patternPos[-1] != 'JJS'and patternPos[-1] != 'JJR'and patternPos[-1] != "FW"and patternPos[-1] != "VBN" and patternPos[-1] != "CC" and patternPos[-1] != "IN" and patternPos[0] != "CC" and patternPos[0] != "IN":
                candidate_Clause.append(s)
            pattern = []
            patternPos = []
            
        idx = idx + 1
    if len(pattern) != 0 :
        s = " ".join(pattern).strip()
        if patternPos[-1] != 'JJ'and patternPos[-1] != 'JJS'and patternPos[-1] != 'JJR'and patternPos[-1] != "FW"and patternPos[-1] != "VBN" and patternPos[-1] != "CC" and patternPos[-1] != "IN":
            candidate_Clause.append(s)
        pattern = []
        patternPos = []
    return candidate_Clause
def clauseDictionary(doc_Lines, candidate_Keyphrase):
    doc_Lines = re.sub('[\s]+',' ',doc_Lines)
    doc_Lines = re.sub('[ ]+',' ',doc_Lines)
    candidate_Keyphrase = list(set(candidate_Keyphrase))
    clause_Dic = {}
    for candidate in candidate_Keyphrase:
        clause_Dic[candidate] = clause_Dic.get(candidate, 0.0) + doc_Lines.count(candidate)
        if clause_Dic[candidate] == 0.0:
            print candidate
    return clause_Dic

def ensembleWeight(srGraph,train_dic):
    vtx_list = srGraph.keys()
    alpha = 1.0 
    new_result = 0.0
    word_sim = 0.0
    for vtx in vtx_list:
        adjacency_vtx_list = srGraph[vtx].keys()
        for adjacency_vtx in adjacency_vtx_list:
            #new_result = alpha*srGraph[vtx][adjacency_vtx] + (1-alpha)*(word_sim+1)
            new_result = alpha*srGraph[vtx][adjacency_vtx]
            srGraph[vtx][adjacency_vtx] = new_result
            new_result = 0.0
            word_sim = 0.0
    return srGraph

"""

'''
this file contains 4 main funtions:
1. 'parseFile' Function to create inverted index from cf74-cf79 xml files;
2. 'readQueryXml' Function to read from 'cfquery.xml' file in which would call the 
3. 'processQuery' Function(which is also the main function in 'server.py') to process and retieve for each query in 'cfquery.xml' file,
and then the 'readQueryXml' Function would evaluate the predications made and return the 3 value sets to plot graphs;
Graphs are drawn with matplotlib in Jupyternotebook and would be inclueded in documentation
'''

import collections
import xml.etree.ElementTree as ET
import os
import re
import string
import time
import math
import heapq


'''
the main function that evaluates our query system and genrates evaluation statistics for ploting graphs
first, read the xml file that stores the standard query results,
then loop through each query and calculate the precision at k by calling the 'calcPrecAtK' function, store value in 'pAt10' list,
also calculate the average precision at k by calling the 'calcAPAtK' function, store value in 'apAt10' list,
calculate the interpolated precision at k value list which has 11 elements by calling the 'generateInterpolatedPList' function
and insert these 11 values into 'interpolatedDct' dictionary,
which uses 11 recall levels(*10) as key and a precision list of all queries as value
'''


def readQueryXml(xmlFile):
    # parse xml file
    tree = ET.parse(xmlFile)
    # get root element
    root = tree.getroot()
    # the P@10 list stores the precision scores at 10 from query1-query100
    pAt10 = []
    # AP list stores the average precision scores at 10 from query1-query100
    apAt10 = []
    # dictionary store the interpolated precision for all queries,
    # recall level(k/10) as key, precision list of all queries as value
    interpolatedDct = {}
    # initialize interpolated dictionary
    for i in range(11):
        interpolatedDct[i] = []
    for child in root:
        # get query
        query = child[1].text.lower()
        # call the 'processQuery' function to get the top10 results
        top10 = processQuery(query)[:10]
        # get result number
        queryNum = int(child[2].text)
        # get result list
        resList = []
        for item in child[3]:
            resList.append(int(item.text))
        # call the 'calcPrecAtK' function to calculate the precision at 10
        pAt10.append(calcPrecAtK(top10, resList, 10))
        # call the 'calcAPAtK' function to calculate the average precison at 10
        ap, correctPredNum = calcAPAtK(top10, resList, 10)
        apAt10.append(ap)
        # calculate the interpolated p@k value list for k in (0,10), for each queries
        pList = generateInterpolatedPList(
            top10, resList, correctPredNum)
        # insert 11 element from p list to each recall level
        for i in range(11):
            interpolatedDct[i].append(pList[i])

    MAP = sum(apAt10)/len(apAt10)
    PRList = []
    for i in range(11):
        PRList.append(
            round(sum(interpolatedDct[i])/len(interpolatedDct[i]), 3))
    return pAt10, MAP, PRList


'''
function to calculate the precision at k
'''


def calcPrecAtK(predication, resList, k):
    l = [x for x in predication if x in resList]
    return len(l)/k


'''
function to calculate and return the average precison at k,
also return the total correct predication number @k, for the 'interpolatePAtK' function to calculate recall
'''


def calcAPAtK(predication, resList, k):
    runningSum = 0
    correctPred = 0
    for i in range(1, k+1):
        if predication[i-1] in resList:
            correctPred += 1
        runningSum += correctPred/i
    # handle divided by 0 error if correct prediction is 0
    if(correctPred == 0):
        return 0, 0
    return runningSum/correctPred, correctPred


'''
given the predication list @k, the result list for query from xml file,
the total correct predication number @k and k which is 10 here,
return the inperpolated P@k list with 11 elements,
the index of every element represents the 11 recall levels(in the 'readQueryXml' function we'll use them after divided by 10)
'''


def generateInterpolatedPList(predication, resList, totalCorrectNum, k=10):
    precList = []
    recallList = []
    # the lists record the precision value according to the 11 recall level
    pList = []
    correctPred = 0
    for i in range(1, k+1):
        if predication[i-1] in resList:
            correctPred += 1
        precList.append(correctPred/i)
        if totalCorrectNum == 0:
            recall = 0
        else:
            recall = correctPred/totalCorrectNum
        recallList.append(recall)
    # if we find any relevant documents in our predication
    if (max(recallList) != 0):
        # the index indicating the start point of new max precision value's recall level
        start = 0
        # break while loop if we got the 11 precision values or the start index is out of precList range
        while(len(pList) < 11 or start < 10):
            # find the max precision value between start index to the end of list
            maxPrec = max(precList[start:])
            # find the index of last occurence of max precision value,
            # add 1 to the corresponding recall value multiplied by 10
            # since we need it to define insert range in the precision value list which use index to represent recall level
            rLevel = int(
                10 * recallList[len(precList) - 1 - precList[::-1].index(maxPrec)]) + 1
            # insert the max precision value from start index to rLevel index(corresponding recall level of max precison value)
            for i in range(start, rLevel):
                pList.append(maxPrec)
            # update new start index
            start = rLevel
    # return a list of 0s indicating we didn't find any thing relevant in this query
    else:
        return[0 for i in range(11)]

    return pList


'''
given the dictionary recording the occurences of each term,the function calculates the term frequency,
dividing each occurence number by the total occurence number, and replace the occurence number with the TF
'''


def normTF(dct):
    # calculate the total occurence of all terms
    total = sum(dct.values(), 0.0)
    # calculate the frequency for each term
    dct = {k: v / total for k, v in dct.items()}
    return dct


'''
 given the docID(i), the query terms TF*IDF dictionary and the document terms TF*IDF dictionary,
 the function calculates the Cosine Similarity for each term in every doc
'''


def calcCosineSimilarity(i, queryTFIDF, docTFIDF):
    vec1 = queryTFIDF.values()
    # vector storing the doc TF*IDF
    vec2 = []
    # because we don't have the records of docs that doesn't have these terms
    # we are inserting those docs' TF*IDF value as 0 into vec2
    for item in docTFIDF.keys():
        # if term appears in current doc
        if docTFIDF[item].get(i) != None:
            vec2.append(docTFIDF[item][i])
        # if term doesn't appear in current doc
        else:
            vec2.append(0)
    dotProduct = sum(i*j for i, j in zip(vec1, vec2))
    queryNorm = math.sqrt(sum(i*i for i in vec1))
    docNorm = math.sqrt(sum(i*i for i in vec2))
    return dotProduct/queryNorm*docNorm


'''
main function to process query, calculate TF*IDF for terms based on 
the TF, IDF dictionary created when creating inverted index in the 'parseFile' function
calculate the Cosine Similarity for every document and then return the top20 relevant docs
'''


def processQuery(queryStr):
    stopWords = processStopist('stoplist.txt')
    # get punctuation list
    puncsLst = list(string.punctuation)
    # remove all stop words
    queryStr = removeStopWords(queryStr, stopWords)
    # handle puncs
    queryLst = extractPunctuations(queryStr, puncsLst)
    # remove stop words again in case some de-hyphenized words contains stop words
    queryLst = [x for x in queryLst if x not in stopWords]
    # keep query words that exists in the inverted index
    queryLst = [x for x in queryLst if x in list(invertedDct.keys())]
    # dictionary to store the tf for every term in query
    queryTF = {}
    # store the occurence of each term in query
    for term in queryLst:
        if term not in queryTF.keys():
            queryTF[term] = 1
        else:
            queryTF[term] += 1

    # Normalize TF: call the 'normTF' function
    # to normalize the tf for each query term in the query
    queryTF = normTF(queryTF)
    # to normalize the tf for overall terms in each doc in the 'docTF' dictionary
    for dct in docTF.values():
        dct = normTF(dct)

    # calculate TF * IDF
    # dictionary of the TF*IDF for each query term
    # term as key, TF*IDF as value
    queryTFIDF = {}
    for item in queryLst:
        queryTFIDF[item] = queryTF[item] * IDF[item]
    # dictionary of the TF*IDF for query terms with documents that they appear,
    # Term as key, a dictionary(which use docID as key and TF*IDF as value)as value
    docTFIDF = {}
    for item in queryLst:
        docTFIDF[item] = {k: v * IDF[item] for k, v in docTF[item].items()}

    # call 'calcCosineSimilarity' function to calculate the Cosine Similarity for every document,
    # and insert into Cosine Similarity dictionary, docID as key
    CosSimiDct = {}
    for i in range(1, docNum+1):
        CosSimiDct[i] = calcCosineSimilarity(i, queryTFIDF, docTFIDF)
    # get a list of docs that have the top 20 Cosine Similarity
    top20 = heapq.nlargest(20, CosSimiDct, key=CosSimiDct.get)
    return top20


'''
create a stop words list
'''


def processStopist(stopistFile):
    stopWords = []
    with open('stoplist.txt', 'r') as file:
        for line in file:
            stopWords.append(line.strip())
    return stopWords


'''
function to convert byte to kb
'''


def convert_bytes(size, unit=None):
    if unit == "KB":
        return round(size / 1024, 3)
    elif unit == "MB":
        return round(size / (1024 * 1024), 3)
    elif unit == "GB":
        return round(size / (1024 * 1024 * 1024), 3)
    else:
        return size


'''
main function
read the xml file uploaded, extract useful infomation from every record
create a inverted index 'dct' that stores the term as key and docID list as value
create a 'docTF' dictionary that stores the term as key and a dictionary(use docID as key and occurences as value) as value
create a 'IDF' dictionary that stores the term as key and IDF as value
'''


def parseFile(*xmlFiles):
    start_time = time.time()
    stopWords = processStopist('stoplist.txt')
    # dictionary storing the inverted index
    dct = {}
    # dictionary storing the IDF for every term in all docs
    # term as key, IDF as value
    IDF = {}
    # dictionary storing occurence of every term in all docs
    # term as key, a dictionary(which use docID as key and occurence as value) as value
    docTF = {}
    docNum = 0
    # loop thru all xmlfiles
    for xmlFile in xmlFiles:
        # parse xml file
        tree = ET.parse(xmlFile)
        # get root element
        root = tree.getroot()
        for child in root:
            # create record number as docID
            docID = int(child[1].text)
            # get the word list for current document
            wordList = createList(child, stopWords)
            # get the total number of terms
            totalNum = len(wordList)
            # insert word, docID pairs into inverted index
            for item in wordList:
                if item not in dct.keys():
                    dct[item] = [docID]
                    # count 1 for an unseen term in doc
                    docTF[item] = {}
                    docTF[item][docID] = 1
                else:
                    dct[item].append(docID)
                    # count 1 for seen term, but unseen docID
                    if docID not in docTF[item].keys():
                        docTF[item][docID] = 1
                    # increase 1 for seen term and seen docID
                    else:
                        docTF[item][docID] += 1

            docNum += 1

    for key in sorted(dct):
        # remove indexed words that are numbers
        if re.match((r"[0-9]+"), key) != None:
            del dct[key]
        # sort docID
        else:
            uniqueDocId = set(dct[key])
            dct[key] = list(sorted(uniqueDocId))
            # inert IDF into dictionary, 1 plus log(total number of docs divided by term appearance times) base e
            IDF[key] = 1 + math.log(float(docNum)/len(dct[key]))
            # print("%-20s%s" % (key, dct[key]))
    # write inverted index into txt file
    index = open("index.txt", "w")
    for key in sorted(dct):
        index.write("%-20s%s\n" % (key, dct[key]))

    stat = os.stat("index.txt")
    size = convert_bytes(stat.st_size, "KB")
    index.write("Size of inverted index file is: %.2f KB.\n" % size)
    index.write("Time taken to build the index: %s seconds.\n" %
                (time.time() - start_time))
    index.close()
    print('Size of inverted index file is: ', size,  'KB.')
    print("Time taken to build the index: %s seconds." %
          (time.time() - start_time))
    return dct, docNum, docTF, IDF


'''
called by parseFile function to extract all useful words into a list
'''


def createList(record, stopWords):
    lst = []
    # get punctuation list
    puncsLst = list(string.punctuation)

    # check if title exists
    if record.find('TITLE') != None:
        titleStr = record.find('TITLE').text.lower()
        # remove all stop words
        titleStr = removeStopWords(titleStr, stopWords)
        # handle puncs
        titleLst = extractPunctuations(titleStr, puncsLst)
        lst.extend(titleLst)

    # check if major subject exists extend words to index list
    if record.find('MAJORSUBJ') != None:
        ele = record.find('MAJORSUBJ')
        for child in ele:
            if child.text != None:
                childLst = extractPunctuations(child.text, puncsLst)
                lst.extend(childLst)

    # check if minor subject exists
    if record.find('MINORSUBJ') != None:
        ele = record.find('MINORSUBJ')
        for child in ele:
            if child.text != None:
                childLst = extractPunctuations(child.text, puncsLst)
                lst.extend(childLst)

    # check if abstract exists
    if record.find('ABSTRACT') != None:
        abstractStr = record.find('ABSTRACT').text.lower()
        # remove all stop words
        abstractStr = removeStopWords(abstractStr, stopWords)
        # handle puncs
        abstractLst = extractPunctuations(abstractStr, puncsLst)
        lst.extend(abstractLst)

    # check if extract exists
    if record.find('EXTRACT') != None:
        extractStr = record.find('EXTRACT').text.lower()
        # remove all stop words
        extractStr = removeStopWords(extractStr, stopWords)
        # handle puncs
        extractLst = extractPunctuations(extractStr, puncsLst)
        lst.extend(extractLst)

    # remove stop words again in case some de-hyphenized words contains stop words
    lst = [x for x in lst if x not in stopWords]

    return lst


'''
called by createList to substract stopWods first and then rejoin into a string
'''


def removeStopWords(Str, stopWords):
    Lst = Str.split()
    # subtract all stop words
    wordLst = [x for x in Lst if x not in stopWords]
    # rejoin string to handle puncs
    emptyStr = " "
    return emptyStr.join(wordLst)


'''
called by createList to handle punctuations and then split string into words
'''


def extractPunctuations(Str, puncs):
    Str = Str.lower()
    for p in puncs:
        Str = Str.replace(p, ' ')
    return Str.split()


# create and get an inverted index dictionary with cf74-79 xml files,
# also get total doc number, get docTF dictionary storing occurence of every term inside one doc)
# and IDF dictionary storing IDF of every term from all docs)
invertedDct, docNum, docTF, IDF = parseFile('cf74.xml', 'cf75.xml', 'cf76.xml',
                                            'cf77.xml', 'cf78.xml', 'cf79.xml')

# get the information needed for ploting graphs
pAt10, MAP, PRList = readQueryXml('cfquery.xml')
print("1. P@10 for %d queries: %s" % (len(pAt10), pAt10))
print("2. Overall Average P@10 for all Queries(MAP): ", MAP)
print("3. Averaged 11-point Precision-Recall list: ", PRList)

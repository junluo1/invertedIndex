from flask import Flask, request, render_template
import xml.etree.ElementTree as ET
import os
import re
import string
import time
import math
import heapq
import collections


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


# create and get an inverted index dictionary with cf74-79 xml files,
# also get total doc number, get docTF dictionary storing occurence of every term inside one doc)
# and IDF dictionary storing IDF of every term from all docs)
invertedDct, docNum, docTF, IDF = parseFile('cf74.xml', 'cf75.xml', 'cf76.xml',
                                            'cf77.xml', 'cf78.xml', 'cf79.xml')


app = Flask(__name__)


@app.route('/')
def my_form():
    return '''
    <!doctype html>
    <title>Query</title>
    <h1>Make a query</h1>
    <form method = "POST" >
        <input name = "text" style="width:500px">
        <input type = "submit" value = "Search">
    </form >
    '''


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    top20Relevant = processQuery(text)
    top20 = map(str, top20Relevant)
    string = " "
    return string.join(top20)


if __name__ == '__main__':
    app.run(debug=True)

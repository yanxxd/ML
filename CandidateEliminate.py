# -*- coding: utf-8 -*-
import os
import pandas as pd
 
def loadData():
    '''数据
Sunny Hot High Week No
Sunny Hot High Strong No
Overcast Hot High Weak Yes
Rain Mild High Weak Yes
Rain Cool Normal Weak Yes
Overcast Cool Normal Strong Yes
Sunny Mild High Weak No
Sunny Cool Normal Weak Yes
Rain Mild Normal Weak Yes
Sunny Mild Normal Strong Yes
Overcast Mild High Strong Yes
Overcast Hot Normal Weak Yes
Rain Mild High Strong No
'''
    set = []
    data = pd.read_csv('data2.txt', sep=r' ', encoding="utf-8", engine='python', header=None)
    for i in xrange(len(data)):
        instance = []
        #print data.iloc[i]
        if 'Sunny' == data.iloc[i][0]:
            instance.append(1);
        elif 'Overcast' == data.iloc[i][0]:
            instance.append(2);
        else: #Rain
            instance.append(4);
                     
        if 'Hot' == data.iloc[i][1]:
            instance.append(1);
        elif 'Mild' == data.iloc[i][1]:
            instance.append(2);
        else: #Cool
            instance.append(4);
             
        if 'High' == data.iloc[i][2]:
            instance.append(1);
        else: #Normal
            instance.append(2);
         
        if 'Strong' == data.iloc[i][3]:
            instance.append(1);
        else: #Weak
            instance.append(2);
         
        if 'Yes' == data.iloc[i][4]:
            instance.append(1);
        else: #No
            instance.append(0);
        set.append(instance)
    return set
 
#判断假设与实例是否一致，代码和isMoreGeneralOrEqual一样
#h : hypothesis 假设
#d : 训练实例
 
def isConsistent(h, d):
    if d[4]: #d为h的正例，h包含d时一致，否则不一致
        for i in xrange(4):
            if not (h[i] == (h[i] | d[i])): #不包含
                return False
        return True
    else: #d为h的反例，h不包含d时一致，否则不一致
        for i in xrange(4):
            if not (h[i] == (h[i] | d[i])): #不包含
                return True
        return False
     
#h1 >= h2 h1是否比h2更一般
#h1 : hypothesis 假设1
#h2 : hypothesis 假设2
def isMoreGeneralOrEqual(h1, h2):
    for i in xrange(4):
        if not (h1[i] == (h1[i] | h2[i])): #不包含
            return False
    return True
 
#生成满足d的极小一般化假设，只能生成1个
def genMinGeneralHypothesis(s, d):
    h = []
    for i in xrange(4):
        if s[i] == d[i]:
            h.append(s[i])
        else:
            h.append(s[i] | d[i])
    return h
 
#生成满足d的极小特殊化假设，最多可以生成n个，n为维度
def genMinSpecialHypothesis(g, d):
    setH = []
    for i in xrange(4):
        h = list(g)
        if not (g[i] == (g[i] | d[i])): #g[i]不包含d[i]
            continue
        else:
            #去掉d[i]都的特征值，并添加到候选假设集
            h[i] = (g[i] & ~d[i]) 
            setH.append(h)
    return setH
 
def printResult(set):
    print '[',
    for i in xrange(len(set)):
        print '[',
        if 7 == set[i][0]:
            print '?',
        elif 0 == set[i][0]:
            print '0',
        else:
            if 4 & set[i][0]:
                print 'Rain',
            if 2 & set[i][0]:
                print 'Overcast',
            if 1 & set[i][0]:
                print 'Sunny',
        print ',',
             
        if 7 == set[i][1]:
            print '?',
        elif 0 == set[i][1]:
            print '0',
        else:
            if 4 & set[i][1]:
                print 'Cool',
            if 2 & set[i][1]:
                print 'Mild',
            if 1 & set[i][1]:
                print 'Hot',
        print ',',
 
        if 3 == set[i][2]:
            print '?',
        elif 2 == set[i][2]:
            print 'Normal',
        elif 1 == set[i][2]:
            print 'High',
        else:
            print '0',
        print ',',
         
        if 3 == set[i][3]:
            print '?',
        elif 2 == set[i][3]:
            print 'Weak',
        elif 1 == set[i][3]:
            print 'Strong',
        else:
            print '0',
        print '], ',
    print ']'
 
def main():
    setS = [[0, 0, 0, 0]] #极大特殊假设
    setG = [[7, 7, 3, 3]] #极大一般假设
    data = loadData()
    for i in xrange(len(data)):
        if data[i][4]: #正例
            #从G中移去所有与d不一致的假设
            j = 0
            while j < len(setG):
                if not isConsistent(setG[j], data[i]):
                    setG.pop(j)
                    j = j - 1
                j = j + 1
            #对S中每个与d不一致的假设s , 从S中移去s
            #并把s的所有的极小一般化式h加入到S中，其中h满足 : h与d一致，而且G的某个成员比h更一般
            j = 0
            while j < len(setS):
                if not isConsistent(setS[j], data[i]):
                    #极小一般化式h  感覺只能添加一個h
                    h = genMinGeneralHypothesis(setS[j], data[i])
                    #从S中移去s
                    setS.pop(j)
                    j = j - 1
                    for k in xrange(len(setG)):
                        if isMoreGeneralOrEqual(setG[k], h):
                            setS.append(h)
                j = j + 1           
 
            #从S中移去所有这样的假设：它比S中另一假设更一般
            j = 0
            while j < len(setS):
                k = j + 1
                while k < len(setS):
                    if isMoreGeneralOrEqual(setS[j], setS[k]):
                        setS.pop(j)
                        j = j - 1
                        break
                    elif isMoreGeneralOrEqual(setS[k], setS[j]):
                        setS.pop(k)
                        continue
                    k = k + 1
                j = j + 1
 
        else: #反例
            #从S中移去所有d不一致的假设
            j = 0
            while j < len(setS):
                if not isConsistent(setS[j], data[i]):
                    setS.remove(j)
                    j = j - 1
                j = j + 1
             
            #对G中每个与d不一致的假设g，从G中移去g            
            #并把g的所有的极小特殊化式h加入到G中，其中h满足:h与d一致，而且S的某个成员比h更特殊
            j = 0
            while j < len(setG):
                if not isConsistent(setG[j], data[i]):
                    #极小特殊化式h
                    setH = genMinSpecialHypothesis(setG[j], data[i])
                    #从G中移去g
                    setG.pop(j)
                    j = j - 1
                    for m in xrange(len(setH)):                        
                        for k in xrange(len(setS)):
                            if isMoreGeneralOrEqual(setH[m], setS[k]):
                                setG.append(setH[m])
                j = j + 1
             
            #从G中移去所有这样的假设：它比G中另一假设更特殊
            j = 0
            while j < len(setG):
                k = j + 1
                while k < len(setG):
                    if isMoreGeneralOrEqual(setG[j], setG[k]):
                        setG.pop(k)
                        continue
                    elif isMoreGeneralOrEqual(setG[k], setG[j]):
                        setG.pop(j)
                        j = j - 1
                        break
                    k = k + 1
                j = j + 1
        if i < len(data) - 1:
            print 'D' + str(i+1)
            print setS
            print setG
            printResult(setS)
            printResult(setG)
        print ''
    return
 
 
if __name__ == "__main__":
    main()

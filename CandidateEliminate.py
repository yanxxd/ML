# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import operator

def loadData(set):
    data = pd.read_csv('data2.txt', sep=r' ', encoding="utf-8", engine='python', header=None)
'''
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
    for i in xrange(len(data)):
        if 'Sunny' == data[i][0]:
            set[i][0] = 0x001;
        esif 'Overcast' == data[i][0]:
            set[i][0] = 0x010;
        else: #Rain
            set[i][0] = 0x100;
                    
        if 'Hot' == data[i][1]:
            set[i][1] = 0x001;
        esif 'Mild' == data[i][1]:
            set[i][1] = 0x010;
        else: #Cool
            set[i][1] = 0x100;
            
        if 'High' == data[i][2]:
            set[i][2] = 0x001;
        else: #Normal
            set[i][2] = 0x010;
        
        if 'Strong' == data[i][3]:
            set[i][3] = 0x001;
        else: #Weak
            set[i][3] = 0x010;
        
        if 'Yes' == data[i][4]:
            set[i][4] = 0x001;
        else: #No
            set[i][4] = 0x000;
    return

#判断假设与实例是否一致
#h : hypothesis 假设
#d : 训练实例
def isConsistent(h, d):
    for i in xrange(4):
        if not (h[i] & instance[i]):
            return False
    return True

#判断假设与实例是否一致 h >= d
#h : hypothesis 假设
#d : 训练实例
def isGreaterOrEqual(h, d):
    for i in xrange(4):
        if not (h[i] & instance[i]):
            return False
    return True

def main():
    loadData()
    setS = [[0, 0, 0, 0]] #极大特殊假设
    setG = [[0x111, 0x111, 0x111, 0x111]] #极大一般假设
    data = []
    loadData(data)
    for i in xrange(len(data)):
        if data[i][4]: #正例
            #从G中移去所有与d不一致的假设
            for j in xrange(len(setG)):
                if not isConsistent(setG[j], data[i]):
                    setG.remove(j)
            #对S中每个与d不一致的假设s , 从S中移去s
            for j in xrange(len(setS)):
                if not isConsistent(setS[j], data[i]):
                    setS.remove(j)
            #把s的所有的极小一般化式h加入到S中，其中h满足 : h与d一致，而且G的某个成员比h更一般
            a 
            
            
            #从S中移去所有这样的假设：它比S中另一假设更一般

        else:
            ;
    return


if __name__ == "main":
    main()
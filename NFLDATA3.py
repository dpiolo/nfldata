#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:51:54 2023

@author: djhoannapiolo
NFL DATA

"""

import regressionEqn as lr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics

#read in the data 
#look at the file names, and I want to plot 
#betting line versus  what actually happened
NFLdata = pd.read_csv("NFL2022.csv")
type(NFLdata)
names = list(NFLdata)
print(names)

n = len(NFLdata)
#set up y which is the delta score = function of x
#x = betting line
#delta = home team score - visit team score

x = NFLdata['line']
y = NFLdata['hScore'] - NFLdata['vScore']

b0,b1 = lr.estimate_coef(x, y)

print(b0,b1)

plt.hist(x, alpha = 0.35, label = 'money line',bins = 20)
plt.hist(y, alpha = 0.45, label = 'actual spread',bins = 20)

plt.xlabel('points')
plt.ylabel('frequency')
plt.grid()

plt.legend()
plt.show()

plt.ylabel('home score - visit score')
plt.xlabel('betting line (points predicted by organization)')


plt.plot(x,y,'.',label = 'data')
xAxis = np.linspace(-20,15)

yPred = b0+b1*xAxis
plt.plot(xAxis,yPred,color ='red',label = 'linear fit')

plt.legend()

plt.hlines(y=0, xmin = -20, xmax = 15, color = 'orange')

plt.show()
#y = total Score x = overUnder line

############################## OVER UNDER LINE VS TOTAL SCORE ########################

total_score = NFLdata['vScore'] + NFLdata['hScore']
over_under_line = NFLdata['overUnder']

b0,b1 = lr.estimate_coef(over_under_line, total_score)


plt.plot(over_under_line,total_score,'.',label = 'Data, (Over-Under line versus Total Score)')


#MAKE FITTED LINE 
fittedline = b0+b1*over_under_line

xAxis = np.linspace(30,60)

plt.plot(xAxis, xAxis , color = 'orange')
plt.plot(over_under_line,fittedline, color = 'red', label = 'linear fit')

plt.ylabel('Total score')
plt.xlabel('Over-Under Line (Predicted by Organization)')

plt.legend()
plt.grid()

plt.show()

############# HISTOGRAMS AND NORMAL DISTRIBUTION CURVE   #################################

histAx = np.linspace(10,90,90)

mean = np.mean(over_under_line)
sd = np.std(over_under_line)
plt.plot(histAx,norm.pdf(histAx,mean,sd),'r-',label = "normal distribution")



plt.hist(over_under_line, alpha = 0.35, label = 'over under line', bins = 20, density = True)
plt.hist(total_score, alpha = 0.45, label = 'total score', bins = 20, density = True)


plt.grid()
plt.legend()
plt.show()

bank = 0
winCount = 0
loseCount = 0

for i in range(n):
    if over_under_line[i] > 52.5:
        if total_score[i] > over_under_line[i]:
            bank += 300
            winCount +=1 
        else: 
            bank -= 330
            loseCount +=1
    if over_under_line[i] < 32.5:
        if total_score[i] < over_under_line[i]:
            bank += 300
            winCount +=1 
        else:
            bank -= 330
            loseCount +=1
        
            
print(f'bank = {bank} wins: {winCount}, losses: {loseCount}')
            


#!/usr/bin/env python

#-*- coding: utf-8 -*-
#  _______ _____    ______    _            _                                            
# (_______|____ \  / _____)  | |          | |                     _                     
#  _____   _   \ \| /        | |      ____| | _   ___   ____ ____| |_  ___   ____ _   _ 
# |  ___) | |   | | |        | |     / _  | || \ / _ \ / ___) _  |  _)/ _ \ / ___) | | |
# | |     | |__/ /| \_____   | |____( ( | | |_) ) |_| | |  ( ( | | |_| |_| | |   | |_| |
# |_|     |_____/  \______)  |_______)_||_|____/ \___/|_|   \_||_|\___)___/|_|    \__  |
#                                                                                (____/ 
# FDC Laboratory
#  _                           _                _____                     
# | |                         (_)              / ____|                    
# | |     ___  __ _ _ __ _ __  _ _ __   __ _  | |    _   _ _ ____   _____ 
# | |    / _ \/ _` | '__| '_ \| | '_ \ / _` | | |   | | | | '__\ \ / / _ \
# | |___|  __/ (_| | |  | | | | | | | | (_| | | |___| |_| | |   \ V /  __/
# |______\___|\__,_|_|  |_| |_|_|_| |_|\__, |_ \_____\__,_|_|    \_/ \___|
# |_   _|                   | | (_)     __/ | | (_)                       
#   | |  _ ____   _____  ___| |_ _  ___|___/| |_ _  ___  _ __             
#   | | | '_ \ \ / / _ \/ __| __| |/ __/ _` | __| |/ _ \| '_ \            
#  _| |_| | | \ V /  __/\__ \ |_| | (_| (_| | |_| | (_) | | | |           
# |_____|_| |_|\_/ \___||___/\__|_|\___\__,_|\__|_|\___/|_| |_|           
#                                                                         
                                                                         
                                                          
# Designed_by_Daseon_#

# See Learning Curve

import sys
                                    
import csv

import matplotlib.pyplot as plt

import numpy as np
                         

# Argument List
#   0. 
#   1. LearningCurve CSV Filename

LearningCurveDataset = []

def read_CSV():
    global LearningCurveDataset
    count_row = 0
    with open(str(sys.argv[1]), newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            #print(row)
            if(count_row >=1):
                LearningCurveDataset.append(row)
                #LearningCurveDataset[count_row] = row
            count_row += 1
    LearningCurveDataset = np.array(LearningCurveDataset, dtype=float)
    print(LearningCurveDataset)

read_CSV()

plt.plot(LearningCurveDataset[:,0], LearningCurveDataset[:,1])
plt.plot(LearningCurveDataset[:,0], LearningCurveDataset[:,2])
plt.title('RWDs-LPF')
plt.xlabel('Episode')
plt.ylabel('reward sum')
plt.show()
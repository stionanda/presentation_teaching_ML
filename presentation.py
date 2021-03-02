#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import statistics as st
import random
import math

import data_load as dl

def sample_deviation(in_arr):
    n = len(in_arr)
    mu = sum(in_arr)/n
    sd = [(x-mu)**2 for x in in_arr]
    return math.sqrt(sum(sd)/(n-1))

data = dl.CommunityViolations().pull_data()

ind_var = data[0]
dep_var = data[1]

def get_class_var(d_row):
    med = st.median(d_row)
    return d_row.apply(lambda x: 0 if x <= med else 1)

dep_bin = pd.DataFrame()
for i in range(len(dep_var.columns)):
     dep_bin = pd.concat([dep_bin, get_class_var(dep_var[dep_var.columns[i]])], axis=1)

ind_bin = pd.DataFrame()
for i in range(len(ind_var.columns)):
    ind_bin = pd.concat([ind_bin, get_class_var(ind_var[ind_var.columns[i]])], axis=1)

y_larc = dep_bin[["larcPerPop"]]

dep_bin = pd.DataFrame()
for i in range(len(dep_var.columns)):
     dep_bin = pd.concat([dep_bin, get_class_var(dep_var[dep_var.columns[i]])], axis=1)

def accuracy_calc(cols, y, test_ratio, which, random_state=60):
    X_train, X_test, y_train, y_test = train_test_split(ind_var[cols], y, test_size=test_ratio, random_state=random_state)
    logreg = LogisticRegression(penalty='none')#, max_iter=500)
    logreg.fit(X_train, y_train)
    if which == 'test':
        return accuracy_score(y_test, logreg.predict(X_test))
    else:
        return accuracy_score(y_train, logreg.predict(X_train))


#first run

rnd_num = 55
test_ratio = 0.5
num_itr = 1

random.seed(rnd_num)
test_acc = []
train_acc = []
test_train_acc_diff = []
x_var = list(ind_var.columns)
test_stdev = []
train_stdev = []
test_train_diff = []
test_train_stdev_diff = []

random_states = [i for i in range(0,100, 20)]
k_vals = [i for i in range(num_itr)]

for i in range(1, len(data[0].columns) + 1):
    test_temp = 0
    train_temp = 0
    
    test_runs = []
    train_runs = []
    test_train_runs_diff = []
    
    for s in random_states:
        test_total = 0
        train_total = 0
        for k in k_vals:
            x_rand = random.sample(x_var, i)
            #X_train, X_test, y_train, y_test = train_test_split(ind_bin[x_rand], y_larc['larcPerPop'], test_size=0.2, random_state=s)
            test_total += accuracy_calc(x_rand, y_larc, test_ratio, 'test')
            train_total += accuracy_calc(x_rand, y_larc, test_ratio, 'train')
        test_runs.append(test_total/num_itr)
        train_runs.append(train_total/num_itr)
        test_train_runs_diff.append((train_total-test_total)/num_itr)
    
    total_iterations = (num_itr*len(random_states))
    test_acc.append(sum(test_runs)/len(test_runs))
    train_acc.append(sum(train_runs)/len(train_runs))
    test_train_acc_diff.append(sum(test_train_runs_diff)/len(test_train_runs_diff))
    
    test_stdev.append(sample_deviation(test_runs))
    train_stdev.append(sample_deviation(train_runs))
    test_train_stdev_diff.append(sample_deviation(test_train_runs_diff))

# one-tailed z=-1.65
boundries = []
for mu, sigma in zip(train_acc, train_stdev):
    boundries.append(mu-(1.65*sigma))

x1 = []
for i in range(len(data[0].columns)):
    x1.append(i)
    
plt.plot(x1, test_acc, label = "Test Accuracy")
plt.plot(x1, train_acc, label = "Train Accuracy")
plt.plot(x1, test_train_acc_diff, label = "Train - Test Accuracy")
plt.plot(x1, boundries, label = "95% Confidence Boundary")
plt.title("Train vs Test Accuracy, {} runs per feature count".format(len(random_states)))
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.legend()
#dont forget to change this
plt.savefig('train_test_acc_{}_states.png'.format(len(random_states)), dpi=200, transparent=True)

plt.clf()
plt.cla()
plt.close()

plt.plot(x1, test_stdev, label = "SampleDev(Test Accuracy)")
plt.plot(x1, train_stdev, label = "SampleDev(Train Accuracy)")
plt.plot(x1, test_train_stdev_diff, label = "SampleDev(Test Acc - Train Acc)")
plt.title("Sample Deviation of Accuracy, {} runs per feature count".format(len(random_states)))
plt.xlabel("Number of features")
plt.ylabel("Standard Deviation")
plt.legend()
#dont forget to change this
plt.savefig('sample_dev_test_train{}_states.png'.format(len(random_states)), dpi=200, transparent=True)

plt.clf()
plt.cla()
plt.close()
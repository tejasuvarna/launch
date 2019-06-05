#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:22:17 2019

@author: saiteja_suvarna
"""

%matplotlib inline
import numpy as np # linear algebra
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sklearn
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D


#STEP 1: Reading data into file
admission_prediction = pd.read_csv("admission_prediction.csv")
admission_prediction.head()
admission_prediction.shape
admission_prediction.columns
admission_prediction = admission_prediction.drop(columns=['Research'])
admission_prediction = admission_prediction.drop(columns=['Serial No.'])
admission_prediction.columns = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP','LOR', 'CGPA', 'Chance_of_Admit']
    #SOP is the rating of the Statement of Purpose
    #LOR is the rating of the letter of recommendation
    #CGPA is the college gpa
    
    
#STEP 2: Visualizations
    #Pairplots
    allscatter = sns.pairplot(admission_prediction)
    
    #3d Scatterplot
    fig = plt.figure(figsize=[15,10])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(admission_prediction['GRE_Score'], admission_prediction['TOEFL_Score'], admission_prediction['Chance_of_Admit'], c='r', marker='o', alpha=1)
    ax.set_xlabel('GRE_Score')
    ax.set_ylabel('TOEFL_Score')
    ax.set_zlabel('Chance_of_Admit')
    plt.show()
    
    fig = plt.figure(figsize=[15,10])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(admission_prediction['SOP'], admission_prediction['LOR'], admission_prediction['Chance_of_Admit'], c='blue', marker='+', alpha=1)
    ax.set_xlabel('Statement of Purpose')
    ax.set_ylabel('Letter of Recommendation')
    ax.set_zlabel('Chance of Admit')
    plt.show()
    

#Conducting multiple linear regression
admission_prediction.GRE_Score = admission_prediction.GRE_Score.astype(float)
admission_prediction.TOEFL_Score = admission_prediction.TOEFL_Score.astype(float)
admission_prediction.University_Rating = admission_prediction.University_Rating.astype(float)

x = admission_prediction.GRE_Score.tolist()
y = admission_prediction.Chance_of_Admit.tolist()    

def linearregression():
  x1 = tf.placeholder(tf.float32, shape=(None, ), name='x')
  y1 = tf.placeholder(tf.float32, shape=(None, ), name='y')

  with tf.variable_scope('lreg') as scope:
    w = tf.Variable(np.random.normal(), name='W')
    b = tf.Variable(np.random.normal(), name='b')
		
    y_pred = tf.add(tf.multiply(w, x1), b)

    loss = tf.reduce_mean(tf.square(y_pred - y1))

  return x1, y1, y_pred, loss



def run():
  losssum = [0]*1700
  x1, y1, y_pred, loss = linearregression()

  optimizer = tf.train.GradientDescentOptimizer(0.000005)
  train_op = optimizer.minimize(loss)

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    feed_dict = {x1: x, y1: y}
		
    for i in range(1700):
      session.run(train_op, feed_dict)
      print(i, "loss:", loss.eval(feed_dict))
      
    print('Predicting')
    y_pred_batch = session.run(y_pred, {x1 : x})
    plt.scatter(x, y)
    plt.plot(x, y_pred_batch, color='red')
    
run()




#plotting line of best fit for results comparison
sns.regplot(x="GRE_Score", y="Chance_of_Admit", data=admission_prediction);

admission_prediction2 = admission_prediction.drop(columns=['CGPA'])
corr = admission_prediction2.corr()
fig = plt.figure(figsize=[15,10])
sns.heatmap(corr)


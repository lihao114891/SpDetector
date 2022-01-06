# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:08:25 2020

@author: LI_HAO
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 22:59:12 2020

@author: LI_HAO
"""

from baseclass.SDetection import SDetection
from sklearn.metrics import classification_report
import numpy as np
from tool import config
from collections import defaultdict
from math import log,exp
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from random import random
#import tensorflow as tf
import tensorflow.compat.v1 as tf 
import random
import os
import math

tf.disable_v2_behavior()
#CoDetector: Collaborative Shilling Detection Bridging Factorization and User Embedding
class SpectrumDetector(SDetection):
    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]'):
        super(SpectrumDetector, self).__init__(conf, trainingSet, testSet, labels, fold)

    def readConfiguration(self):
        super(SpectrumDetector, self).readConfiguration()
        extraSettings = config.LineConfig(self.config['SpectrumDetector'])
        self.k = int(extraSettings['-k'])
        self.negCount = int(extraSettings['-negCount'])  # the number of negative samples
        if self.negCount < 1:
            self.negCount = 1

        self.regR = float(extraSettings['-gamma'])
        self.filter = int(extraSettings['-filter'])

        learningRate = config.LineConfig(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        self.maxLRate = float(learningRate['-max'])
        self.maxIter = int(self.config['num.max.iter'])
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regU, self.regI = float(regular['-u']), float(regular['-i'])

    def printAlgorConfig(self):
        super(SpectrumDetector, self).printAlgorConfig()
        print('k: %d' % self.negCount)
        print('regR: %.5f' % self.regR)
        print('filter: %d' % self.filter)
        print('=' * 80)

    def initModel(self):
        super(SpectrumDetector, self).initModel()
        #self.l_k = 51
        self.l_k=50
        self.i_k = 50
        self.half_k = 25
        self.l_ks = 50
        
        self.G = np.random.rand(len(self.dao.all_User)+1, self.k) / 20  # context embedding
        self.P = np.random.rand(len(self.dao.all_User), self.l_k) / 20  # latent user matrix
        self.P_item = np.random.rand(len(self.dao.all_Item)+1, self.i_k) / 20  # latent user matrix
        self.u_item = np.zeros(len(self.dao.all_User))#user similarity error
        self.u_r = np.zeros([len(self.dao.all_User),3])#rmax , rin, rmax-rmin
        
        self.P_high = np.random.rand(len(self.dao.all_User), self.half_k) / 20
        self.P_low = np.random.rand(len(self.dao.all_User), self.half_k) / 20
        self.Y = np.random.rand(len(self.dao.all_User), 1) / 20  # latent user matrix
        self.Q = np.random.rand(len(self.dao.all_Item)+1, self.l_k) / 20  # latent item matrix
        self.draw_label = np.ones(len(self.dao.all_User))
        
        self.P_user = np.random.rand(len(self.dao.all_User), self.l_k) / 3
        self.Q_item = np.random.rand(len(self.dao.all_Item)+1, self.l_k) / 3
        self.real_rate = np.ones(len(self.dao.all_User))#user fraud rate
        
        self.H = np.zeros((len(self.dao.all_User),len(self.dao.all_Item)+1))
        self.H2 = np.zeros((len(self.dao.all_User),len(self.dao.all_Item)+1))
        self.H_high = np.zeros((len(self.dao.all_User),len(self.dao.all_Item)+1))
        self.H_low = np.zeros((len(self.dao.all_User),len(self.dao.all_Item)+1))
        
        self.H_item = np.zeros((len(self.dao.all_Item)+1,len(self.dao.all_User)))
        self.H2_item = np.zeros((len(self.dao.all_Item)+1,len(self.dao.all_User)))
        #new
        self.u_idx = tf.placeholder(tf.int32, [None], name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, [None], name="v_idx")
        self.U = tf.Variable(tf.truncated_normal(shape=[len(self.dao.all_User), self.l_k], stddev=0.005), name='U')
        self.V = tf.Variable(tf.truncated_normal(shape=[len(self.dao.all_Item)+1, self.l_k], stddev=0.005), name='V')
        self.U_p = np.zeros((len(self.dao.all_User),self.l_k))
        self.V_p = np.zeros((len(self.dao.all_Item)+1, self.l_k))
        
        self.user_embedding = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.item_embedding = tf.nn.embedding_lookup(self.V, self.v_idx)

        
        self.w1 = tf.Variable(tf.random_normal([self.l_k,100],stddev=1,seed=1))
        self.w2 = tf.Variable(tf.random_normal([100,200],stddev=1,seed=1))
        self.w3 = tf.Variable(tf.random_normal([200,1],stddev=1,seed=1))
        self.b1 = tf.Variable(tf.zeros(shape=[100]))
        self.b2 = tf.Variable(tf.zeros(shape=[200]))
        self.b3 = tf.Variable(tf.zeros(shape=[1]))
        #self.w2_ = tf.Variable(tf.random_normal([200,1],stddev=1,seed=1))
        #self.b2_ = tf.Variable(tf.zeros(shape=[1]))
        #new
       # self.x_ = tf.placeholder(tf.float32,shape=(1,2*self.l_k),name='x-input')
        #self.user = tf.placeholder(tf.float32,shape=(None,self.l_ks),name='user')
        #self.item = tf.placeholder(tf.float32,shape=(None,self.l_ks),name='item')
        self.rating = tf.placeholder(tf.float32,shape=(None),name='rating')
        self.lbl = tf.placeholder(tf.int32,shape=(None),name='labels')
        self.real = tf.placeholder(tf.float32,shape=(None),name='real')#user real rate        
        self.x = tf.placeholder(tf.float32,shape=(None,self.l_k),name='x-input')
        self.x_all = tf.placeholder(tf.float32,shape=(None,self.l_k),name='x_all-input')
        self.y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')
    def buildModel(self):
        self.dao.ratings = dict(self.dao.trainingSet_u, **self.dao.testSet_u)
        print(len(self.dao.all_User))
        print(len(self.dao.all_Item))
        for user in self.dao.ratings:
            for item in self.dao.ratings[user]:
                rating = self.dao.ratings[user][item]
                u = self.dao.all_User[user]
                i = self.dao.all_Item[item]
                self.draw_label[u] = self.labels[user]
                self.H[u,i]=1
                self.H2[u,i]=rating
                self.H_item[i,u]=1
                self.H2_item[i,u]=rating
                if rating>= 4:
                    self.H_high[u,i] = 1
                else :
                    self.H_low[u,i]=1
        self.H = np.array(self.H)
        self.H_item = np.array(self.H_item)
        '''
        self.H_high = np.array(self.H_high)
        self.H_low = np.array(self.H_low)
        '''
        print (self.H)
        print (self.H_item)
        n_edge = self.H.shape[1]

        '''
        n_edge_high = self.H_high.shape[1]
        n_edge_low = self.H_low.shape[1]
        '''
            # the weight of the hyperedge
        W = np.ones(n_edge)

        '''
        W_high = np.ones(n_edge_high)
        W_low = np.ones(n_edge_low)
        '''
            # the degree of the node
        DV = np.sum(self.H * W, axis=1)

        '''
        DV_nol = np.zeros((1658,1658))
        print (DV[1])
        for x in range(1658):
            DV_nol[x][x] = DV[x]
        
        
        DV_high = np.sum(self.H_high * W_high, axis=1)
        DV_low = np.sum(self.H_low * W_low, axis=1)
        '''
            # the degree of the hyperedge
        DE = np.sum(self.H, axis=0)

        '''
        
        DE_high = np.sum(self.H_high, axis=0)
        DE_low = np.sum(self.H_low, axis=0)
        '''
        for i in range(len(self.dao.all_Item)+1):
            if DE[i]<1:
                DE[i] = 0.1

            '''
            if DE_high[i]<1:
                DE_high[i] = 0.1
            if DE_low[i]<1:
                DE_low[i] = 0.1
           

        for i in range(len(self.dao.all_User)):
            if DV_high[i]<1:
                DV_high[i] = 0.1
            if DV_low[i]<1:
                DV_low[i] = 0.1
            '''
            
        #for u in range(len(self.dao.all_User)):
            #if DE[u]<1:
              #  DE[u] = 1.0
        invDE = np.mat(np.diag(np.power(DE, -1)))

        '''

        invDE_high = np.mat(np.diag(np.power(DE_high, -1)))
        invDE_low = np.mat(np.diag(np.power(DE_low, -1)))
        '''
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))

        '''

        DV2_high = np.mat(np.diag(np.power(DV_high, -0.5)))
        DV2_low = np.mat(np.diag(np.power(DV_low, -0.5)))
        '''
        W = np.mat(np.diag(W))

        '''

        W_high = np.mat(np.diag(W_high))
        W_low = np.mat(np.diag(W_low))
        '''
        self.H = np.mat(self.H)
        '''

        self.H_high = np.mat(self.H_high)
        self.H_low = np.mat(self.H_low)
        '''
        HT = self.H.T

        '''

        HT_high = self.H_high.T
        HT_low = self.H_low.T
        '''
        #if variable_weight:
           # DV2_H = DV2 * H
           # invDE_HT_DV2 = invDE * HT * DV2
           # return DV2_H, W, invDE_HT_DV2
        #else:
        #G = DV2 * self.H * W * invDE * HT * DV2
        L = DV2 * (DV-self.H2*W*invDE*HT)*DV2
        #L_nol = DV_nol-self.H
        '''
        n_edge_item = self.H_item.shape[1]
        W_item = np.ones(n_edge_item)
        DV_item = np.sum(self.H_item * W_item, axis=1)
        DE_item = np.sum(self.H_item, axis=0)
        for i in range(len(self.dao.all_Item)+1):
            if DV_item[i]<1:
                DV_item[i] = 0.1
        invDE_item = np.mat(np.diag(np.power(DE_item, -1)))
        DV2_item = np.mat(np.diag(np.power(DV_item, -0.5)))
        W_item = np.mat(np.diag(W_item))
        self.H_item = np.mat(self.H_item)
        HT_item = self.H_item.T
        L_item = DV2_item * (DV_item-self.H2_item*W_item*invDE_item*HT_item)*DV2_item
        '''
        
        #L_high = DV2_high * (DV_high-self.H_high*W_high*invDE_high*HT_high)*DV2_high
        #L_low = DV2_low * (DV_low-self.H_low*W_low*invDE_low*HT_low)*DV2_low
           # return G
        #print (L.shape)
        #print (L_item.shape)
        print('begin_linalg')
        #e_vals_item,e_vecs_item = np.linalg.eig(L_item)
        e_vals,e_vecs = np.linalg.eig(L)
        print('down_linalg')
        #e_vals,e_vecs = np.linalg.eig(L_nol)
        
        #e_vals_high,e_vecs_high = np.linalg.eig(L_high)
        #e_vals_low,e_vecs_low = np.linalg.eig(L_low)
        self.all=[]
        self.allid=[]
        self.training = []
        self.test = []
        self.trainingLabels = []
        self.testLabels = []
        self.trainreal = []
        self.trainfake = []
        for ss in range(len(self.dao.all_User)):
            #self.P_high[ss]=e_vecs_high[ss][:,0:self.half_k]
            #self.P_low[ss]=e_vecs_low[ss][:,0:self.half_k]
            self.P[ss][0:self.l_k-1]=e_vecs[ss][:,0:self.l_k-1]
            #self.P_item[ss][0:self.i_k-1]=e_vecs_item[ss][:,0:self.i_k-1]
        #for ss in range(len(self.dao.all_Item)):
            #self.Q[ss]=e_vecs_item[ss][:,0:self.l_k]       
        # self.P = np.hstack((self.P_high,self.P_low))
        
        for user in self.dao.ratings:
            for item in self.dao.ratings[user]:
                u = self.dao.all_User[user]
                i = self.dao.all_Item[item]
                rating = self.dao.ratings[user][item]
                if rating>self.u_r[u][0]:
                    self.u_r[u][0]=rating
                if self.u_r[u][1]==0:
                    self.u_r[u][1]=rating
                if self.u_r[u][1]>rating:
                    self.u_r[u][1]=rating
                self.u_r[u][2]=self.u_r[u][0]-self.u_r[u][1]

        maxrating=defaultdict(list)
        minrating=defaultdict(list)          
        for user in self.dao.ratings:
            for item in self.dao.ratings[user]:
                rating = self.dao.ratings[user][item]
                u = self.dao.all_User[user]
                i = self.dao.all_Item[item]
                if rating==self.u_r[u][0]:
                    maxrating[u].append(i)
                if rating==self.u_r[u][1]:
                    minrating[u].append(i)
        #print(maxrating[0])
        #print(minrating[0])
        
        for user in self.dao.all_User:
            u = self.dao.all_User[user]
            for r1 in maxrating[u]:
                for r2 in minrating[u]:
                    sim = (self.P_item[r1].dot(self.P_item[r2]))
                    #sim2= math.sqrt(0.02*(self.P_item[r1].dot(self.P_item[r2])))
                    if sim>self.u_item[u]:
                        self.u_item[u]=sim
        '''
        realss_x=[]
        realss_y=[]
        fakess_x=[]
        fakess_y=[]
        realss_x_1 = []
        realss_y_1 = []
        fakess_x_1 = []
        fakess_y_1 = []
        realss_x_2 = []
        realss_y_2 = []
        fakess_x_2 = []
        fakess_y_2 = []
        realss_x_3 = []
        realss_y_3 = []
        fakess_x_3 = []
        fakess_y_3 = []
        realss_x_4 = []
        realss_y_4 = []
        fakess_x_4 = []
        fakess_y_4 = []
        realss_x_5 = []
        realss_y_5 = []
        fakess_x_5 = []
        fakess_y_5 = []
        realss_x_6 = []
        realss_y_6 = []
        fakess_x_6 = []
        fakess_y_6 = []
        realss_x_7 = []
        realss_y_7 = []
        fakess_x_7 = []
        fakess_y_7 = []
        realss_x_10 = []
        realss_y_10 = []
        fakess_x_10 = []
        fakess_y_10 = []

        
        for user in self.dao.all_User:
            u = self.dao.all_User[user]
            if int(self.labels[user])==1:
                fakess_x.append(u)
                fakess_y.append(self.u_item[u])
            if int(self.labels[user])==0:
                realss_x.append(u)
                realss_y.append(self.u_item[u])
        for i in range(len(realss_y)):
            if realss_y[i] < 0.01:
                realss_y_1.append(realss_y[i])
            if realss_y[i] >= 0.01 and realss_y[i]<0.02:
                realss_y_2.append(realss_y[i])
            if realss_y[i] >= 0.02 and realss_y[i]<0.03:
                realss_y_3.append(realss_y[i])
            if realss_y[i] >= 0.03 and realss_y[i]<0.04:
                realss_y_4.append(realss_y[i])
            if realss_y[i] >= 0.04 and realss_y[i]<0.05:
                realss_y_5.append(realss_y[i])
            if realss_y[i] >= 0.05 and realss_y[i]<0.06:
                realss_y_6.append(realss_y[i])
            if realss_y[i] >= 0.06 and realss_y[i]<0.07:
                realss_y_7.append(realss_y[i])
            if realss_y[i] >= 0.07:
                realss_y_10.append(realss_y[i])
        yy1 = len(realss_y_1)/len(realss_y)
        yy2 = len(realss_y_2)/len(realss_y)
        yy3 = len(realss_y_3)/len(realss_y)
        yy4 = len(realss_y_4)/len(realss_y)
        yy5 = len(realss_y_5)/len(realss_y)
        yy6 = len(realss_y_6)/len(realss_y)
        yy7 = len(realss_y_7)/len(realss_y)
        yy10 = len(realss_y_10)/len(realss_y)
        for i in range(len(fakess_y)):
            if fakess_y[i] < 0.01:
                fakess_y_1.append(fakess_y[i])
            if fakess_y[i] >= 0.01 and fakess_y[i]<0.02:
                fakess_y_2.append(fakess_y[i])
            if fakess_y[i] >= 0.02 and fakess_y[i]<0.03:
                fakess_y_3.append(fakess_y[i])
            if fakess_y[i] >= 0.03 and fakess_y[i]<0.04:
                fakess_y_4.append(fakess_y[i])
            if fakess_y[i] >= 0.04 and fakess_y[i]<0.05:
                fakess_y_5.append(fakess_y[i])
            if fakess_y[i] >= 0.05 and fakess_y[i]<0.06:
                fakess_y_6.append(fakess_y[i])
            if fakess_y[i] >= 0.06 and fakess_y[i]<0.07:
                fakess_y_7.append(fakess_y[i])
            if fakess_y[i] >= 0.07:
                fakess_y_10.append(fakess_y[i])
        yf1 = len(fakess_y_1)/len(fakess_y)
        yf2 = len(fakess_y_2)/len(fakess_y)
        yf3 = len(fakess_y_3)/len(fakess_y)
        yf4 = len(fakess_y_4)/len(fakess_y)
        yf5 = len(fakess_y_5)/len(fakess_y)
        yf6 = len(fakess_y_6)/len(fakess_y)
        yf7 = len(fakess_y_7)/len(fakess_y)
        yf10 = len(fakess_y_10)/len(fakess_y)
        print('0-0.01:',yy1)
        print('0.01-0.02:',yy2)
        print('0.02-0.03:',yy3)
        print('0.03-0.04:',yy4)
        print('0.04-0.05:',yy5)
        print('0.05-0.06:',yy6)
        print('0.06-0.07:',yy7)
        print('0.07:',yy10)
        print('0-0.01:',yf1)
        print('0.01-0.02:',yf2)
        print('0.02-0.03:',yf3)
        print('0.03-0.04:',yf4)
        print('0.04-0.05:',yf5)
        print('0.05-0.06:',yf6)
        print('0.06-0.07:',yf7)
        print('0.07:',yf10)
        print('fakeISO:',np.mean(fakess_y))
        print('realISO:',np.mean(realss_y))
        plt.scatter(realss_x,realss_y,color='blue')
        plt.scatter(fakess_x,fakess_y,color='red')
        '''
        #for user in self.dao.all_User:
            #u = self.dao.all_User[user]
            #self.P[u][0]=self.u_item[u]    
        
        '''
        tsne = TSNE(n_components = 2)
        X_tsne = tsne.fit_transform(self.P)
        print(X_tsne.shape)
        X,Y = X_tsne[:,0], X_tsne[:,1]

        for x,y,s in zip(X,Y,self.draw_label):
            if(int(s)==0): 
                plt.plot(x,y,'b.')  
                p1 = plt.scatter(x, y, c='blue', marker='.' )
            if(int(s)==1):
                plt.plot(x,y,'r.')
                p2 = plt.scatter(x, y, c='red', marker='.' )
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max()); plt.title('SpDetector')
        plt.legend([p1, p2], ['real user', 'fake user'], loc='lower right', scatterpoints=1)
        plt.savefig('D:\\python_practice\\goods.svg', format="svg")
        plt.show()
        '''
        
        tsne = TSNE(n_components=3)
        self.Y = tsne.fit_transform(self.P)
        
        self.normalUsers = []
        self.spammers = []
        for user in self.labels:
            if self.labels[user] == '0':
                self.normalUsers.append(user)
            else:
                self.spammers.append(user)
        self.normalfeature = np.zeros((len(self.normalUsers), 3))
        self.spamfeature = np.zeros((len(self.spammers), 3))
        normal_index = 0
        for normaluser in self.normalUsers:
            if normaluser in self.dao.all_User:
                self.normalfeature[normal_index] = self.Y[self.dao.all_User[normaluser]]
                normal_index += 1
        spam_index = 0
        for spamuser in self.spammers:
            if spamuser in self.dao.all_User:
                self.spamfeature[spam_index] = self.Y[self.dao.all_User[spamuser]]
                spam_index += 1
        self.randomNormal = np.zeros((500,3))
        self.randomSpam = np.zeros((500,3))
        for i in range(500):
            self.randomNormal[i] = self.normalfeature[random.randint(0,len(self.normalfeature)-1)]
            self.randomSpam[i] = self.spamfeature[random.randint(0,len(self.spamfeature)-1)]
        ax = plt.subplot(projection = '3d')
        ax.scatter(self.spamfeature[:, 0], self.spamfeature[:, 1],self.spamfeature[:, 2], c='red',marker='.',label='Attacker')
        ax.scatter(self.normalfeature[:, 0], self.normalfeature[:, 1], self.normalfeature[:, 2],c='blue',marker='.',label='Genuine User')
        
        plt.legend(loc='upper left')
        #plt.title('SpDetector')
        plt.savefig('D:\\python_practice\\sp2.svg', format="svg")
        
        for user in self.dao.trainingSet_u:
            self.training.append(self.P[self.dao.all_User[user]])
            self.trainingLabels.append(self.labels[user])
            self.all.append(self.P[self.dao.all_User[user]])
            self.allid.append(user)
            if int(self.labels[user])==1:
                self.trainreal.append(self.P[self.dao.all_User[user]])
            if int(self.labels[user])==0:
                self.trainfake.append(self.P[self.dao.all_User[user]])
        for user in self.dao.testSet_u:
            self.test.append(self.P[self.dao.all_User[user]])
            self.testLabels.append(self.labels[user])
            self.all.append(self.P[self.dao.all_User[user]])
            self.allid.append(user)
    
    def get_batch(self,batch_size):
        user_batch=[]
        label_batch = []
        a_list=[]
        while len(a_list)<batch_size:
            d_int = random.randint(0,len(self.training)-1)
            if(d_int not in a_list):
                a_list.append(d_int)
            else :
                pass
        for i in range(batch_size):
            user_batch.append(self.training[a_list[i]])
            self.Y[i]=self.trainingLabels[a_list[i]]
            label_batch.append(self.Y[i])
        return user_batch, label_batch
    
    def get_rating_batch(self,batch_size):
        user_batch=[]
        item_batch=[]
        rating_batch=[]
        a_list = []
        while len(a_list)<batch_size:
            d_int = random.randint(0,len(self.train_g)-1)
            if d_int not in a_list:
                a_list.append(d_int)
            else:
                pass
        for bs in range(batch_size):
            u,i,rating = self.train_g[bs]
            user_batch.append(tf.nn.embedding_lookup(self.U, u))
            item_batch.append(tf.nn.embedding_lookup(self.V, i))
            rating_batch.append(rating)
        return user_batch,item_batch,rating_batch
    
    def predict(self):
        classifier =  DecisionTreeClassifier(criterion='entropy')
        classifier.fit(self.training, self.trainingLabels)
        pred_labels = classifier.predict(self.test)
        print('Decision Tree:')
        print (pred_labels)
        return pred_labels
    def neural(self,x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, self.w2) + self.b2)
        D_logit = tf.matmul(D_h2, self.w3) + self.b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_logit,D_prob
    def generate_train_test_set(self):
        self.dao.rating_train = dict(self.dao.trainingSet_u,**self.dao.testSet_u)
        train=[]
        test=[]
        for user in self.dao.rating_train:
            for item in self.dao.rating_train[user]:
                rating = self.dao.rating_train[user][item]
                u = self.dao.all_User[user]
                i = self.dao.all_Item[item]
                l = int(self.labels[user])
                real_rate = self.real_rate[u]
                if random.random()<0.2:
                    test.append([u,i,float(rating),real_rate,l])
                else:
                    train.append([u,i,float(rating),real_rate,l])
        return train,test
    def predict2(self):
        print ("init")
        self.rating_label=[]
        self.rating_error=[]
        lam = 0.01
        #D_real_logits,D_real_prob = self.neural(self.x)
        #D_fake_logits,D_fake_prob = self.neural(self.x)
        #real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
        #fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
        #real_solver = tf.train.AdamOptimizer(0.0005).minimize(real_loss)
        #fake_solver = tf.train.AdamOptimizer(0.0005).minimize(fake_loss)
        self.train_g, self.test_g = self.generate_train_test_set()
        
        #for fraud detection
        self.user_batch,self.user_label = self.get_batch(batch_size = 310)
        #self.user_em, self.item_em, self.rating_em = self.get_rating_batch(batch_size = 330)
        D_real_logits,y = self.neural(self.x)
        
        self.r_hat = tf.reduce_sum(tf.multiply(self.user_embedding, self.item_embedding), axis=1)
        self.loss_rating = tf.nn.l2_loss(self.real*(self.rating- self.r_hat))
        loss_fraud = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=self.y_))        
        #loss_rating = tf.reduce_mean(tf.reduce_sum((tf.multiply(self.user,self.item)-self.rating)**2))
        
        loss_total = loss_fraud+lam*self.loss_rating
        real_solver = tf.train.AdamOptimizer(0.0002).minimize(loss_total)
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
           

        
        #train_real_set = []
       # for i in range(len(self.trainreal)):
           # self.trainreal[i] = np.mat(self.trainreal[i])
           # train_real_set.append(np.resize(self.trainreal[i],(8,8,1)))
      #  train_fake_set = []
        #for i in range(len(self.trainfake)):
           # self.trainfake[i] = np.mat(self.trainfake[i])
          #  train_fake_set.append(np.resize(self.trainfake[i],(8,8,1)))
        for iters in range(10000):
            #for rating detect
            batch_idx = np.random.randint(len(self.train_g), size=128)
            
            user_idx = [self.train_g[idx][0] for idx in batch_idx]
            item_idx = [self.train_g[idx][1] for idx in batch_idx]
            rating_idx = [self.train_g[idx][2] for idx in batch_idx]
            real_idx = [self.train_g[idx][3] for idx in batch_idx]
            loss_r_,s_= sess.run([loss_total, real_solver], feed_dict={self.x: self.user_batch, self.y_: self.user_label, 
                                                                       self.u_idx: user_idx,self.v_idx: item_idx,self.rating: rating_idx,self.real:real_idx})
            
            fraud_rates=sess.run([y],feed_dict={self.x:self.P})
            #r1,r2 = sess.run([rating_error])
            for it in range(len(self.dao.all_User)):
                self.real_rate[it]=1-fraud_rates[0][it]
               #self.fraud_rate[it] = fraud_rates[0][it]
            #random.shuffle(self.trainreal)
            #random.shuffle(self.trainfake)
            #x_ = self.trainreal[0:800]
            #loss_r_, _ = sess.run([real_loss, real_solver], {self.x: x_})
            #x_x = self.trainfake[0:800]
            #loss_f_, _ = sess.run([fake_loss, fake_solver], {self.x: x_x})
            if iters%100 == 0:
                print('iter:',iters,'loss_r:',loss_r_)
        print("Training finish!... begin testing")      
        pred_labels=[]
        D_test,y_test = self.neural(self.x)
        D_all,y_all=self.neural(self.x_all)
        dt = sess.run([y_test],feed_dict={self.x:self.test})
        dall = sess.run([y_all],feed_dict={self.x_all:self.all})
        
        self.dao.rating_train = dict(self.dao.trainingSet_u,**self.dao.testSet_u)
        self.realrating=[]
        self.fakerating=[]
        self.U_p=sess.run(self.U)
        self.V_p=sess.run(self.V)
        self.rating_errorsss = defaultdict(list)
        for user in self.dao.rating_train:
            for item in self.dao.rating_train[user]:
                ratings = self.dao.rating_train[user][item]
                u = self.dao.all_User[user]
                i = self.dao.all_Item[item]
                rating_hat = self.U_p[u].dot(self.V_p[i])
                self.rating_errorsss[u].append(abs(ratings-rating_hat))
        for allusers in range(len(self.all)):
            user = self.allid[allusers]
            u = self.dao.all_User[user]
            errors = dall[0][allusers]
            #self.rating_errorsss[u]=[i*errors for i in self.rating_errorsss[u]]
            if int(self.labels[user])==0:
                self.realrating.extend(self.rating_errorsss[u])
            if int(self.labels[user])==1:
                self.fakerating.extend(self.rating_errorsss[u])
        print('real',np.mean(self.realrating))
        print('fake',np.mean(self.fakerating))


        for it in range(len(self.test)):
            if dt[0][it]>0.5:
                pred_labels.append(str(1))
            else:
                pred_labels.append(str(0))
        return pred_labels
            
        
        '''
        testt = []
        for user in self.dao.ratings:   
            u = self.dao.all_User[user]        
            testt.append(np.resize(self.P[u],(1,self.l_k)))
        for user in self.dao.ratings:   
            u = self.dao.all_User[user]
            testt[u] = tf.convert_to_tensor(tf.cast(testt[u],tf.float32))
            d_test_logit,d_test= self.neural(testt[u])
            self.frad[u] = d_test.eval()
            print("user",u) 
        
        
                  
        iteration = 0
        while iteration < 50:
            self.loss = 0
            for res in self.train:
                u,i,rating=res
                error = self.frad[u]*(rating - self.P_user[u].dot(self.Q_item[i]))
                #self.loss += self.frad[u]*(error**2)
                self.loss += error**2
                p = self.P_user[u]
                q = self.Q_item[i]
    
                #update latent vectors
                self.P_user[u] += 0.03*error*q
                self.Q_item[i] += 0.03*error*p
            iteration += 1
            print('iteration:',iteration)   
        self.res = []    
        self.res_mae = []
        for res_t in self.test:
            u,i,rating=res_t
            rating_hat = self.P_user[u].dot(self.Q_item[i])
            if rating_hat>5:
                rating_hat =5
            if rating_hat<1:
                rating_hat =1
            a = (rating-rating_hat)**2
            b = abs(rating-rating_hat)
            self.res.append(a)
            self.res_mae.append(b)
        Rmse = math.sqrt(sum(self.res)/len(self.res))
        mae = sum(self.res_mae)/len(self.res_mae)
        print("rmse:",Rmse)
        print("mae:",mae)
     
        print("Training finish!... begin testing")
        pred_labels=[]
        test_set = []
        for it in range(len(self.test)):
            test_set.append(np.resize(self.test[it],(1,self.l_k)))
        print("testing set lenth:",len(test_set))
        ii=0
        for it in range(len(test_set)):
            ii+=1
            #print(ii)
            t_ = test_set[it]
            t_ = tf.convert_to_tensor(tf.cast(t_,tf.float32))
            d_test_logit,d_test = self.neural(t_)
            dt = d_test.eval()
            if dt > 0.5:
                pred_labels.append(1)
            if dt<=0.5:
                pred_labels.append(0)
        print (pred_labels)
        return pred_labels
        '''
        
    

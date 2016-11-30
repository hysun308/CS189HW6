import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import random
import sklearn.metrics as metrics
import csv

def csv_out(label):
    with open('heyi1.csv','w') as file:
        fwriter = csv.writer(file)
        fwriter.writerow(['Id','Category'])
        for i in range(len(label)):
            fwriter.writerow([i+1,int(label[i])])

if __name__ =="__main__":
    data = sio.loadmat("./joke_data/joke_train.mat")
    X_train = data['train']
    validation_data = np.loadtxt('./joke_data/validation.txt',delimiter=',')
    validation_index = validation_data[:,0:2]-1
    validation_index = validation_index.astype(int)
    labels_valid = validation_data[:,2].astype(int)
    test_data = np.loadtxt('./joke_data/query.txt',delimiter=',')
    test_data = test_data[:,1:3]
    #print(X_train.shape)

    X_zero = X_train
    X_zero[np.isnan(X_zero)] = 0
    U, s, Vh = scipy.linalg.svd(X_zero,full_matrices=False)
    S = np.diag(s)

    d = 2;
    R1 = U[:, 0:d].dot(S[0:d,0:d]).dot(Vh[0:d, :])
    MSE1 = np.sum((R1[~np.isnan(X_train)] - X_train[~np.isnan(X_train)]) ** 2)
    print("MSE:",MSE1)
    l1 = []
    for row in validation_index:
        l1.append(R1[row[0], row[1]]>0)
    pred_labels_valid1 = np.array(l1)
    print("Validation accuracy: {0}".format(metrics.accuracy_score(labels_valid, pred_labels_valid1)))

    d2 = 5
    R2 = U[:, 0:d2].dot(S[0:d2,0:d2]).dot(Vh[0:d2, :])
    MSE2 = np.sum((R2[~np.isnan(X_train)] - X_train[~np.isnan(X_train)]) ** 2)
    print("MSE:",MSE2)
    l2 = []
    for row in validation_index:
        l2.append(R2[row[0], row[1]]>0)
    pred_labels_valid2 = np.array(l2)
    print("Validation accuracy: {0}".format(metrics.accuracy_score(labels_valid, pred_labels_valid2)))

    d3 = 10
    R3 = U[:, 0:d3].dot(np.dot(S[0:d3,0:d3],Vh[0:d3, :]))
    MSE3 = np.sum((R3[~np.isnan(X_train)] - X_train[~np.isnan(X_train)]) ** 2)
    print("MSE:",MSE3)
    l3 = []
    for row in validation_index:
        l3.append(R3[row[0], row[1]]>0)
    pred_labels_valid3 = np.array(l3)
    print("Validation accuracy: {0}".format(metrics.accuracy_score(labels_valid, pred_labels_valid3)))

    d4 = 20
    R4 = U[:, 0:d4].dot(np.dot(S[0:d4,0:d4],Vh[0:d4, :]))
    MSE4 = np.sum((R4[~np.isnan(X_train)] - X_train[~np.isnan(X_train)]) ** 2)
    print("MSE:",MSE4)
    l4 = []
    for row in validation_index:
        l4.append(R4[row[0], row[1]] > 0)
    pred_labels_valid4 = np.array(l4)
    print("Validation accuracy: {0}".format(metrics.accuracy_score(labels_valid, pred_labels_valid4)))
    print("Validation on Regularization")


    Reg = 50
    dnew = 10
    Regid = Reg * np.eye(dnew)
    Vnew = Vh[0:dnew,:].T
    Unew = U[:,0:dnew]
    print(Vnew.shape)
    print(Unew.shape)
    for i in range(10000):
        Unew = np.linalg.solve((Vnew.T.dot(Vnew)+Regid), X_zero.dot(Vnew).T)
        Unew = Unew.T
        Vnew = np.linalg.solve((Unew.T.dot(Unew)+Regid), X_zero.T.dot(Unew).T)
        Vnew = Vnew.T
    print(Unew.shape)
    print(Vnew.shape)
    Rnew = Unew.dot(np.dot(S[0:dnew,0:dnew],Vnew.T))
    lnew = []
    for row in validation_index:
        lnew.append(Rnew[row[0], row[1]] > 0)
    pred_labels_validnew = np.array(lnew)
    print("Validation accuracy: {0}".format(metrics.accuracy_score(labels_valid, pred_labels_validnew)))
    print("Validation on Regularization")

    #print(Unew.shape)
    

 #   for i in range(n):
 #       for j in range(d):
  #          if(not np.isnan(X_train[i,j])):
  #             Unew = (Vh.dot(Vh.T)+Regid)

'''
    R1r =
    MSE1r = np.sum((R1r[~np.isnan(X_train)] - X_train[~np.isnan(X_train)]) ** 2)
    print(MSE1r)
    l1r = []
    for row in validation_index:
        l1r.append(R1r[row[0], row[1]] > 0)
    pred_labels_valid1r = np.array(l1r)
    print("Validation accuracy: {0}".format(metrics.accuracy_score(labels_valid, pred_labels_valid1r)))

    R2r =
    MSE2r = np.sum((R2r[~np.isnan(X_train)] - X_train[~np.isnan(X_train)]) ** 2)
    print(MSE2r)
    l2r = []
    for row in validation_index:
        l2r.append(R2r[row[0], row[1]] > 0)
    pred_labels_valid2r = np.array(l2r)
    print("Validation accuracy: {0}".format(metrics.accuracy_score(labels_valid, pred_labels_valid2)))

    R3r =
    MSE3r = np.sum((R3r[~np.isnan(X_train)] - X_train[~np.isnan(X_train)]) ** 2)
    print(MSE3r)
    l3r = []
    for row in validation_index:
        l3r.append(R3r[row[0], row[1]] > 0)
    pred_labels_valid3r = np.array(l3r)
    print("Validation accuracy: {0}".format(metrics.accuracy_score(labels_valid, pred_labels_valid3r)))

    R4r =
    MSE4r = np.sum((R4r[~np.isnan(X_train)] - X_train[~np.isnan(X_train)]) ** 2)
    print(MSE4r)
    l4r = []
    for row in validation_index:
        l4r.append(R4r[row[0], row[1]] > 0)
    pred_labels_valid4r = np.array(l4r)
    print("Validation accuracy: {0}".format(metrics.accuracy_score(labels_valid, pred_labels_valid4r)))

    t = []
    for row in test_data:
        t.append(RT[row[0],row[1]] > 0)
    pred_labels_test = np.array(t)
    csv_out(pred_labels_test)
    '''


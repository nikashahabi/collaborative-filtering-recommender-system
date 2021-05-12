import numpy as np
import pandas as pd
import sklearn.model_selection as sc
from sklearn.metrics.pairwise import pairwise_distances as pairwise_distance
from math import sqrt
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp
from scipy.sparse.linalg import svds



def numberOfUsers(df):
    # returns the number of unique users in dataset
    return df.user_id.unique().shape[0]


def numberOfItems(df):
    # returns the number of unique items in dataset
    return df.item_id.unique().shape[0]


def getTrainTest(df, testSize=0.25):
    # users sklearn.modelselection to shuffle and split data into train and test according to testSize percentage
    trainData, testData = sc.train_test_split(df, test_size=0.25)
    return trainData, testData


def readFromCSV(location):
    # reads from u.data file and returns df
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(location, sep='\t', names=header)
    return df


def createDataMatrix(data, df):
    # returns a matrix which has the rating user i has given to movie j in (i,j)
    dataMatrix = np.zeros((numberOfUsers(df), numberOfItems(df)))
    for line in data.itertuples():
        dataMatrix[line[1] - 1, line[2] - 1] = line[3]
    return dataMatrix


def predict(ratings, similarity, type="user"):
    # returns a prediction matrix which has the predicted rating that user i has given to movie j in (i,j)
    if type == "user":
        meanUserRating = ratings.mean(axis=1)
        ratingsDiff = (ratings - meanUserRating[:, np.newaxis])
        prediction = meanUserRating[:, np.newaxis] + similarity.dot(ratingsDiff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == "item":
        prediction = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return prediction


def rmse(prediction, groundTruth):
    # evaluates the prediction according to groundTruth (test set)
    prediction = prediction[groundTruth.nonzero()].flatten()
    groundTruth = groundTruth[groundTruth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, groundTruth))

def modelBasedCFPrediction(trainDataMatrix):
    # predicts ratings with SVD
    u, s, vt = svds(trainDataMatrix, k=20)
    sDiagMatrix = np.diag(s)
    prediction = np.dot(np.dot(u, sDiagMatrix), vt)
    return prediction



def recommenderSystem(dataset):
    # performs memory and model based collaborative filtering
    df = readFromCSV(dataset)
    trainData, testData = getTrainTest(df)
    trainDataMatrix = createDataMatrix(trainData, df)
    testDataMatrix = createDataMatrix(testData, df)
    # computes user similarity for user-item CF
    userSimilarity = pairwise_distance(trainDataMatrix, metric="cosine")
    # computes item similarity for item-item CF
    itemSimilarity = pairwise_distance(trainDataMatrix.T, metric="cosine")
    # prediction for CF user-item
    userPrediction = predict(trainDataMatrix, itemSimilarity, type="item")
    # prediction for CF item-item prediction
    itemPrediction = predict(trainDataMatrix, userSimilarity, type="user")
    print("user-item memory-based CF prediction = \n ", userPrediction)
    print("item-item memory-based CF prediction = \n ", itemPrediction)
    sparsity = round(1.0 - len(df) / float(numberOfUsers(df) * numberOfItems(df)), 3)
    # performs model-based CF
    modelBasedPrediction = modelBasedCFPrediction(trainDataMatrix)
    print("the sparsity level is " + str(sparsity * 100) + "%")
    print("model-based CF prediction = \n", modelBasedPrediction)
    print('model-based CF evaluation with RMSE = ' + str(rmse(modelBasedPrediction, testDataMatrix)))
    print("user-item memory-based CF evaluation with RMSE = " + str(rmse(userPrediction, testDataMatrix)))
    print("item-item memory-based CF evaluation with RMSE = " + str(rmse(itemPrediction, testDataMatrix)))


recommenderSystem(dataset='ml-100k/u.data')


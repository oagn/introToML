from naiveBayes.utilNB.class_vis import prettyPicture
from naiveBayes.utilNB.map_terrain_data import makeTerrainData
from naiveBayes.utilNB.classify import NBAccuracy

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy
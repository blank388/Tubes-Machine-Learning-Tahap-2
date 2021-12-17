#import library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from NaiveBayesClass import NBClassifier
import matplotlib.pyplot as plt
import seaborn as sb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler as RUS
#import dataset
trainDf = pd.read_csv('kendaraan_train_cleanforclassification.csv')

testDf = pd.read_csv('kendaraan_test_cleanforclassification.csv')

# sb.catplot(x='Tertarik', data=trainDf, kind='count')
# plt.show()

#memisahkan feature & target(label)
trainDfX = trainDf.drop('Tertarik',axis=1)
trainDfY = trainDf['Tertarik']
testDFX = testDf.drop('Tertarik',axis=1)
testDFY = testDf['Tertarik']

# #Eksperimen menggunakan kolom fitur yang independent
# heatmap = sb.heatmap(trainDfX.corr(), linewidths= 0.5, cmap='coolwarm',annot=True)
# plt.title("Korelasi_Kendaraan_Train")
# plt.show()

# trainDfX = trainDfX[["SIM","Kode_Daerah","Premi","Kanal_Penjualan","Lama_Berlangganan"]]
# testDFX = testDFX[["SIM","Kode_Daerah","Premi","Kanal_Penjualan","Lama_Berlangganan"]]

# print(trainDfX.head(5))
# print(testDFX.head(5))

#overSampling 
samplingOV = SMOTE(random_state=42)
trainDfX, trainDfY = samplingOV.fit_resample(trainDfX, trainDfY)

# #Eksperimen underSampling
# samplingUS = RUS(random_state=42)
# trainDfX, trainDfY = samplingUS.fit_resample(trainDfX, trainDfY)

# splitting
trainX , testX, trainY, testY = tts(trainDfX, trainDfY, test_size= 0.2)

#ubah dataframe menjadi np array
trainX = np.array(trainX.copy())    
trainY = np.array(trainY.copy())
testX = np.array(testX.copy())
testY = np.array(testY.copy())
testDFX = np.array(testDFX.copy())
testDFY = np.array(testDFY.copy())

def akurasi(targetAsli, targetPrediksi):
    akurasi = np.sum(targetAsli == targetPrediksi) / len(targetAsli)
    return akurasi

#Evaluasi Model
naivebayes = NBClassifier()
naivebayes.fit(trainX, trainY)
prediksi = naivebayes.prediksi(testX)
print(prediksi)
print("Akurasi Training Model Naive Bayes = ", akurasi(testY, prediksi))

#Validasi Model
prediksiTest = naivebayes.prediksi(testDFX)
print("Akurasi Validasi Model Naive Bayes = ", akurasi(testDFY, prediksiTest))
import numpy as np 

class NBClassifier :
    
    def fit(self, dataFitur, dataTarget):
        jmlDataFitur, jmlFitur = dataFitur.shape
        self.jenisTarget = np.unique(dataTarget)
        jmlJenisTarget = len(self.jenisTarget)

        #insialisasi array mean(tiap fitur dari tiap label)
        self.mean = np.zeros((jmlJenisTarget, jmlFitur), dtype=np.float64)
        #inisialisasi array variansi(tiap fitur dari tiap label)
        self.Var = np.zeros((jmlJenisTarget, jmlFitur), dtype=np.float64)
        #inisialisasi array prob priors(target)
        self.priors = np.zeros(jmlJenisTarget, dtype= np.float64)

        #memulai perhitungan dari array yang sudah dibuat
        for i , value in enumerate(self.jenisTarget):
            #mengambil nilai data fitur yang sesuai dengan tiap target pada jenisTarget
            X_i = dataFitur[value == dataTarget]
            #mean dari axis 0 atau kolom
            self.mean[i, :] = X_i.mean(axis=0)
            #varians dari axis 0 atau kolom
            self.Var[i, :] = X_i.var(axis=0)
            #prob dari tiap target pada jenisTarget
            #membagi jumlah dataFitur tiap target dengan jumlah data fitur keseluruhan
            self.priors[i] = X_i.shape[0] / float(jmlDataFitur)

    def prediksi(self, dataTest):
        #revisi
        target_pred = []
        for i in dataTest:
            target_pred.append(self.predict(i))
        return target_pred

        #sebelumnya
        # target_pred = []
        # for i in dataTest:
        #     target_pred = self.predict(i)
        # return target_pred
        
    def predict(self, sample):
        #P(y|X) = P(X|y) * P(y)
        #P(y|X) atau posterior
        posteriors = []

        for i, value in enumerate(self.jenisTarget):
            #P(y) atau Prior prob dari y 
            prior = np.log(self.priors[i])
            #P(X|y) atau class conditional prob
            classConditional = np.sum(np.log(self.gauss_ProbDensityFunc(i, sample)))
            posterior = prior + classConditional
            posteriors.append(posterior)
        #pilih target dengan prob tertinggi dari semua target
        return self.jenisTarget[np.argmax(posteriors)]
    
    def gauss_ProbDensityFunc(self, idxClass, sample):
        mean = self.mean[idxClass]
        var = self.Var[idxClass]
        numerator = np.exp(-((sample - mean)**2) / (2 *var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
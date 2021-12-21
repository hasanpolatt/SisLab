import matplotlib.pyplot as plt
import numpy as np
import pandas

#data_0123.csv veri setini ana dizinden çekme
df = pandas.read_csv('data_0123.csv', header=None)
print(df)

#output olarak alacağımız değerin 6. satırda olduğunu belirtme
output = df.iloc[0:100, 6].values

#0 değerleri -1, 1 değerleri 1 olarak output dizisinde tutma
output = np.where(output == 1, -1, 1)

#input olarak kullanılacak değerler, 2. ve 4. sütunlara gelecek sütunlara bakarak gelen yeni verileri karşılaştırma
input = df.iloc[0:100, [2, 4]].values

plt.title('2d view', fontsize=16)

#0 sınıfına ait verileri çizdirme
plt.scatter(input[:49, 0], input[:49, 1], color='black', marker='o', label='ones')
#1 sınıfına ait verileri çizdirme
plt.scatter(input[49:98, 0], input[49:98, -1], color='green', marker='x', label='twos')

plt.xlabel('sapel length')
plt.ylabel('petal length')
plt.legend(loc='upper left')

plt.show()

class Perceptron(object):
    def __init__(self, learnRate=0.1, iteration=10):
        self.learnRate = learnRate
        self.iteration = iteration

    def learn(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        self.errors = []
        for _ in range(self.iteration):
            error = 0
            for xi, hedef in zip(X, y):
                degisim = self.learnRate * (hedef - self.goal(xi))
                self.w[1:] += degisim * xi
                self.w[0] += degisim
                error += int(degisim != 0.0)
            self.errors.append(error)
        return self

    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def goal(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

#Hatanın sıfırlanması 40 iterasyon sürdü
classifier = Perceptron(learnRate=0.1, iteration=40)

print(classifier.learn(input, output))

print(classifier.w)

print(classifier.errors)

plt.plot(range(1, len(classifier.errors) + 1), classifier.errors)
plt.xlabel('Test')
plt.ylabel('Number of Wrong Guesses')
plt.show()
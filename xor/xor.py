import matplotlib.pyplot as plt
import numpy as np

input = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
output = np.array([0, 1, 1, 0])

plt.title('XOR', fontsize=16)
plt.scatter(input[:, 0], input[:, 1], s=400, c=output)
plt.grid()
plt.show()

class Perceptron(object):
    def __init__(self, learn=0.1, iteration=10):
        self.learn = learn
        self.iteration = iteration

    def ogren(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        self.errors = []
        for _ in range(self.iteration):
            error = 0
            for xi, target in zip(X, y):
                change = self.learn * (target - self.goal(xi))
                self.w[1:] += change * xi
                self.w[0] += change
                error += int(change != 0.0)
            self.errors.append(error)
        return self

    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def goal(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, 0)

#Hatanın sıfırlanması 10 iterasyon sürdü
classifier = Perceptron(learn=0.1, iteration=10)

print(classifier.ogren(input, output))

print(classifier.w)

print(classifier.errors)

plt.plot(range(1, len(classifier.errors) + 1), classifier.errors)
plt.xlabel('Test')
plt.ylabel('Number of Wrong Guesses')
plt.show()
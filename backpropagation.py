import numpy as np

x = np.random.rand(4,2)

class layer:
    def __init__(self, inputs, outputs):
        self.w = np.random.rand(inputs,outputs)
        self.b = np.zeros((2, outputs)) #Laurens zijn bealo oplossing houdt geen rekening met de toekomst: STOUT, deze 1 moet dezelfde zijn als het aantal kolommen van x

    def forward (self, x):
        self.z = np.dot(self.w.T,x) + self.b.T

    def softmax(self):
        self.a = np.exp(self.z)/np.sum(np.exp(self.z))

def cost(a_L,y):
    assert np.shape(a_L)[0] == np.shape(y)[0]
    return np.sum(np.power(a_L-y,2))/np.shape(a_L)[0]

y = np.array([1,2,3])
xx = np.array([2,1,3])
print(cost(xx,y))

l1 = layer(4,2)
l1.forward(x)
l1.softmax()
print(l1.a)
l2 = layer(2,2)
l2.forward(l1.a)
print(l2.z)

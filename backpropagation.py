import numpy as np

x = np.random.rand(4,1)
learning_rate = 0.01

class layer:
    def __init__(self, inputs, outputs):
        self.w = np.random.rand(inputs,outputs)
        self.b = np.zeros((1, outputs)) #Laurens zijn bealo oplossing houdt geen rekening met de toekomst: STOUT, deze 1 moet dezelfde zijn als het aantal kolommen van x

    def forward (self, x):
        self.z = np.dot(self.w.T,x) + self.b.T

    def softmax(self,z):
        return np.exp(z)/np.sum(np.exp(z))

    def d_softmax(self, z): #i==j, da/dz
        return self.softmax(z)*(1-self.softmax(z))

    #dz/dw = a_L-1 = x

    def backProp(self, x, z, a, y):
        dw = np.dot(x, (self.d_softmax(z) * self.d_cost(a,y)).T)
        self.w -= learning_rate * dw

    def d_cost(self, a_L,y): #dC/da
        return 2*(a_L-y)

def cost(a_L,y):
    assert float(np.shape(a_L)[0]) == float(np.shape(y)[0])
    return np.sum(np.power(a_L-y,2))/float(np.shape(a_L)[0])





y = np.random.rand(2,1)
l1 = layer(4,2)
l1.forward(x)
l1.a = l1.softmax(l1.z)
print(l1.w)
print("cost is", cost(l1.a,y))
print(l1.a)
for i in range(100000):
    l1.backProp(x,l1.z,l1.a,y)
    l1.forward(x)
    l1.a = l1.softmax(l1.z)
print("cost is", cost(l1.a,y))

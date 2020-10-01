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

    def backProp(self, x, z, c_func):#voor de laatste layer is c_func d_cost en daarvoor da
        dw = np.dot(x, (self.d_softmax(z) * c_func).T)
        self.w -= learning_rate * dw

    def calc_da(self, w, z, a, y):
        self.da = np.dot(w, self.d_softmax(z) * d_cost(a,y))

    def backProp2(self, x, z, da): #da is dC/da en komt overeen met da van de volgende layer
        dw = np.dot(x, (self.d_softmax(z) * da).T)
        self.w -= learning_rate * dw

def d_cost( a_L,y): #dC/da
    return 2*(a_L-y)

def cost(a_L,y):
    assert float(np.shape(a_L)[0]) == float(np.shape(y)[0])
    return np.sum(np.power(a_L-y,2))/float(np.shape(a_L)[0])





y = np.random.rand(3,1)
l1 = layer(4,4)
l1.forward(x)
l1.a = l1.softmax(l1.z)
l2 = layer(4,3)
l2.forward(l1.a)
l2.a = l2.softmax(l2.z)

print ("cost is", cost(l2.a,y))
for i in range(100000):
    l2.backProp(l1.a,l2.z,d_cost(l2.a,y))
    l2.calc_da(l2.w, l2.z, l2.a,y)
    l1.backProp(x, l1.z, l2.da)
    l1.forward(x)
    l1.a = l1.softmax(l1.z)
    l2.forward(l1.a)
    l2.a = l2.softmax(l2.z)
print ("cost is",cost(l2.a,y))

# print(l1.w)
# print("cost is", cost(l1.a,y))
# print(l1.a)
# for i in range(100000):
#     l1.backProp(x,l1.z,l1.a,y)
#     l1.forward(x)
#     l1.a = l1.softmax(l1.z)
# print("cost is", cost(l1.a,y))

#Single Layer neural network using numpy only
import numpy as np


def ReLu(x):  
    return np.maximum(x,0)

def Cost(Y_pred,Y_output):  #Categorical crossEntropy
    loss = Y_output*np.log(Y_pred)
    return -np.sum(loss)

def softmax(Y_pred):
    exp_scores = np.exp(Y_pred - np.max(Y_pred, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    

def ReLud(x):
    return (x > 0).astype(int)


class Network:
    def __init__(self):
        self.w1 = np.random.randn(784, 64) * 0.01
        self.w2 = np.random.randn(64, 10) * 0.01
        self.b1 = np.zeros((1,64))
        self.b2 = np.zeros((1,10))

    def l1(self,input_X): 
        x = input_X #(1X784)
        l1out = (x@(self.w1) + self.b1)  #(1X784)X(784X64)+(1X64)
        return l1out.reshape(1,64) #(1,64)
    
    def l1relu(self,input_X):
        return ReLu(self.l1(input_X)).reshape(1,64) #(1,64)
    
    def l2(self,input_X):
        l1out = self.l1relu(input_X)
        l2out = (l1out@(self.w2) + self.b2)#(1,10)
        return l2out.reshape(1,10)
    
    def softmaxl2(self,input_X):
        return softmax(self.l2(input_X))#(1,10)

    
    def train_SGD(self,data_set,lr, epochs,output_Y):
        l = data_set.shape[0] #batch of trainset
        for ee in range(epochs):
            indices = np.arange(l)
            np.random.shuffle(indices)
            data_set_shuffled = data_set[indices]
            output_Y_shuffled = output_Y[indices]
            for i in range(l):
                #calculate grad of cost
                #backpropagation
                
                input_X = data_set_shuffled[i]
                input_X = input_X.reshape(1,784)
                Yout = np.zeros((1,10),dtype=int)
                Yout[0,output_Y_shuffled[i,0]] = 1


                dz2 = self.softmaxl2(input_X)-Yout #(1,10)
                dw2 = self.l1relu(input_X).T @ dz2 #(64,10)
                db2 = np.sum(dz2,axis=0,keepdims=True) #(1,10)
                dA1 = dz2@((self.w2).T) #(1,10) (10,64) #(1,64)
                dZ1 = dA1 * ReLud(self.l1(input_X)) #(1,64) 
                dW1 = input_X.T@dZ1 #`(784,1) (1,64) #(784,64)
                db1 = np.sum(dZ1,axis = 0,keepdims=True)

                self.w1 = self.w1 - lr * (dW1)
                self.w2 = self.w2 - lr* (dw2)
                self.b1 = self.b1 - lr * (db1)
                self.b2 = self.b2 - lr * (db2)
                print("Loss on",i,"th sample in",ee,"epoch is", Cost(self.softmaxl2(input_X),Yout))

        print("Training finished successfully!!")


    def test_model(self,test_data,output_Y):
        l = test_data.shape[0]
        coor = 0
        tot = l
        for i in range(l):
            Yp = self.softmaxl2(test_data[i].reshape(1,784))
            Ypcl = np.argmax(Yp)
            print(Ypcl,output_Y[i,0])
            if output_Y[i,0]==Ypcl:
                coor = coor + 1
            else:
                pass
        print("Accuracy over test data is ", coor*100/tot)


           
#load training data please ensure the csv files according to below path
data = np.loadtxt('fashion-mnist_train.csv',delimiter=',', skiprows=1)
output_Y = data[:,0:1].astype(int)
input_X = data[:,1:]
input_X = input_X/255.0

nn1 = Network()
lr = 0.01
epoch = 40
nn1.train_SGD(input_X,lr,epoch,output_Y)


#load test data
data_t = np.loadtxt('fashion-mnist_test.csv',delimiter=',', skiprows=1)
output_Yt = data_t[:,0:1].astype(int)
input_Xt = data_t[:,1:]
input_Xt = input_Xt/255.0
nn1.test_model(input_Xt,output_Yt)

from function import Conv, ReLU, MaxPool, FC, Softmax
from abc import ABCMeta, abstractmethod

class Net(metaclass=ABCMeta):
    # Neural network super class

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass


class LeNet5(Net):
    # LeNet5

    def __init__(self):
        self.conv1 = Conv(3, 6, 5)
        self.ReLU1 = ReLU()
        self.pool1 = MaxPool(2,2)
        
        self.conv2 = Conv(6, 16, 5)
        self.ReLU2 = ReLU()
        self.pool2 = MaxPool(2,2)
        
        self.conv3 = Conv(16, 16, 5)
        self.ReLU3 = ReLU()
        self.pool3 = MaxPool(2,2)
        
        self.conv4 = Conv(16, 120, 5)
        self.ReLU4 = ReLU()
                                      
        
        self.FC1 = FC(120, 84)
        self.ReLU5 = ReLU()
        self.FC2 = FC(84, 50)
        self.ReLU6 = ReLU()      
        self.Softmax = Softmax()

        self.a4_shape = None
        

    def forward(self, X):
        h1 = self.conv1._forward(X)
        a1 = self.ReLU1._forward(h1)
        p1 = self.pool1._forward(a1)
        
        h2 = self.conv2._forward(p1)
        a2 = self.ReLU2._forward(h2)
        p2 = self.pool2._forward(a2)
        
        h3 = self.conv3._forward(p2)
        a3 = self.ReLU3._forward(h3)
        p3 = self.pool3._forward(a3) # (5, 5)
        
        h4 = self.conv4._forward(p3)
        a4 = self.ReLU4._forward(h4)       
        #print("a4.shape=", a4.shape)                      
        
        self.a4_shape = a4.shape # shape for backward
       
        b, c, h, w = a4.shape
        a4 = a4.reshape(b, c*h*w)         
        
        fc1 = self.FC1._forward(a4)
        fc1 = self.ReLU5._forward(fc1)
        fc2 = self.FC2._forward(fc1)
        fc2 = self.ReLU6._forward(fc2)       
        fc3 = self.Softmax._forward(fc2)                         
        
        return fc3

    def backward(self, dout):       
            
        dout = self.ReLU6._backward(dout)
        dout = self.FC2._backward(dout)
        dout = self.ReLU5._backward(dout)
        dout = self.FC1._backward(dout)
        
        dout = dout.reshape(self.a4_shape) # reshape
        
        dout = self.ReLU4._backward(dout)
        dout = self.conv4._backward(dout)  # (100, 16, 5, 5)
        #print("dout.shape=", dout.shape)
        
        dout = self.pool3._backward(dout) # (100, 16, 9, 9)
        dout = self.ReLU3._backward(dout) 
        dout = self.conv3._backward(dout) 
              
        dout = self.pool2._backward(dout)
        dout = self.ReLU2._backward(dout)
        dout = self.conv2._backward(dout) 
        
        dout = self.pool1._backward(dout)
        dout = self.ReLU1._backward(dout)
        dout = self.conv1._backward(dout)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.conv3.W, self.conv3.b, self.conv4.W, self.conv4.b, 
                      self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.conv3.W, self.conv3.b, self.conv4.W, self.conv4.b, 
                      self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b] = params

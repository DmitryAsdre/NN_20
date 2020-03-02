import numpy as np

class Linear:
    def __init__(self, input_size, output_size):
        '''
        Creates weights and biases for linear layer.
        Dimention of inputs is *input_size*, of output: *output_size*.
        '''
        self.W = np.random.normal(0, 0.1, (input_size, output_size))
        self.b = np.zeros(output_size)
        #### YOUR CODE HERE
        #### Create weights, initialize them with samples from N(0, 0.1).
        pass
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, input_size).
        Returns output of size (N, output_size).
        Hint: You may need to store X for backward pass
        '''
        self.X = X
        self.Y = X @ self.W + self.b
        #### YOUR CODE HERE
        #### Apply layer to input
        return self.Y
    
    def backward(self, dLdy):
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        #### YOUR CODE HERE
        self.dLdW = self.X.T.dot(dLdy)
        self.dLdx = dLdy @ self.W.T
        self.dLdb = np.sum(dLdy, axis=0)
        return self.dLdx
    
    def step(self, learning_rate):
        '''
        1. Apply gradient dLdw to network:
        w <- w - l*dLdw
        '''
        self.W -= learning_rate * self.dLdW
        self.b -= learning_rate * self.dLdb
        #### YOUR CODE HERE
        pass

class Sigmoid:
    def __init__(self):
        pass
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        #### YOUR CODE HERE
        #### Apply layer to input
        
        self.X = X
        self.Y = 1/(1 + np.exp(-X))
        return self.Y
    
    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        #### YOUR CODE HERE
        self.dLdx = dLdy * (1/(np.exp(-self.X/2) + np.exp(self.X/2))**2)
        return self.dLdx
    
    def step(self, learning_rate):
        pass
    
class NLLLoss:
    def __init__(self):
        '''
        Applies Softmax operation to inputs and computes NLL loss
        '''
        #### YOUR CODE HERE
        #### (Hint: No code is expected here, just joking)
        pass
    
    def forward(self, X, y):
        '''
        Passes objects through this layer.
        X is np.array of size (N, C), where C is the number of classes
        y is np.array of size (N), contains correct labels
        '''
        #### YOUR CODE HERE
        #### Apply layer to input
        self.X = X
        self.y = y
        self.NLLSoftMax = np.sum(-X[np.arange(0, len(y)), y] + np.log(np.sum(np.exp(X), axis=1)))
        return self.NLLSoftMax
    
    def backward(self):
        '''
        Note that here dLdy = 1 since L = y
        1. Compute dLdx
        2. Return dLdx
        '''
        #### YOUR CODE HERE
        Exp = np.exp(self.X)
        self.dLdx = ((Exp).T/ np.sum(Exp, axis=1)).T
        self.dLdx[np.arange(0, len(self.y)), self.y] -= 1
        return self.dLdx
    
class NeuralNetwork:
    def __init__(self, modules):
        '''
        Constructs network with *modules* as its layers
        '''
        #### YOUR CODE HERE
        self.modules = modules
    
    def forward(self, X):
        #### YOUR CODE HERE
        #### Apply layers to input
        res = X
        for module in self.modules:
            res = module.forward(res)
        return res
    
    def backward(self, dLdy):
        '''
        dLdy here is a gradient from loss function
        '''
        #### YOUR CODE HERE
        dLdx = dLdy
        for module in reversed(self.modules):
            dLdx = module.backward(dLdx)
        return dLdx
    
    def step(self, learning_rate):
        for module in self.modules:
            module.step(learning_rate)
        pass
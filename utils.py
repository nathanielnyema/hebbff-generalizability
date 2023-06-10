import numpy as np
from scipy.special import expit, erfcinv
import itertools



def generate_dataset(T, R, d=100, p=0.5, noisy = False, noise = 1):
    """
    function to generate a recognition memory dataset
    
    args
    -----
    T: int
        number of trials to simulate
    R: int
        repeat interval
    d: int
        dimensionality of the inputs
    p: float
        probability of seeing a repeat trial
    noisy: bool
        whether or not to introduce noise into repeat
        trials
    noise: int
        hamming distance of the desired noise

    returns
    -------
    x: numpy.ndarry
        Txd array of inputs
    y: numpy.ndarray
        
    
    """
    
    x = np.zeros((T,d))
    y = np.zeros(T)
    
    for i in range(T):
        repeat = False
        if (i>=R) and (np.random.rand()<p):
            # if enough trials generate a random number
            # to determine if this should be a repeat trial
            if not y[i-R]:
                # if the random number satisfies criteria for 
                # this to be a repeat trial chack that the trial
                # R time steps ago wasn't already a repeat
                repeat=True
        if repeat:
            if noisy and noise>0:
                _x = x[i-R].copy()
                flip_idx = np.random.choice(d, noise, replace=False)
                _x[flip_idx] =  -_x[flip_idx]
                x[i] = _x
            else:
                x[i] = x[i-R]
            y[i] = 1
        else:
            # generate a new input 
            x[i] = 2*np.round(np.random.rand(d)) - 1
    return x, y


class IdealHebbFF:
    def __init__(self, d, n, f=0.5, Pfp=0.01, Ptp=0.99):
        """
        Args:
        -----
        d: int
            dimensionality of the input
        n: int
            the number of input dimension to use for addressing
        
        """
        self.d = d
        self.n = n
        self.D = self.d - self.n  # the number of input dimensions whose weights will be plastic
        self.N = 2**self.n # width of hidden layer

        self.w1 = self.D * np.array(list(itertools.product([-1,1],repeat = self.n)))
        a = (erfcinv(2*Pfp) - erfcinv(2*Ptp))*np.sqrt(2*np.e)
        self.gam = 1 - (np.square(a)*f)/(2*self.D*self.N) 
        b = erfcinv(2*Pfp)*np.sqrt(2)/a - n
        self.B = self.D*b 
        self.reset()
        self.frozen = False

    def forward(self, x, ret_h = False):
        h = np.heaviside(self.w1.dot(x[:self.n]) + self.A.dot(x[self.n:]) + self.B, 0)
        yhat = np.all(h==0).astype(float)
        if not self.frozen :
            self.A = self.gam * self.A - np.outer(h, x[self.n:])
        if ret_h:
            return yhat, h
        else:
            return yhat

    def reset(self):
        self.A = np.zeros((self.N, self.D))
        
    def freeze(self):
        self.frozen = True
    
    def unfreeze(self):
        self.frozen = False
        
        
def generate_emnist_dataset(train_data_bin, train_labels, R, d=100, p=0.5, randomize = False):
    """
    function to generate a recognition memory dataset
    
    args
    -----
    train_data_bin: numpy.ndarray
        binarized data array (nsamples x ndim)
        ndim must be >= d
    train_labels: numpy.ndarray
        array of labels for the training data
    R: int
        repeat interval
    d: int
        dimensionality of the inputs
    p: float
        probability of seeing a repeat trial
    randomize: bool
        whether or not to randomize the repeat trials\

    
    returns
    -------
    x: numpy.ndarry
        Txd array of inputs
    y: numpy.ndarray
        
    
    """
    
    labels = np.unique(train_labels)
    trial_labels = []
    x = []
    y = []
    
    i = 0 
    running = True

    while running:
        repeat = False
        if (i>R) and (np.random.rand()<p):
            if not y[-R]:
                repeat = True
        if repeat:
            if randomize:
                label = trial_labels[-R]
                label_data = train_data_bin[train_labels==label, :d]
                x.append(label_data[np.random.choice(label_data.shape[0])])
            else:
                x.append(x[-R])
            y.append(True)
            trial_labels.append(label)
        else:
            label = np.random.choice(labels)
            label_data = train_data_bin[train_labels==label, :d]
            x.append(label_data[np.random.choice(label_data.shape[0])])
            y.append(False)
            trial_labels.append(label)
            labels = labels[labels!=label]
            if labels.size ==0:
                running = False
        i+=1
        
    x = np.array(x).squeeze()
    y = np.array(y)
    
    return x, y
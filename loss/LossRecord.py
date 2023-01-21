class LossRecord():
    def __init__(self):
        self.D = 0.0
        self.G = 0.0
        self.GAdv = 0.0
        self.GCmp = 0.0
        self.GPer = 0.0
        self.counter = 0
    
    def add(self, D, G, GAdv, GCmp, GPer):
        self.D += D
        self.G += G
        self.GAdv += GAdv
        self.GCmp += GCmp
        self.GPer += GPer
        self.counter += 1
    
    def mean(self):
        self.D /= self.counter
        self.G /= self.counter
        self.GAdv /= self.counter
        self.GCmp /= self.counter
        self.GPer /= self.counter
        self.counter = 0
    
    def clear(self):
        self.D = 0.0
        self.G = 0.0
        self.GAdv = 0.0
        self.GCmp = 0.0
        self.GPer = 0.0
        self.counter = 0

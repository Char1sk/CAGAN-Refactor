class LossRecord():
    def __init__(self):
        self.D = 0.0
        self.G = 0.0
        self.GAdv = 0.0
        self.GCmp = 0.0
        self.GPer = 0.0
    
    def add(self, D, G, GAdv, GCmp, GPer):
        self.D += D
        self.G += G
        self.GAdv += GAdv
        self.GCmp += GCmp
        self.GPer += GPer
    
    def clear(self):
        self.D = 0.0
        self.G = 0.0
        self.GAdv = 0.0
        self.GCmp = 0.0
        self.GPer = 0.0

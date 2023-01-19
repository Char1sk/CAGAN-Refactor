class Config:
    def __init__(self, **kargs) -> None:
        for k, v in kargs.items():
            setattr(self, k, v)
            # print(k, v)
    
    def __repr__(self):
        ret = '=== Configs ===\n'
        for k, v in self.__dict__.items():
            ret += f'{k}: {v};\n'
        ret += '\n'
        return ret

TrainConfig = Config(
    # Dataset
    # dataFolder = '../Datasets/CUFS-CAGAN/',
    dataFolder = '../Datasets/CUFS-CAGAN-Change/',
    trainFile = 'files/list_train.txt',
    testFile  = 'files/list_test.txt',
    # Change may be better
    fidPath = [
        '../Datasets/CUFS-CAGAN/AR/photos',
        '../Datasets/CUFS-CAGAN/CUHK/photos',
        '../Datasets/CUFS-CAGAN/XM2VTS/photos',
    ],
    # Data
    imageChannels = 1,
    conptChannels = 8,
    labelChannels = 3,
    # Size
    targetSize = 256,
    # Models
    vggModel = '../Models/vgg.model',
    # Loss
    alpha = 0.7,
    layersVGG = 3,
    # Optim
    lr = 0.0002,
    beta1 = 0.5,
    beta2 = 0.999,
    lamda = 0.01,
    gamma = 10,
    # Train
    epochs = 1,
    batchSize = 1,
    # Result
    printPeriod = 50,
    testPeriod = 1,
    needImage = True,
)

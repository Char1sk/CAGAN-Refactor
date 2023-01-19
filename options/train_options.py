import argparse


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.isInitialized = False
        self.isTrain = True
        self.opt = None
    
    def initialize(self):
        ## Dataset Path
        self.parser.add_argument('--data_folder', default='../Datasets/CUFS-CAGAN-Change/', help='the dataset folder')
        self.parser.add_argument('--train_list', default='files/list_train.txt', help='the list of train samples in dataset')
        self.parser.add_argument('--test_list', default='files/list_test.txt', help='the list of test samples in dataset')
        self.parser.add_argument('--fid_list', default='files/list_fid.txt', help='the list of label folders for FID')
        
        ## Data Description
        self.parser.add_argument('--input_nc', type=int, default=1, help='number of input sketch channels')
        self.parser.add_argument('--conpt_nc', type=int, default=8, help='number of input conponents channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='number of output photo channels')
        
        ## Model Path
        self.parser.add_argument('--vgg_model', default='../Models/vgg.model', help='path of VGG model for Perceptual Loss')
        
        ## Loss Params
        self.parser.add_argument('--alpha', type=float, default=0.7, help='weight of Global L1 Loss in Compositional Loss')
        self.parser.add_argument('--vgg_layers', type=int, default=3, help='layers of VGG to use for Perceptual Loss')

        ## Optim Params
        self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of all optimizers')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of all Adam optimizers')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of all Adam optimizers')
        self.parser.add_argument('--lamda', type=float, default=0.01, help='weight of Compositional Loss')
        self.parser.add_argument('--gamma', type=float, default=10, help='weight of Perceptual Loss')
        
        ## Train Options
        self.parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
        self.parser.add_argument('--batch_size', type=int, default=4, help='dataloader batch size')
        
        ## Test Options
        self.parser.add_argument('--test_start', type=int, default=50, help='which epoch to start test')
        self.parser.add_argument('--test_period', type=int, default=50, help='how many epochs to test once')
        self.parser.add_argument('--save_image_when_test', action='store_true', help='save test fake images or not')
        
        ## Logs and Saves
        self.parser.add_argument('--logs_folder', default='./Logs', help='logs folder for TensorBoard')
        self.parser.add_argument('--log_name', default='exp_debug', help='log name of current run')
        self.parser.add_argument('--model_saves_folder', default='./Saves', help='saves folder for model')
        self.parser.add_argument('--image_saves_folder', default='./Saves', help='saves folder for image')
        
    
    def parse(self):
        if not self.isInitialized:
            self.initialize()
        
        self.opt = self.parser.parse_args()
        
        return self.opt
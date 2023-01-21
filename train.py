import torch
from torch.utils.data.dataloader import DataLoader
import torchvision

import os
import time
import logging

from options.train_options import TrainOptions
from utils.weight_init import weights_init
from utils.fid_score import get_fid
from data.dataset import MyDataset
from models.VggEncoder import MyVggEncoder
from models.GeneratorEncoder import MyGeneratorEncoder
from models.GeneratorDecoder import MyGeneratorDecoder
from models.Discriminator import MyPatchDiscriminator
from loss.AdversarialLoss import AdversarialLoss
from loss.CompositionalLoss import CompositionalLoss
from loss.PerceptualLoss import PerceptualLoss
from loss.LossRecord import LossRecord


def getLogger(logDir='./Logs', logPath=None):
    
    if logPath == None:
        if not os.path.exists(logDir):
            os.mkdir(logDir)
        nowTime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        logPath = f"{logDir}/{nowTime}.log"
    
    logger = logging.getLogger()
    
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    
    fh = logging.FileHandler(logPath)
    sh = logging.StreamHandler()
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    
    return logger


def main():
    # Option
    opt = TrainOptions().parse()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')
    
    # Data
    trainSet = MyDataset(opt, True)
    testSet  = MyDataset(opt, False)
    trainLoader = DataLoader(dataset=trainSet, batch_size=opt.batch_size, shuffle=True)
    testLoader  = DataLoader(dataset=testSet,  batch_size=1, shuffle=False)
    
    # Models
    VGG = MyVggEncoder(opt.vgg_model)
    GenAppE = MyGeneratorEncoder(in_channels = opt.input_nc)
    GenComE = MyGeneratorEncoder(in_channels = opt.conpt_nc)
    GenD = MyGeneratorDecoder(out_channels = opt.output_nc)
    Disc = MyPatchDiscriminator(in_channels = opt.input_nc + opt.conpt_nc + opt.output_nc)
    
    # Cuda
    VGG.to(device)
    GenAppE.to(device)
    GenComE.to(device)
    GenD.to(device)
    Disc.to(device)
    
    # Init
    GenAppE.apply(weights_init)
    GenComE.apply(weights_init)
    GenD.apply(weights_init)
    Disc.apply(weights_init)
    
    # Loss
    criterionAdv = AdversarialLoss()
    criterionCmp = CompositionalLoss(opt.alpha)
    criterionPer = PerceptualLoss(VGG, opt.vgg_layers)
    
    # Optim
    optimGenAppE = torch.optim.Adam(GenAppE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimGenComE = torch.optim.Adam(GenComE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimGenD    = torch.optim.Adam(GenD.parameters(),    lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimDisc    = torch.optim.Adam(Disc.parameters(),    lr=opt.lr, betas=(opt.beta1, opt.beta2))
    
    # Log
    logger = getLogger(logDir=opt.logs_folder)
    
    # Print Params
    logger.info(str(opt))
    
    # Train
    logger.info('=========== Training Begin ===========')
    
    epochRecord = LossRecord()
    for epoch in range(1, opt.epochs+1):
        logger.info(f'Epoch: {epoch:>3d}')
        
        batchRecord = LossRecord()
        for (i, data) in enumerate(trainLoader, 1):
            ### Get Data
            inputs, conpts, labels = [d.to(device) for d in data]
            
            ### Calculation
            AppFeatures = GenAppE(inputs)
            ComFeatures = GenComE(conpts)
            preds = GenD(AppFeatures, ComFeatures)
            
            fakePair = torch.cat((preds,  conpts, inputs), 1)
            realPair = torch.cat((labels, conpts, inputs), 1)
            fakeJudge = Disc(fakePair.detach())
            realJudge = Disc(realPair)
            
            ### Update D
            optimDisc.zero_grad()
            
            lossDFake = criterionAdv(fakeJudge, False)
            lossDReal = criterionAdv(realJudge, True)
            lossD = (lossDFake + lossDReal) * 0.5
            lossD.backward()
            
            optimDisc.step()
            
            ### Update G
            optimGenAppE.zero_grad()
            optimGenComE.zero_grad()
            optimGenD.zero_grad()
            
            fakeJudge = Disc(fakePair)
            # lossGAdv = criterionAdv(fakeJudge, True)
            # lossGCmp = criterionCmp(preds, labels, conpts)
            # lossGPer = criterionPer(preds, labels)
            # lossG = lossGAdv + opt.lamda*lossGCmp + opt.gamma*lossGPer
            lossGAdv = criterionAdv(fakeJudge, True)
            lossGCmp = opt.lamda * criterionCmp(preds, labels, conpts)
            lossGPer = opt.gamma * criterionPer(preds, labels)
            lossG = lossGAdv + lossGCmp + lossGPer
            lossG.backward()
            
            optimGenAppE.step()
            optimGenComE.step()
            optimGenD.step()
            
            ### Record
            batchRecord.add(lossD.item(), lossG.item(), lossGAdv.item(), lossGCmp.item(), lossGPer.item())
            epochRecord.add(lossD.item(), lossG.item(), lossGAdv.item(), lossGCmp.item(), lossGPer.item())
            
            # ### Print Every Batch Period
            # if i % conf.printPeriod == 0:
            #     logger.info(f'Batch: {i:>3d};' +
            #                 f' D loss: {batchRecord.D/conf.printPeriod:>7.5f};'       +
            #                 f' G loss: {batchRecord.G/conf.printPeriod:>7.5f};'       +
            #                 f' GAdv loss: {batchRecord.GAdv/conf.printPeriod:>7.5f};' +
            #                 f' GCmp loss: {batchRecord.GCmp/conf.printPeriod:>7.5f};' +
            #                 f' GPer loss: {batchRecord.GPer/conf.printPeriod:>7.5f};' )
            #     batchRecord.clear()
        
        ### Print Every Epoch
        logger.info(f'Epoch: {epoch:>3d};' +
                    f' D loss: {epochRecord.D/len(trainLoader):>7.5f};'       +
                    f' G loss: {epochRecord.G/len(trainLoader):>7.5f};'       +
                    f' GAdv loss: {epochRecord.GAdv/len(trainLoader):>7.5f};' +
                    f' GCmp loss: {epochRecord.GCmp/len(trainLoader):>7.5f};' +
                    f' GPer loss: {epochRecord.GPer/len(trainLoader):>7.5f};' + '\n')
        epochRecord.clear()
        
        ### Test and Save Every Epoch Period
        if epoch >= opt.test_start and epoch % opt.test_period == 0:
            logger.info('=========== Test ===========')
            with torch.no_grad():
                testRecord = LossRecord()
                for (i, data) in enumerate(testLoader, 1):
                    inputs, conpts, labels = [d.to(device) for d in data]
                    AppFeatures = GenAppE(inputs)
                    ComFeatures = GenComE(conpts)
                    preds = GenD(AppFeatures, ComFeatures)
                    
                    fakePair = torch.cat((preds,  conpts, inputs), 1)
                    realPair = torch.cat((labels, conpts, inputs), 1)
                    fakeJudge = Disc(fakePair)
                    realJudge = Disc(realPair)
                    
                    lossDFake = criterionAdv(fakeJudge, False)
                    lossDReal = criterionAdv(realJudge, True)
                    lossD = (lossDFake + lossDReal) * 0.5
                    # lossGAdv = criterionAdv(fakeJudge, True)
                    # lossGCmp = criterionCmp(preds, labels, conpts)
                    # lossGPer = criterionPer(preds, labels)
                    # lossG = lossGAdv + opt.lamda*lossGCmp + opt.gamma*lossGPer
                    lossGAdv = criterionAdv(fakeJudge, True)
                    lossGCmp = opt.lamda * criterionCmp(preds, labels, conpts)
                    lossGPer = opt.gamma * criterionPer(preds, labels)
                    lossG = lossGAdv + lossGCmp + lossGPer
                    
                    testRecord.add(lossD.item(), lossG.item(), lossGAdv.item(), lossGCmp.item(), lossGPer.item())
                    
                    if opt.save_image_when_test:
                        saveDir = os.path.join(opt.image_saves_folder, str(epoch))
                        if not os.path.exists(saveDir):
                            os.mkdir(saveDir)
                        torchvision.utils.save_image(preds, f'{saveDir}/{str(i)}.jpg', normalize=True, scale_each=True)
                
                if opt.save_image_when_test:
                    fid = get_fid([saveDir, opt.fid_list])
                    logger.info(f'Epoch: {epoch:>3d}; FID: {fid:>9.5f};')
                    
                logger.info(f'Epoch: {epoch:>3d};' +
                    f' D loss: {testRecord.D/len(testLoader):>7.5f};'       +
                    f' G loss: {testRecord.G/len(testLoader):>7.5f};'       +
                    f' GAdv loss: {testRecord.GAdv/len(testLoader):>7.5f};' +
                    f' GCmp loss: {testRecord.GCmp/len(testLoader):>7.5f};' +
                    f' GPer loss: {testRecord.GPer/len(testLoader):>7.5f};' + '\n')
                # Save
                if not os.path.exists(opt.model_saves_folder):
                    os.mkdir(opt.model_saves_folder)
                Disc_out_path = os.path.join(opt.model_saves_folder, f'Disc_epoch_{epoch}.weight')
                torch.save(Disc.state_dict(), Disc_out_path)
                GenD_out_path = os.path.join(opt.model_saves_folder, f'GenD_epoch_{epoch}.weight')
                torch.save(GenD.state_dict(), GenD_out_path)
                GenAppE_out_path = os.path.join(opt.model_saves_folder, f'GenAppE_epoch_{epoch}.weight')
                torch.save(GenAppE.state_dict(), GenAppE_out_path)
                GenComE_out_path = os.path.join(opt.model_saves_folder, f'GenComE_epoch_{epoch}.weight')
                torch.save(GenComE.state_dict(), GenComE_out_path)
    
    logger.info('=========== Training  End  ===========')


if __name__ == '__main__':
    main()

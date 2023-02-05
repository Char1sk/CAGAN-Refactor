import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

import os

from options.train_options import TrainOptions
from utils.weight_init import weights_init
from utils.fid_score import get_fid, get_folders_from_list, get_paths_from_list
from utils.my_logger import get_logger, log_loss, write_loss
from data.dataset import MyDataset
from models.VggEncoder import MyVggEncoder
from models.GeneratorEncoder import MyGeneratorEncoder
from models.GeneratorDecoder import MyGeneratorDecoder
from models.Discriminator import MyPatchDiscriminator
from loss.AdversarialLoss import AdversarialLoss
from loss.CompositionalLoss import CompositionalLoss
from loss.PerceptualLoss import PerceptualLoss
from loss.LossRecord import LossRecord


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
    train_testLoader = DataLoader(dataset=trainSet,  batch_size=1, shuffle=False)
    
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
    criterionAdv = AdversarialLoss(device)
    criterionCmp = CompositionalLoss(opt.alpha, device)
    criterionPer = PerceptualLoss(VGG, opt.vgg_layers)
    
    # Optim
    optimGenAppE = torch.optim.Adam(GenAppE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimGenComE = torch.optim.Adam(GenComE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimGenD    = torch.optim.Adam(GenD.parameters(),    lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimDisc    = torch.optim.Adam(Disc.parameters(),    lr=opt.lr, betas=(opt.beta1, opt.beta2))
    
    # Log
    logDir = os.path.join(opt.logs_folder, opt.log_name)
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    writer = SummaryWriter(logDir)
    logger = get_logger(logDir)
    
    # Print Params
    logger.info(str(opt))
    
    # Train
    logger.info('=========== Training Begin ===========')
    for epoch in range(1, opt.epochs+1):
        logger.info(f'Epoch: {epoch:>3d}')
        epochRecord = LossRecord()
        
        ### Batch Step
        for (i, data) in enumerate(trainLoader, 1):
            batchRecord = LossRecord()
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
            write_loss(writer, batchRecord, 'train_batch', (epoch-1)*len(trainLoader)+(i-1))
        
        ### Every Epoch: Print
        epochRecord.mean()
        log_loss(logger, epochRecord, epoch)
        write_loss(writer, epochRecord, 'train_epoch', (epoch-1))
        
        ### Every Epoch Period: Train FID, Test Loss/FID, and Sav
        if epoch >= opt.test_start and (epoch-opt.test_start) % opt.test_period == 0:
            
            ### Train FID/Image
            logger.info('=========== Train Set ==========')
            with torch.no_grad():
                
                saveDir = os.path.join(opt.image_saves_folder, 'Train', str(epoch))
                if not os.path.exists(saveDir):
                    os.makedirs(saveDir)
                
                for (i, data) in enumerate(train_testLoader, 1):
                    inputs, conpts, labels = [d.to(device) for d in data]
                    AppFeatures = GenAppE(inputs)
                    ComFeatures = GenComE(conpts)
                    preds = GenD(AppFeatures, ComFeatures)
                    
                    if i in opt.train_show_list:
                        writer.add_image(f'gen_photos_train/{i}', preds.squeeze(0)/255, epoch)
                    torchvision.utils.save_image(preds, f'{saveDir}/{i}.jpg', normalize=True, scale_each=True)
                
                fid = get_fid([saveDir, tuple(get_paths_from_list(opt.data_folder, opt.train_list))], path=opt.inception_model)
                logger.info(f'Epoch: {epoch:>3d}; FID: {fid:>9.5f};')
                writer.add_scalar('FID/train', fid, epoch)
            
            ### Test Loss/FID/Image
            logger.info('=========== Test Set ===========')
            with torch.no_grad():
                
                saveDir = os.path.join(opt.image_saves_folder, 'Test', str(epoch))
                if not os.path.exists(saveDir):
                    os.makedirs(saveDir)
                
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
                    
                    lossGAdv = criterionAdv(fakeJudge, True)
                    lossGCmp = opt.lamda * criterionCmp(preds, labels, conpts)
                    lossGPer = opt.gamma * criterionPer(preds, labels)
                    lossG = lossGAdv + lossGCmp + lossGPer
                    
                    testRecord.add(lossD.item(), lossG.item(), lossGAdv.item(), lossGCmp.item(), lossGPer.item())
                    
                    if i in opt.test_show_list:
                        writer.add_image(f'gen_photos_test/{i}', preds.squeeze(0)/255, epoch)
                    torchvision.utils.save_image(preds, f'{saveDir}/{i}.jpg', normalize=True, scale_each=True)
                
                fid = get_fid([saveDir, tuple(get_paths_from_list(opt.data_folder, opt.test_list))], path=opt.inception_model)
                logger.info(f'Epoch: {epoch:>3d}; FID: {fid:>9.5f};')
                writer.add_scalar('FID/test', fid, epoch)
                
                testRecord.mean()
                log_loss(logger, testRecord, epoch)
                write_loss(writer, testRecord, 'test', epoch)
                
            ### Save Models
            if opt.save_models:
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

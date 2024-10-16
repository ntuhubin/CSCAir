import os.path
import subprocess
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset

from utils.dataset_utils import TrainDataset

from option import options as opt
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.loss_utils import CELoss,NNCELoss,CELoss4
from transformer import Restormer

def test(net, loader_test, max_psnr, max_ssim, step,device):
    net.eval()
    torch.cuda.empty_cache()
    psnr = AverageMeter()
    ssim = AverageMeter()

    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(device)
        targets = targets.to(device)
        pred = net(inputs)
        temp_psnr, temp_ssim, N = compute_psnr_ssim(pred, targets)
        psnr.update(temp_psnr, N)
        ssim.update(temp_ssim, N)
    return psnr.avg, ssim.avg

def test_Denoise(net, dataset,device, sigma=15):
    output_path = opt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)

            restored,_ = net(degrad_patch)


            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            #save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Deonise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


def test_Derain_Dehaze(net, dataset,device, task="derain"):
    output_path = opt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)

            restored,_ = net(degrad_patch)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            #save_image_tensor(restored, output_path + degraded_name[0] + '.png')

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        return psnr.avg, ssim.avg

def test_Derain_DehazePP(net, dataset,device, task="derain"):
    output_path = opt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)

            restored,_ = net(clean_patch)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            #save_image_tensor(restored, output_path + degraded_name[0] + '.png')

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))

def test_Derain_DehazeNN(net, dataset,device, task="derain"):
    output_path = opt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)

            restored,_ = net(degrad_patch)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, degrad_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            #save_image_tensor(restored, output_path + degraded_name[0] + '.png')

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))

def ExtractFeature(pp_net, nn_net, clear, rainy):
    _, ppout, _ = pp_net.E(clear, clear)
    _, nnout, _ = nn_net.E(rainy, rainy)
    return ppout, nnout


if __name__ == '__main__':
    torch.cuda.set_device(opt.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    subprocess.check_output(['mkdir', '-p', opt.ckpt_path])

    trainset = TrainDataset(opt)
    trainloader = DataLoader(trainset, batch_size=4, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    opt.ckpt_path = "ckpt/"

    # Network Construction
    denoise_set = DenoiseTestDataset(opt)
    derain_set = DerainDehazeDataset(opt)
    # Start training
    print('Start training...')
    modelname = opt.ckpt_path+"model_epoch_50.pt"


    max_ssim = 0
    max_psnr = 0
    best_epoch = 0
    startepoch= 0
    net = Restormer()
    net = nn.DataParallel(net)
    net = net.to(device)




    posnet = Restormer()
    posnet = nn.DataParallel(posnet)
    posnet = posnet.to(device)

    negnet = Restormer()
    negnet = nn.DataParallel(negnet)
    negnet = negnet.to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    ppoptimizer  = optim.Adam(posnet.parameters(), lr=opt.lr)
    nnoptimizer = optim.Adam(negnet.parameters(), lr=opt.lr)
    l1 = nn.L1Loss().to(device)
    ppl1 = nn.L1Loss().to(device)
    nnl1 = nn.L1Loss().to(device)
    CE = CELoss4().to(device)
    PPCE = CELoss().to(device)
    NNCE = NNCELoss().to(device)
    startepoch=0
    for epoch in range(startepoch,opt.epochs):
        #posnet.eval()
        #negnet.eval()
        net.train()
        posnet.train()
        negnet.train()

        for ([clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2) in tqdm(trainloader):
            degrad_patch_1, degrad_patch_2 = degrad_patch_1.to(device), degrad_patch_2.to(device)
            clean_patch_1, clean_patch_2 = clean_patch_1.to(device), clean_patch_2.to(device)
            #rain, clear = rain.to(device), clear.to(device)

            optimizer.zero_grad()
            ppoptimizer.zero_grad()
            nnoptimizer.zero_grad()

            restored, out= net(degrad_patch_1)
            pprestored,ppout = posnet(clean_patch_1)
            nnrestored,nnout = negnet(degrad_patch_1)


            celoss = CE(out, ppout.detach(), nnout.detach())

            l1_loss = l1(restored, clean_patch_1)

            loss = l1_loss+0.01*celoss

            # backward


            pp_loss = ppl1(pprestored, clean_patch_1)
            ppceloss = PPCE(ppout, out.detach(), nnout.detach())
            pp_loss = pp_loss + 30*ppceloss

            nn_loss = nnl1(nnrestored, degrad_patch_1)
            nnceloss = NNCE(nnout, out.detach(), ppout.detach())
            nn_loss = nn_loss + 30*nnceloss

            loss.backward()
            optimizer.step()

            pp_loss.backward()
            ppoptimizer.step()


            nn_loss.backward()
            nnoptimizer.step()


        print(
            'Epoch (%d)  Loss: l1_loss:%0.4f    Loss: l1_loss:%0.4f \n' % (
                epoch, loss.item(), l1_loss.item()
            ), '\r', end='')

        print(
            'Epoch (%d)  PPLoss: %0.4f    nnloss: %0.4f \n' % (
                epoch, pp_loss.item(), nn_loss.item()
            ), '\r', end='')

  
        if (epoch + 1) % 20 == 0 and epoch>300:
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }





            print('Start testing Sigma=15...')
            test_Denoise(net, denoise_set, device, sigma=15)

            print('Start testing Sigma=25...')
            test_Denoise(net, denoise_set, device, sigma=25)

            print('Start testing Sigma=50...')
            test_Denoise(net, denoise_set, device, sigma=50)

            print('Start testing rain streak removal...')
            ssim1,psnr1=test_Derain_Dehaze(net, derain_set, device, task="derain")

            print('Start testing SOTS...')
            ssim2,psnr2=test_Derain_Dehaze(net, derain_set, device, task="dehaze")

            ssim_eval=(ssim1+ssim2)/2
            psnr_eval=(psnr1+psnr2)/2
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = ssim_eval
                max_psnr = max_psnr
                best_epoch = epoch
                # modelname = opt.ckpt_path + 'all_epoch_' + str(epoch + 1) + '.pk'
                torch.save({
                    'model': net.state_dict()
                }, opt.ckpt_path + 'maxcc' + '.pth')


        lr = 0.0001 * (0.5 ** ((epoch - opt.epochs_encoder) // 125))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in ppoptimizer.param_groups:
            param_group['lr'] = lr
        for param_group in nnoptimizer.param_groups:
            param_group['lr'] = lr
    #print(f'\nbest- step :{best_epoch} |ssim:{max_ssim:.4f}| psnr:{max_psnr:.4f}')

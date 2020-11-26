from torch.autograd import Variable
import torch.autograd as autograd
import torch
import torch.optim as optim
from helper import write_log, write_figures, savefig
# from hdr_loss import HDRLoss
import numpy as np
from dataset import get_loader
import torch.nn as nn
from model import HDRWaveEnDe_simple_decompose, define_D, Model
from tqdm import tqdm
from torchvision import transforms as T
# from msssim import MSSSIM
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# from attention_unpool import Model

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha)
                                            * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1, 62, 62).fill_(
        1.0), requires_grad=False)
#     torch.Size([8, 6, 224, 224])
# torch.Size([8, 1, 26, 26])
    # print(real_samples.shape)
    # print(real_samples.size(0))
    # print(alpha.shape)
    # print(interpolates.shape)
    # print(d_interpolates.shape)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

#-----多GPU训练的模型读取的代码，multi-gpu training---------
def load_network(network):
    save_path = 'output/weight_model2.pth'
    state_dict = torch.load(save_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:] # remove `module.`
        new_state_dict[namekey] = v
    # load params
    network.load_state_dict(new_state_dict)
    return network

def fit(epoch, model, wave_model, net_d, optimizer, optimizer_d, criterion, criterionMSE, criterionGAN, msssim, tvloss, device, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()
    show = 0
    running_loss = 0
    total_loss_d = 0
# index, (low, high, out, groundtruth, filename) in
# enumerate(train_dataloader):
    for low, high, out, groundtruth, filename in tqdm(data_loader):
        inputs = out[:,0:1,:,:].to(device)
        targets = groundtruth[:,0:1,:,:].to(device)
        low = low[:,0:1,:,:].to(device)
        high = high[:,0:1,:,:].to(device)
        if phase == 'predict':
            model.eval()
            # load_network(model)
            model.load_state_dict(torch.load(
                'output/weight.pth', map_location=device))

        if phase == 'training':
            optimizer.zero_grad()
            skips = wave_model(low, high)
            outputs, d2, d3, d4, d5, d6, db = model(inputs, skips)




            # discriminator
            # # # forward
            real_a = inputs.to(device)
            real_b = groundtruth[:,0:1,:,:].to(device)
            fake_b = outputs.to(device)
            fake_b.data.clamp_(-1, 1)

            ######################
            # (1) Update D network
            ######################

            optimizer_d.zero_grad()

            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab.detach())
            loss_d_fake = criterionGAN(pred_fake, False)

            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = net_d.forward(real_ab)
            loss_d_real = criterionGAN(pred_real, True)

# Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                net_d, real_ab.data, fake_ab.data)
            # Adversarial loss
            # d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
            # Combined D loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5 + 10 * gradient_penalty
            total_loss_d += loss_d.item()

            loss_d.backward()  # retain_graph=True

            optimizer_d.step()
        else:
            with torch.no_grad():
                skips = wave_model(low, high)
                outputs, d2, d3, d4, d5, d6, db = model(inputs, skips)
        # with torch.autograd.set_detect_anomaly(True):
            # loss
        # lossMSSIM = msssim(outputs, targets)
                    #tv loss
        lossTV = tvloss(outputs)
        loss1 = criterion(outputs, targets)
        loss2 = criterion(d2, targets)
        loss3 = criterion(d3, targets)
        loss4 = criterion(d4, targets)
        loss5 = criterion(d5, targets)
        loss6 = criterion(d6, targets)
        loss7 = criterion(db, targets)
        lossM = criterionMSE(outputs, targets)
        lossM2 = criterionMSE(d2, targets)
        lossM3 = criterionMSE(d3, targets)
        lossM4 = criterionMSE(d4, targets)
        lossM5 = criterionMSE(d5, targets)
        lossM6 = criterionMSE(d6, targets)
        lossM7 = criterionMSE(db, targets)
        # X: (N,3,H,W) a batch of normalized images (-1 ~ 1)
        # Y: (N,3,H,W)
        # outputs_msssim = (outputs + 1) / 2  # [-1, 1] => [0, 1]
        # targets_msssim = (targets + 1) / 2
        # ms_ssim_loss = ms_ssim( outputs_msssim, targets_msssim, data_range=1, size_average=True)

        lossM = lossM + lossM2 * 0.8 + lossM3 * 0.7 + lossM4 * \
            0.6 + lossM5 * 0.5 + lossM6 * 0.4 + lossM7 * 0.3
        lossL1 =  loss1 * 0.8 + loss2 * 0.7 + loss3 * \
            0.6 + loss4 * 0.5 + loss5 * 0.4 + loss6 * 0.3 + loss7 * 0.3
        loss = lossL1 + lossM + lossTV # + ms_ssim_loss

        if phase == 'training':
            print('lossM: %.4f   lossL1: %.4f    lossd: %.4f    penalty: %.4f  lossTV: %.4f  ' %(lossM.item(), lossL1.item(), (loss_d_fake + loss_d_real).item(), gradient_penalty.item(), lossTV.item()))
        else:
            print("lossM: %.4f   lossL1: %.4f   lossTV: %.4f   "%(lossM.item(), lossL1.item(), lossTV.item()))
        #+ lossMSSIM
        running_loss += loss1.item() + loss2.item() + loss3.item() + loss4.item() + \
            loss5.item() + loss6.item() + loss7.item() + lossM.item() #+ lossM2.item() + lossM3.item() + lossM4.item() + lossM5.item() + lossM6.item() + lossM7.item()
        # savefig(epoch, out[:,0:1,:,:].cpu(), out[:,1:3,:,:].cpu(), filename)
        if phase == 'training':
            show = 0
            loss.backward()
            optimizer.step()
        elif phase == 'validation' and show <= 20:
            # savefig(epoch, outputs.cpu(), out[:,1:3,:,:].cpu(), filename)
            show += 1
        elif phase == 'predict':
            savefig(epoch, outputs.cpu(), out[:,1:3,:,:].cpu(), filename)
    epoch_loss = running_loss / len(data_loader.dataset)
    print('[%d][%s] loss: %.4f' % (epoch, phase, epoch_loss))
    return epoch_loss


def train(root, device, model, wave_model, net_d, epochs, bs, lr):
    # print('start training ...........')
    train_loader, val_loader = get_loader(
        root=root, batch_size=bs, shuffle=True)

    criterion = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)
    criterionGAN = GANLoss().to(device)
    msssim = MS_SSIM()
    tvloss = TVLoss()
    train_losses, val_losses, l1_losses, mse_losses, msssim_losses, d_losses = [
    ], [], [], [], [], []
    lr_d = 0.002
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr_d, betas=(0.5, 0.999))
    for epoch in range(epochs):
        if epoch % 40 == 0:
            lr /= 10
            lr_d /= 10
            # print(' change lr and lr_d to ', lr, lr_d)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.5)
        # train_epoch_loss = fit(epoch, model, wave_model, net_d, optimizer, optimizer_d,
        #                        criterion, criterionMSE, criterionGAN, msssim, tvloss, device, train_loader, phase='training')
        # val_epoch_loss = fit(epoch, model, wave_model, net_d, optimizer, optimizer_d, criterion,
        #                      criterionMSE, criterionGAN, msssim,tvloss, device, val_loader, phase='validation')
        val_epoch_loss = fit(epoch, model, wave_model, net_d, optimizer, optimizer_d, criterion,
                             criterionMSE, criterionGAN, msssim,tvloss, device, val_loader, phase='predict')
        print('-----------------------------------------')

        if epoch == 0 or val_epoch_loss <= np.min(val_losses):
            torch.save(model.state_dict(), 'output/weight.pth')

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        write_figures('output', train_losses, val_losses)
        write_log('output', epoch, train_epoch_loss, val_epoch_loss)


if __name__ == "__main__":
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = Model(3, 3).to(device)
    wave_model = HDRWaveEnDe_simple_decompose().to(device)
    net_d = define_D(2, 64, 'basic', gpu_id=device).to(device)
    batch_size = 1
    num_epochs = 200
    learning_rate = 0.01
    root = 'data/train'
    train(root, device, model, wave_model, net_d,
          num_epochs, batch_size, learning_rate)

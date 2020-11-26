import matplotlib.pyplot as plt

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.autograd as autograd
import torch
import numpy as np
import os
from torchvision import  transforms as T
from skimage.color import ycbcr2rgb
from skimage.io import imsave
from skimage import img_as_ubyte
def write_figures(location, train_losses, val_losses):
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.savefig(location + '/loss.png')
    plt.close('all')


def write_log(location, epoch, train_loss, val_loss):
    if epoch == 0:
        f = open(location + '/log.txt', 'w+')
        f.write('epoch\t\ttrain_loss\t\tval_loss\n')
    else:
        f = open(location + '/log.txt', 'a+')

    f.write(str(epoch) + '\t' + str(train_loss) + '\t' + str(val_loss) + '\n')

    f.close()

def savefig(epoch, outputs, cbcr, filename):
    test_mean, test_std = torch.tensor([0.5 ,0.5 ,0.5]), torch.tensor([0.5 ,0.5, 0.5])
    if epoch %1 == 0:
        # np.clip(outputs,-1,1)
        # outputs.data.clamp_(-1,1)


        outputs = torch.cat([outputs,cbcr],dim=1)

        prediction = outputs * test_std.view(3,1,1) + test_mean.view(3,1,1)
        prediction = prediction *255
        # print(prediction.shape)
        # prediction.data.clamp_(0,1)


        #ycbcr to rgb
        prediction = ycbcr2rgb(prediction[0].numpy().transpose(1,2,0))
        prediction = np.clip(prediction, -1, 1)
        # print(prediction)
        imsave(".\\Result_model2\\%06s_fused.png" % (filename[0].split('\\')[-2]), img_as_ubyte(prediction))
        # image_list = T.ToPILImage()(prediction).convert('RGB')
        # # image_list = T.ToPILImage()(prediction[0].cpu()).convert('RGB')
        # filepath = '.\\Result'
        # if not os.path.exists(filepath):
        #     os.mkdir(filepath,7777)
        # print('=============>save image to  ',filepath + r'\{}.png'.format(filename[0].replace('.png','').split('\\')[-1].split('\\')[0]))
        # image_list.save(filepath + '\\{}.png'.format(filename[0].split('\\')[-2]),'png')

def predict(model, low, high):
    model.eval()
    model.load_state_dict(torch.load('output/weight.pth', map_location=device))
    skips = wave_model(low, high)
    outputs, d2, d3, d4, d5, d6, db = model(inputs, skips)

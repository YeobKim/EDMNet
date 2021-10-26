import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.net import *
from utils.utils import *
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from imageio import imwrite
import PIL
from PIL import Image
from PIL import ImageFilter
from datetime import datetime
from utils.dataset import *
from utils.ssim import *
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# import torchsummary.summary as summary
from warmup_scheduler import GradualWarmupScheduler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="EDMNet:Deblurring Network Using Edge Module, Deformable Convolution and Multi-Stage Network.")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=2, help="Training batch size")
parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
parser.add_argument("--resume_epochs", type=int, default=0, help="Number of training epochs When training resume")
parser.add_argument("--decaystart_epochs", type=int, default=600, help="Number of training epochs When training resume")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--step", type=int, default=200, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--tensorboardx", type=str, default="logs/gopro3", help='path of log files')
parser.add_argument("--patchsize", type=int, default=256, help='patch size of image')
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')

opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True, batchSize=opt.batchSize, patchSize=opt.patchsize).data
    dataset_val = Dataset(train=False).data
    print("# of training samples: %d\n" % int(len(dataset_train)))

    toTensor = transforms.Compose([transforms.ToTensor()])
    toPILImg = transforms.ToPILImage()

    # Set Seeds
    # random.seed(1234)
    # np.random.seed(1234)
    # torch.manual_seed(1234)
    # torch.cuda.manual_seed_all(1234)

    # Build model
    net = edmnet()
    # net.apply(weights_init_kaiming)
    criterion = nn.L1Loss()
    criterion2 = nn.MSELoss()

    # Move to GPU
    model = net.cuda()

    ## if you want to train resuming
    # load = torch.load(os.path.join(opt.logdir, 'gopro_rfa_330_0.03.pth'))
    # model.load_state_dict(load)
    # summary(net, (3, 128, 128))

    criterion.cuda()
    criterion2.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # training
    writer = SummaryWriter(opt.tensorboardx)
    step = 0

    start_time = datetime.now()
    print('Training Start!!')
    print(start_time)

    for epoch in range(opt.resume_epochs, opt.epochs):
        # set learning rate
        if (epoch + 1) < opt.decaystart_epochs:
            current_lr = opt.lr
        else:
            current_lr = opt.lr * (0.5 ** (((epoch + 1) - (opt.decaystart_epochs - 200)) // opt.step))
            # current_lr = current_lr * 0.5
            if current_lr <= 0.5e-6:
                current_lr = 0.5e-6

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %.8f' % current_lr)
        loss_val = 0

        # train
        for i, (blur_train, gt_train) in enumerate(dataset_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # Make Edge Image.
            original_edge = torch.FloatTensor(blur_train.shape[0], 3, opt.patchsize, opt.patchsize)

            for j in range(blur_train.shape[0]):
                original_edge[j] = toTensor((toPILImg(gt_train[j]).filter(ImageFilter.FIND_EDGES)))

            blur_train, gt_train = Variable(blur_train.cuda()), Variable(gt_train.cuda())
            original_edge = Variable(original_edge.cuda())

            out_train, edge = model(blur_train)

            if epoch < 1600:
                blur_loss = criterion(out_train, gt_train)
                edge_loss = criterion(edge, original_edge)
            else:
                blur_loss = criterion2(out_train, gt_train)
                edge_loss = criterion(edge, original_edge)


            loss = blur_loss + 0.5 * edge_loss
            loss_val += loss.item()

            loss.backward()
            optimizer.step()

            # results
            model.eval()
            out_train, edge = model(blur_train)
            out = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out, gt_train, 1.)
            # i%100 == 0 -> each 100 epochs, print loss and psnr.
            if i % 500 == 0 :
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                    (epoch+1, i+1, len(dataset_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if i == 0:
                Img = utils.make_grid(gt_train[0].data, nrow=8, normalize=True, scale_each=True)
                Imgn = utils.make_grid(blur_train[0].data, nrow=8, normalize=True, scale_each=True)
                edgeImg = utils.make_grid(edge[0].data, nrow=8, normalize=True, scale_each=True)
                Irecon = utils.make_grid(out[0].data, nrow=8, normalize=True, scale_each=True)

                # Compare clean, blur, deblurring image
                fig = plt.figure()
                fig.suptitle('edmnet %d' % (epoch + 1))
                rows = 2
                cols = 2

                ax1 = fig.add_subplot(rows, cols, 1)
                ax1.imshow(np.transpose(Img.cpu(), (1,2,0)), cmap="gray")
                ax1.set_title('gt image')

                ax2 = fig.add_subplot(rows, cols, 2)
                ax2.imshow(np.transpose(Imgn.cpu(), (1,2,0)), cmap="gray")
                ax2.set_title('blur image')

                ax3 = fig.add_subplot(rows, cols, 3)
                ax3.imshow(np.transpose(edgeImg.cpu(), (1,2,0)), cmap="gray")
                ax3.set_title('edge image')

                ax4 = fig.add_subplot(rows, cols, 4)
                ax4.imshow(np.transpose(Irecon.cpu(), (1, 2, 0)), cmap="gray")
                ax4.set_title('deblur image [%.4f %.4f]' % (loss.item(), psnr_train))

                # plt.savefig('./fig_result/epoch_{:d}.png'.format(epoch + 1))
                plt.show()

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        # the end of each epoch
        # model.eval()
        if (epoch + 1) % 10 == 0:
            loss_val /= len(dataset_train)
            print("Average Loss : %.4f" % (loss_val))

            midtime = datetime.now() - start_time
            print(midtime)

            torch.save(model.state_dict(), os.path.join(opt.outf, 'edmnet' + str(epoch + 1) + "_" + str(round(loss_val, 4)) + '.pth'))

    end_time = datetime.now()
    print('Training Finished!!')
    print(end_time)

if __name__ == "__main__":
    main()

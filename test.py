import argparse
from torch.autograd import Variable
from models.net import *
from utils.utils import *
from utils.dataset import *
import time
from datetime import datetime
from torchsummary import summary
from ptflops import get_model_complexity_info


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="EDMNet:Deblurring Network Using Edge Module, Deformable Convolution and Multi-Stage Network.")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='val_blur_jpeg', help='GoPro Testset')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = edmnet()
    model = net.cuda()
    # summary(model, (3, 256, 256))
    macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=False,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    load = torch.load(os.path.join(opt.logdir, 'edmnet_1000_0.0153.pth'))
    model.load_state_dict(load)
    model.eval()
    # load data info
    print('Loading data info ...\n')
    dataset_val = Dataset(train=False).data

    psnr_val = 0
    ssim_val = 0
    val_time = 0.

    start_time = datetime.now()
    print('Model Test Start!')
    print(start_time)

    for i, (blur_val, gt_val) in enumerate(dataset_val, 0):
        with torch.no_grad():
            gt_val, blur_val = Variable(gt_val.cuda()), Variable(blur_val.cuda())
            beforetime = time.time()
            out_val = model(blur_val)

            aftertime = time.time() - beforetime
            val_time += aftertime

        psnr_test = batch_PSNR(out_val[0], gt_val, 1.)
        psnr_val += psnr_test
        ssim_test = batch_SSIM(out_val[0], gt_val, 1.)
        ssim_val += ssim_test
        # print(val_time)
        print("[%d/%d] PSNR: %.4f SSIM: %.4f" % (i+1, len(dataset_val),psnr_test, ssim_test))

    psnr_val /= len(dataset_val)
    ssim_val /= len(dataset_val)
    val_time /= len(dataset_val)
    print("PSNR_val: %.4f SSIM_val: %.4f Time_avr: %.3f\n" % (psnr_val, ssim_val, val_time))

    end_time = datetime.now() - start_time
    print(end_time)


if __name__ == "__main__":
    main()

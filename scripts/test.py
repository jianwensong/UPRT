import argparse
import os
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from basicsr.models.archs.UPRT_arch import UPRT
import scipy.io

def calc_phase(preds):
    img1 = preds[:,0,:,:]
    img2 = preds[:,1,:,:]
    phase = torch.atan2(-img1,-img2)
    return phase.unsqueeze(1)
def toTensor(img):
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    return img.float()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='experiments/pretrained_model/UPRT.pth')
    parser.add_argument('--eval_file', type=str, default='datasets/test')  #validation
    parser.add_argument('--outputs-dir', type=str, default='results')
    args = parser.parse_args()

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    if not os.path.exists(args.outputs_dir+'/AbsolutePhase'):
        os.makedirs(args.outputs_dir+'/AbsolutePhase')
    if not os.path.exists(args.outputs_dir+'/Error'):
        os.makedirs(args.outputs_dir+'/Error')

    cudnn.benchmark = True
    device = torch.device('cuda:0')


    torch.manual_seed(123)
    model = UPRT().to(device)
    test_state = torch.load(args.log_dir,map_location=device)  
    model.load_state_dict(test_state['params'])
    model.eval()


    datasets = ['FP1000','FP672','FP147','FP1523']
    fs = [48,100,32,80]

    for j in range(len(datasets)):
      MAEs = []
      RMSEs = []
      eval_dataset = args.eval_file+'/'+datasets[j]  #validation
      f = fs[j]
      os.makedirs(args.outputs_dir+'/AbsolutePhase/'+datasets[j], exist_ok=True)
      os.makedirs(args.outputs_dir+'/Error/'+datasets[j], exist_ok=True)
      file_list = sorted(os.listdir(eval_dataset+'/iPattern'))

      for idx_iter in range(len(file_list)):

          inputs = scipy.io.loadmat(eval_dataset+'/iPattern/' + file_list[idx_iter])
          inputs = toTensor(np.array(inputs['iPattern'])/255.)
          label1 = scipy.io.loadmat(eval_dataset+'/PSin/' + file_list[idx_iter])
          label1 = toTensor(np.array(label1['PSin'])/127.5)
          label2 = scipy.io.loadmat(eval_dataset+'/PCos/' + file_list[idx_iter])
          label2 = toTensor(np.array(label2['PCos'])/127.5)
          Order = scipy.io.loadmat(eval_dataset+'/Fringe/' + file_list[idx_iter])
          Order = toTensor(np.array(Order['Order']))

          inputs = inputs.unsqueeze(0).to(device)
          labels = torch.cat([label1,label2],dim=0).unsqueeze(0).to(device)
          Order = Order.unsqueeze(0).to(device)
          GroundWrap = calc_phase(labels)
          maskl = torch.zeros(GroundWrap.shape).float().to(device)

          GroundWrap =torch.where(torch.abs(labels[:,0,:,:]).unsqueeze(1)==0,maskl.float(),GroundWrap)

          if datasets[j]=='FP1000':
            GroundWrap = -GroundWrap + math.pi
          else:
            GroundWrap = GroundWrap + math.pi
          Phase = (GroundWrap + 2*math.pi*Order)/f

          with torch.no_grad():
              preds = model(inputs)

          WrappedPhase = calc_phase(preds)

          WrappedPhase = torch.where(labels[:,0,:,:].unsqueeze(1)==0,maskl,WrappedPhase)

          if datasets[j]=='FP1000':
            WrappedPhase = -WrappedPhase+math.pi
          else:
            WrappedPhase = WrappedPhase+math.pi

          Neworder = torch.round((f*Phase-WrappedPhase)/math.pi/2)
          OrderGround = torch.round((f*Phase-GroundWrap)/math.pi/2)

          NewPhase = (WrappedPhase + 2*math.pi*Neworder.float())/f
          Phase = (GroundWrap + 2*math.pi*OrderGround.float())/f
          Dif = Phase - NewPhase

          MAE = torch.mean(torch.abs(Dif)).cpu()
          RMSE = torch.sqrt(torch.mean((Dif) ** 2)).cpu()
          print('{}:{:.2f}/{:.2f} * 10^-4 rad'.format(file_list[idx_iter].split('.')[0], MAE*10000,RMSE*10000))

          AbsolutePhase = NewPhase.squeeze(0).squeeze(0).float().cpu().numpy()
          scipy.io.savemat(args.outputs_dir + '/AbsolutePhase/'+datasets[j]+'/'+file_list[idx_iter], {'AbsolutePhase':AbsolutePhase})
          Error = Dif.squeeze(0).squeeze(0).float().cpu().numpy()
          scipy.io.savemat(args.outputs_dir + '/Error/'+datasets[j]+'/'+file_list[idx_iter], {'Error':Error})

          MAEs.append(MAE)
          RMSEs.append(RMSE)

      print('Dataset: {} MAE: {:.2f} RMSE: {:.2f} * 10^-4 rad'.format(datasets[j],np.array(MAEs).mean()*10000,np.array(RMSEs).mean()*10000))
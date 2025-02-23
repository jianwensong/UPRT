import torch
import math

def calc_phase(preds):
    img1 = preds[:,:1,:,:]
    img2 = preds[:,1:2,:,:]
    phase = torch.atan2(img1,-img2)
    return phase.unsqueeze(1)

def calculate_phase_rmse(sr,gt,frequency):
    GroundWrap = calc_phase(gt)
    maskl = torch.zeros(GroundWrap.shape).float().to(gt.device)
    GroundWrap =torch.where(gt[:,:1,:,:]==0,maskl.float(),GroundWrap)
    GroundWrap = GroundWrap + math.pi
    Phase = (GroundWrap + 2*math.pi*gt[:,-1:,:,:])/frequency
    WrappedPhase = calc_phase(sr)
    WrappedPhase = torch.where(gt[:,:1,:,:]==0,maskl,WrappedPhase)
    WrappedPhase = WrappedPhase+math.pi
    Neworder = torch.round((frequency*Phase-WrappedPhase)/math.pi/2)
    NewPhase = (WrappedPhase + 2*math.pi*Neworder.float())/frequency
    Dif = Phase - NewPhase
    return math.sqrt(torch.mean((Dif) ** 2)) * 10000

def calculate_SnRn(sr,gt):
    Dif = sr - gt[:,:2,:,:]
    return math.sqrt(torch.mean((Dif) ** 2))
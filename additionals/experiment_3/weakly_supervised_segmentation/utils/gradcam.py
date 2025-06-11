# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot

# Extra library Usage: We used more than three pip installs but this
# code was for the experiment and these extra libraries are not used
# for the final implementation of weakly supervised segmentation framework

import os
import gc
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import optuna

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from .common import log_message
from torchmetrics import JaccardIndex


def percentile_threshold(cam, percentile=80):
    # to avoid CPU/GPU hops
    q = torch.quantile(cam, percentile / 100.0)
    return (cam > q).float()


class GradCAM:
    # Generates Grad-CAM visualizations for a model and target layer
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Register hooks to capture activations and gradients
        def forward_hook(module, inp, outp):
            self.activations = outp
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def __call__(self, input_tensor, target_class=None):

        # Compute GradCAM for the given input tensor

        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        model_out = self.model(input_tensor)
        if isinstance(model_out, dict) and 'out' in model_out:
            output = model_out['out']
            if target_class is None:
                target_class = torch.argmax(output.mean(dim=[2, 3])).item()
            # Creating one-hot map for segmentation
            one_hot = torch.zeros_like(output)
            one_hot[:, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
        else:
            logits = model_out
            _, pred = torch.max(logits, dim=1)
            if target_class is None:
                target_class = pred.item()
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class] = 1
            logits.backward(gradient=one_hot, retain_graph=True)

        # Compute weighted combination of activations and gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach(), target_class

    def remove_hooks(self):
        # Remove all registered hooks
        for handle in self.hook_handles:
            handle.remove()
        for h in self.hook_handles:
            h.remove()
        for h in self.hook_handles:
            h.remove()


def multi_scale_gradcam(model, img, target_class):
    import torch.nn as nn
    # Attempt to locate ResNet-style layers
    net = model
    layer4 = layer3 = None
    if hasattr(net, 'layer4') and hasattr(net, 'layer3'):
        layer4 = net.layer4[-1]
        layer3 = net.layer3[-1]
    else:
        for attr in ('model', 'backbone', 'features'):  # common wrappers
            if hasattr(net, attr):
                sub = getattr(net, attr)
                if hasattr(sub, 'layer4') and hasattr(sub, 'layer3'):
                    layer4 = sub.layer4[-1]
                    layer3 = sub.layer3[-1]
                    break
    if layer4 is not None and layer3 is not None:
        # Weighted combination of two layers
        g4 = GradCAM(model, layer4)
        cam4, _ = g4(img, target_class)
        g3 = GradCAM(model, layer3)
        cam3, _ = g3(img, target_class)
        g4.remove_hooks()
        g3.remove_hooks()
        return cam4 * 0.7 + cam3 * 0.3
    # Fallback - use the last convolutional layer in the model
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise AttributeError(f"No Conv2d layers found in model {model.__class__.__name__}")
    last_conv = convs[-1]
    g = GradCAM(model, last_conv)
    cam, _ = g(img, target_class)
    g.remove_hooks()
    return cam


def apply_crf(
    image, cam, initial_mask=None,
    sxy_gauss=2, compat_gauss=4,
    sxy_bilat=60, srgb=10, compat_bilat=12,
    n_iters=5
):
    img_np = (image.detach().cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)
    cam_np = cam.detach().cpu().numpy()[0,0]
    if initial_mask is not None:
        init = initial_mask.cpu().numpy()[0,0]
        prob = 0.5*np.stack([1-init, init],axis=0) + 0.5*np.stack([1-cam_np,cam_np],axis=0)
    else:
        prob = np.stack([1-cam_np,cam_np],axis=0)
    prob = np.ascontiguousarray(prob)
    d = dcrf.DenseCRF2D(img_np.shape[1], img_np.shape[0], 2)
    d.setUnaryEnergy(unary_from_softmax(prob))
    d.addPairwiseGaussian(sxy=sxy_gauss, compat=compat_gauss)
    d.addPairwiseBilateral(sxy=sxy_bilat, srgb=srgb, rgbim=img_np, compat=compat_bilat)
    Q = d.inference(n_iters)
    mask = np.argmax(Q,axis=0).reshape(cam_np.shape)
    return torch.tensor(mask,device=cam.device).float().unsqueeze(0).unsqueeze(0)


def contour_refinement(binary_mask):
    m = binary_mask.cpu().numpy()[0,0].astype(np.uint8)
    ctrs,_ = cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    if ctrs:
        c = max(ctrs,key=cv2.contourArea)
        cv2.drawContours(out,[c],-1,1,-1)
    return torch.tensor(out,device=binary_mask.device).float().unsqueeze(0).unsqueeze(0)


def save_gradcam_visualization(image, cam, mask, path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    img = image.detach().cpu().numpy()[0].transpose(1,2,0)
    img = (img-img.min())/(img.max()-img.min()+1e-8)
    heat = cam.detach().cpu().numpy()[0,0]
    binm = mask.detach().cpu().numpy()[0,0]
    heat_col = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_col = cv2.cvtColor(heat_col,cv2.COLOR_BGR2RGB)/255.0
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.imshow(img); plt.title('Image'); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(heat,cmap='jet'); plt.title('CAM'); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(binm,cmap='gray'); plt.title('Mask'); plt.axis('off')
    plt.tight_layout(); plt.savefig(path); plt.close()


def generate_pseudo_masks(
    classifier, dataloader, device,
    save_visualizations=False,
    visualization_dir='vis',
    crf_params=dict(
        sxy_gauss=2, compat_gauss=4,
        sxy_bilat=60, srgb=10, compat_bilat=12,
        n_iters=5
    )
):
    classifier.eval()
    g4 = GradCAM(classifier, classifier.layer4[-1])
    g3 = GradCAM(classifier, classifier.layer3[-1])
    out=[]
    stats={'total':0,'fg':0,'pix':0}
    if save_visualizations:
        os.makedirs(visualization_dir,exist_ok=True)
        log_message(f'Saving in {visualization_dir}')
    for batch in tqdm(dataloader,desc='Masks'):
        imgs,labels = batch['image'].to(device), batch['label'].to(device)
        for i in range(imgs.size(0)):
            img=imgs[i:i+1].clone().detach().requires_grad_(True)
            cls=labels[i].item()
            cam = multi_scale_gradcam(classifier, img, cls)
            mask = percentile_threshold(cam,80)
            if mask.sum()<0.01*mask.numel():
                for p in (70,60,50):
                    mask=percentile_threshold(cam,p)
                    if mask.sum()>=0.01*mask.numel(): break
            if mask.sum()==0:
                mask=(cam>cam.mean()).float()
            mask=apply_crf(img,cam,initial_mask=mask,**crf_params)
            mask=contour_refinement(mask)
            stats['total']+=1; stats['fg']+=mask.sum().item(); stats['pix']+=mask.numel()
            if save_visualizations:
                fname=f"vis_{stats['total']}_cls{cls}.png"
                save_gradcam_visualization(imgs[i:i+1],cam,mask,os.path.join(visualization_dir,fname))
            out.append({'image':imgs[i:i+1].cpu(),'pseudo_mask':mask.cpu(),'true_class':cls})
    g4.remove_hooks(); g3.remove_hooks()
    avg_cov=100*stats['fg']/stats['pix'] if stats['pix']>0 else 0
    log_message(f"Generated {stats['total']} masks, avg coverage {avg_cov:.2f}%")
    return out


def tune_crf_params_optuna(
    model, val_dataloader, device,
    gt_extractor=lambda batch: batch['mask'],
    n_trials=10, max_batches=30
):
    # Bayesian optimization of CRF hyperparams using Optuna
    def objective(trial):
        cfg = {
            'sxy_gauss': trial.suggest_float('sxy_gauss', 1.0, 4.0),
            'compat_gauss': trial.suggest_int('compat_gauss', 1, 5),
            'sxy_bilat': trial.suggest_float('sxy_bilat', 20.0, 60.0),
            'srgb': trial.suggest_float('srgb', 5.0, 10.0),
            'compat_bilat': trial.suggest_int('compat_bilat', 1, 10),
            'n_iters': trial.suggest_int('n_iters', 5, 10)
        }
        tune_dev = torch.device('cpu') if device.type=='mps' else device
        model_loc = model.to(tune_dev).eval()
        iou = JaccardIndex(task='binary').to(tune_dev)
        for bidx,batch in enumerate(val_dataloader):
            if bidx >= max_batches: break
            imgs = batch['image'].to(tune_dev)
            lbls = batch['label'].to(tune_dev)
            gtms=gt_extractor(batch).to(tune_dev).long()
            preds, gts = [], []
            for i in range(imgs.size(0)):
                img=imgs[i:i+1].clone().detach().requires_grad_(True)
                cls=lbls[i].item()
                cam=multi_scale_gradcam(model_loc,img,cls)
                m0=percentile_threshold(cam,80)
                pm=apply_crf(img,cam,initial_mask=m0,**cfg)
                pm=contour_refinement(pm).long().squeeze()
                preds.append(pm); gts.append(gtms[i])
            if preds:
                p=torch.stack(preds); t=torch.stack(gts)
                iou.update(p,t)
                del p,t,preds,gts; gc.collect()
        return iou.compute().item()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective,n_trials=n_trials)
    bp, bs = study.best_params, study.best_value
    log_message(f"Best CRF (Optuna): {bp} → IoU {bs:.4f}")
    return bp, bs

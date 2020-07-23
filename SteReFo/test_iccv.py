import os
import re 
from PIL import Image
import math
from SSIM_PIL import compare_ssim

import numpy as np
#%%
folder = r"D:\output_experiments\output_experiments_ICCV\comparison_selection"
files = os.listdir(folder)

files_blur_up = [each_name for each_name in os.listdir(folder) if re.match(r'.*_blur_upsampling.png$', each_name) ]
files_cv_ref = [each_name for each_name in os.listdir(folder) if re.match(r'.*_cv_refinement.png$', each_name) ]
files_gt = [each_name for each_name in os.listdir(folder) if re.match(r'.*_GT.png$', each_name) ]
files_sequ = [each_name for each_name in os.listdir(folder) if re.match(r'.*_sequential.png$', each_name) ]
files_seqy_app = [each_name for each_name in os.listdir(folder) if re.match(r'.*_sequential_aperture.png$', each_name) ]

numbers = [re.findall(r'(.*)_(.*)_blur_upsampling.png', each_name)[0] for each_name in files_blur_up]
nb_files = len(numbers)
#%%
ssims_blur_up = []
ssims_cv_ref = []
ssims_gt = []
ssims_sequ = []
ssims_sequ_app = []
    
for m,n in numbers:
    file_blur_up = folder+"/"+'%d_%d_blur_upsampling.png'%(int(m),int(n))
    file_cv_ref = folder+"/"+'%d_%d_cv_refinement.png'%(int(m),int(n))
    file_gt = folder+"/"+'%d_%d_GT.png'%(int(m),int(n))
    file_sequ = folder+"/"+'%d_%d_sequential.png'%(int(m),int(n))
    file_sequ_app = folder+"/"+'%d_%d_sequential_aperture.png'%(int(m),int(n))
    
    blur_up = Image.open(file_blur_up)
    cv_ref = Image.open(file_cv_ref)
    gt = Image.open(file_gt)
    sequ = Image.open(file_sequ)
    sequ_app = Image.open(file_sequ_app)
    
    ssims_blur_up.append(compare_ssim(gt, blur_up))
    ssims_cv_ref.append(compare_ssim(gt, cv_ref))
    ssims_sequ.append(compare_ssim(gt, sequ))
    ssims_sequ_app.append( compare_ssim(gt, sequ_app))
    #%%
print("mean SSIM blur upsampling = %f (std dev %f)"%(np.mean(ssims_blur_up), np.std(ssims_blur_up)))
print("mean SSIM CV refinement = %f (std dev %f)"%(np.mean(ssims_cv_ref), np.std(ssims_cv_ref)))
print("mean SSIM blur sequ = %f (std dev %f)"%(np.mean(ssims_sequ), np.std(ssims_sequ)))
print("mean SSIM blur sequ app = %f (std dev %f)"%(np.mean(ssims_sequ_app), np.std(ssims_sequ_app)))
#%%
def ssim2dssim(ssim):
    return (1-np.asarray(ssim))/2

print("mean DSSIM blur upsampling = %f (std dev %f)"%(np.mean(ssim2dssim(ssims_blur_up)), np.std(ssim2dssim(ssims_blur_up))))
print("mean DSSIM CV refinement = %f (std dev %f)"%(np.mean(ssim2dssim(ssims_cv_ref)), np.std(ssim2dssim(ssims_cv_ref))))
print("mean DSSIM blur sequ = %f (std dev %f)"%(np.mean(ssim2dssim(ssims_sequ)), np.std(ssim2dssim(ssims_sequ))))
print("mean DSSIM blur sequ app = %f (std dev %f)"%(np.mean(ssim2dssim(ssims_sequ_app)), np.std(ssim2dssim(ssims_sequ_app))))

#%%
psnrs_blur_up = []
psnrs_cv_ref = []
psnrs_gt = []
psnrs_sequ = []
psnrs_sequ_app = []

def compare_psnr(img1, img2):
    mse = np.mean( (np.asarray(img1) - np.asarray(img2)) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))    


for m,n in numbers:
    file_blur_up = folder+"/"+'%d_%d_blur_upsampling.png'%(int(m),int(n))
    file_cv_ref = folder+"/"+'%d_%d_cv_refinement.png'%(int(m),int(n))
    file_gt = folder+"/"+'%d_%d_GT.png'%(int(m),int(n))
    file_sequ = folder+"/"+'%d_%d_sequential.png'%(int(m),int(n))
    file_sequ_app = folder+"/"+'%d_%d_sequential_aperture.png'%(int(m),int(n))
    
    blur_up = Image.open(file_blur_up)
    cv_ref = Image.open(file_cv_ref)
    gt = Image.open(file_gt)
    sequ = Image.open(file_sequ)
    sequ_app = Image.open(file_sequ_app)
    
    psnrs_blur_up.append(compare_psnr(gt, blur_up))
    psnrs_cv_ref.append(compare_psnr(gt, cv_ref))
    psnrs_sequ.append(compare_psnr(gt, sequ))
    psnrs_sequ_app.append(compare_psnr(gt, sequ_app))
    

print("mean PSNR blur upsampling = %f (std dev %f)"%(np.mean(psnrs_blur_up), np.std(psnrs_blur_up)))
print("mean PSNR CV refinement = %f (std dev %f)"%(np.mean(psnrs_cv_ref), np.std(psnrs_cv_ref)))
print("mean PSNR blur sequ = %f (std dev %f)"%(np.mean(psnrs_sequ), np.std(psnrs_sequ)))
print("mean PSNR blur sequ app = %f (std dev %f)"%(np.mean(psnrs_sequ_app), np.std(psnrs_sequ_app)))

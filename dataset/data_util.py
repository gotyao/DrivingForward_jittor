import os

import numpy as np
import PIL.Image as pil

import jittor as jt
from jittor import transform

_DEL_KEYS= ['rgb', 'rgb_context', 'rgb_original', 'rgb_context_original', 'intrinsics', 'contexts', 'splitname', 'ego_pose'] 


def transform_mask_sample(sample, data_transform):
    """
    This function transforms masks to match input rgb images.
    """
    image_shape = data_transform.keywords['image_shape']
    # resize transform
    resize_transform = transform.Resize(image_shape, mode=pil.ANTIALIAS)
    sample['mask'] = resize_transform(sample['mask'])
    # totensor transform
    tensor_transform = transform.ToTensor()
    sample['mask'] = tensor_transform(sample['mask'])
    return sample


def img_loader(path):
    """
    This function loads rgb image.
    """
    with open(path, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('RGB')


def mask_loader_scene(path, mask_idx, cam):
    """
    This function loads mask that correspondes to the scene and camera.
    """
    fname = os.path.join(path, str(mask_idx), '{}_mask.png'.format(cam.upper()))    
    with open(fname, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('L')


def align_dataset(sample, scales, contexts):
    """
    This function reorganize samples to match our trainer configuration.
    """
    K = sample['intrinsics']
    aug_images = sample['rgb']
    aug_contexts = sample['rgb_context']
    org_images = sample['rgb_original']
    org_contexts= sample['rgb_context_original']
    ego_poses = sample['ego_pose']

    n_cam, _, w, h = aug_images.shape

    # initialize intrinsics
    resized_K = np.expand_dims(np.eye(4), 0).repeat(n_cam, axis=0)
    resized_K[:, :3, :3] = K

    # augment images and intrinsics in accordance with scales 
    for scale in scales:
        scaled_K = resized_K.copy()
        scaled_K[:,:2,:] /= (2**scale)
        
        sample[('K', scale)] = scaled_K.copy()
        sample[('inv_K', scale)]= np.linalg.pinv(scaled_K).copy()

        resized_org = jt.nn.interpolate(org_images,
                                          size=(int(w//(2**scale)),int(h//(2**scale))),
                                          mode = 'bilinear',
                                          align_corners=False)
        resized_aug = jt.nn.interpolate(aug_images,
                                          size=(int(w//(2**scale)),int(h//(2**scale))),
                                          mode = 'bilinear',
                                          align_corners=False)            
            
        sample[('color', 0, scale)] = resized_org
        sample[('color_aug', 0, scale)] = resized_aug

    # for context data
    for idx, frame in enumerate(contexts):
        sample[('color', frame, 0)] = org_contexts[idx]        
        sample[('color_aug',frame, 0)] = aug_contexts[idx]
        sample[('cam_T_cam', 0, frame)] = ego_poses[idx]
        
    # delete unused arrays
    for key in list(sample.keys()):
        if key in _DEL_KEYS:
            del sample[key]
    return sample

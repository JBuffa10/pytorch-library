import numpy as np
import torch
class HeatMap(object):
    def __init__(self, sigma=4):
        self.sigma = sigma

    def __call__(self, image, keypoints):

        from scipy.ndimage import gaussian_filter

        H = image.shape[1]
        W = image.shape[2]
        nKeypoints = len(keypoints)

        img_hm = np.zeros(shape=(H, W, nKeypoints), dtype=np.float32)
    
        for i in range(nKeypoints-1):
            x = int(keypoints[i][1])
            y = int(keypoints[i][0])

            z = np.zeros(shape=(H,W))
            if x < W and y < H:
                z[x, y] = 1
                channel_hm = gaussian_filter(z, sigma=self.sigma)
                img_hm[:, :, i] = channel_hm
                # img_hm[:,:,i] = z
            else:
                continue

        im_hm_reshape = img_hm.sum(axis=2, keepdims=False)
        hm = torch.unsqueeze(torch.tensor(im_hm_reshape),0)
        # hm = torch.tensor(im_hm_reshape)
        # torch.unsqueeze(hm,0) ## Didn't work with Crossentropy loss??

        return hm

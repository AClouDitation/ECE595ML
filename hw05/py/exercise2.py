#! /usr/bin/env python3
import numpy as np
import scipy.ndimage
from scipy.misc import imsave

def readImg(pix):
    return  scipy.ndimage.imread(pix, mode='L') /255


def get_info():
    train_cat   = np.matrix(np.loadtxt('../data/train_cat.txt', delimiter=','))
    train_grass = np.matrix(np.loadtxt('../data/train_grass.txt', delimiter=','))

    mu_cat      = np.asmatrix(np.mean(train_cat, 1))
    mu_grass    = np.asmatrix(np.mean(train_grass, 1))

    cov_cat     = np.asmatrix(np.cov(train_cat))
    cov_grass   = np.asmatrix(np.cov(train_grass))

    pi_cat      = len(train_cat.T) / (len(train_cat.T) + len(train_grass.T))
    pi_grass    = len(train_grass.T) / (len(train_cat.T) + len(train_grass.T))

    return {"mean":mu_cat, "cov":cov_cat, "prior":pi_cat}, {"mean":mu_grass, "cov":cov_grass, "prior":pi_grass}



def CWAttack(cat_info, grass_info, img, rounds = 300, alpha = 0.0001, lm = 5):

    Wcat = np.linalg.inv(cat_info["cov"])
    wcat = -Wcat * cat_info["mean"]
    w0cat = (cat_info["mean"].T * Wcat * cat_info["mean"]) / 2 \
        + np.log(np.linalg.det(cat_info["cov"])) / 2 - np.log(cat_info["prior"])

    Wgrass = np.linalg.inv(grass_info["cov"])
    wgrass = -Wgrass * grass_info["mean"]
    w0grass = grass_info["mean"].T * Wgrass * grass_info["mean"] / 2 \
        + np.log(np.linalg.det(grass_info["cov"])) / 2 - np.log(grass_info["prior"])


    def __gee(x):
        
        # Wt == Wgrass,  Wj == Wcat
        ret = x.T * (Wgrass - Wcat) * x / 2 + (wgrass - wcat).T * x + (w0grass - w0cat)
        return ret


    def gradient(z_vector, z_0, lm):
        # Calculate g_j, g_t, determine if patch_vec is already in target class
        # If patch_vec is in target class, do not add any perturbation (return zero gradient!)
        # Else, calculate the gradient, using results from 1(c)(ii)
        
        if __gee(z_vector) > 0:
            return 2 * (z_vector - z_0) + lm * ((Wgrass - Wcat) * z_vector + (wgrass - wcat)) 
        else:
            #return 2 * (z_vector - z_0)
            return np.zeros((64,1))


    M,N = img.shape
    img_orig = img.copy()
    range_i = range(0,M-8,8)
    range_j = range(0,N-8,8)


    def classify(img):
        output = np.zeros((M,N))
        for i in range_i:
            for j in range_j:
                z = img[i:i+8, j:j+8]
                z_vector = np.asmatrix(z.flatten('F')).T
                if __gee(z_vector) > 0:
                    for ii in range(8):
                        for jj in range(8):
                            output[i+ii][j+jj] = 1
        return output


    def cntclass(img):
        num_cat = 0
        num_grass = 0
        for i in range_i:
            for j in range_j:
                z = img[i:i+8, j:j+8]
                z_vector = np.asmatrix(z.flatten('F')).T
                if __gee(z_vector) < 0:
                    num_cat+=1
                else:
                    num_grass+=1
        return num_grass, num_cat

    for r in range(rounds):
        if r % (50 / lm) == 0: 
            print(f"round: {r}")
            imsave(f'../pix/CW_Nonoverlap_{lm}_r{r}.png', 255*classify(img))

        img_prev = img.copy()
        for i in range_i:
            for j in range_j:
                z = img[i:i+8, j:j+8]
                z_vector = np.asmatrix(z.flatten('F')).T
                z_vector_orig = np.asmatrix(img_orig[i:i+8, j:j+8].flatten('F')).T
                z_vector -= alpha * gradient(z_vector, z_vector_orig, lm)
                img[i:i+8, j:j+8] = np.reshape(np.clip(z_vector, 0.0, 1.0), (8,8), order='F')

            
        change = np.linalg.norm(img - img_prev)
        if(change < 0.001):
            print(f"finished, total round: {r}")
            break;


    cnts = cntclass(img)
    print(f"number of grass patches {cnts[0]}, cat patches {cnts[1]}")
    change = np.linalg.norm(img - img_orig)
    print(f"Frobenius norm: {change}")

    imsave(f'../pix/CW_Nonoverlap_{lm}.png', img)
    imsave(f'../pix/CW_Nonoverlap_{lm}_perturbation.png',img-img_orig)
    imsave(f'../pix/CW_Nonoverlap_{lm}_classified_pertub.png', classify(img))

if __name__ == "__main__":
    cat_info, grass_info = get_info()
    CWAttack(cat_info, grass_info, readImg('../data/cat_grass.jpg'), lm = 1)
    CWAttack(cat_info, grass_info, readImg('../data/cat_grass.jpg'), lm = 5)
    CWAttack(cat_info, grass_info, readImg('../data/cat_grass.jpg'), lm = 10)



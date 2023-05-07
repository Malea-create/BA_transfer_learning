from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
from scipy.spatial import distance
from scipy.special import kl_div
from sklearn.metrics import mean_absolute_percentage_error

import sys
modname = globals()['__name__']
modobj  = sys.modules[modname]


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define MMD

"""
    Source: https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
    
"""

def MMD(x, y, kernel):
    
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
      

    return torch.mean(XX + YY - 2. * XY)

"""
    Ende of code taken from: https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
    
"""

"""
    Source: https://stackoverflow.com/questions/50307707/how-do-i-convert-a-pandas-dataframe-to-a-pytorch-tensor
    
"""

# determine the supported device

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU
    return device
    
# convert a df to tensor to be used in pytorch

def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)

"""
    Ende of code taken from: https://stackoverflow.com/questions/50307707/how-do-i-convert-a-pandas-dataframe-to-a-pytorch-tensor
    
"""

def get_transferability_metrics(df_src, df_tar):

    '''
    calculate all transferability metrics

    :param df_src: dataframe including all source domain sampels

    :param df_tar: dataframe including all target domain sampels

    :return beta: list with all transferability metriks
    '''

    save_values = []

    if len(df_src) and len(df_tar) > 0:

        save_values.append("df_src")
        save_values.append("df_tar")

        # source
        Xs = df_src.iloc[:,:-1]
        Ys = df_src.iloc[:,-1]

        # target_train
        Xt = df_tar.iloc[:,:-1]
        Yt = df_tar.iloc[:,-1]

        #Kullback-Leibler Divergenz
        kull = []
        interator = int (len(df_src)/len(df_tar))
        c=0 

        for i in range (1, interator+1):

            kull.append(kl_div(df_src[c:i*len(df_tar)],df_tar).mean())
            c=i*len(df_tar)

        kld = np.ma.masked_invalid(kull).mean()
        save_values.append(kld)

            
        # Cosine Distances
        cosine = []
        interator = int (len(df_src)/len(df_tar))
        c=0 

        for i in range (1, interator+1):

            cosine.append(paired_cosine_distances(df_src[c:i*len(df_tar)],df_tar).mean())
            c=i*len(df_tar)

        cd = np.mean(cosine)
        save_values.append(cd)


        # Wasserstein Distance
        was = wasserstein_distance(Ys,Yt)
        save_values.append(was)

        # Ad-Distance
        ad = ks_2samp(Ys,Yt)[0]
        save_values.append(ad)

        # Jensen Shannon
        jensen = []
        interator = int (len(df_src)/len(df_tar))
        c=0 

        for i in range (1, interator+1):

            jensen.append(distance.jensenshannon(df_src[c:i*len(df_tar)],df_tar).mean())
            c=i*len(df_tar)

        js = np.mean(jensen)
        save_values.append(js)

        # MMD

        mmd_list = []
        interator = int (len(df_src)/len(df_tar))
        c=0 

        for i in range (1, interator+1):

            df_src_tensor = df_to_tensor(df_src[c:i*len(df_tar)])
            df_tar_tensor = df_to_tensor(df_tar)
            result = MMD(df_src_tensor, df_tar_tensor, kernel="multiscale")
            mmd_list.append(result.item())
            c=i*len(df_tar)

        mmd = np.mean(mmd_list)
        save_values.append(mmd)

        Values.append(save_values)

        return Values
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.rcParams['text.usetex'] =  True
import numpy as np
import os
# Read a .trc file and print its content
import threading
import time
from lecroyscope import Trace

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tifffile import imread
from tqdm import trange
from skimage import filters
from skimage.feature import peak_local_max
from datetime import datetime 

def tagger_event(name,ceil,ceil_hminus,left,right):
    
    '''
    ceil        [V] : threshold for mixing events
    ceil_hminus [V] : threshold for H- events
    left        [s] : left time limit for H- events
    right       [s] : right time limit for H- events
    '''

    fname          = name
    traceC4        = Trace(fname)
    traceC3        = Trace(fname.replace('_C4.trc', '_C3.trc'))
    traceC4time    = traceC4.time
    traceC4voltage = traceC4.voltage
    traceC4voltage = traceC4voltage-traceC4voltage[-1]
    traceC3voltage = traceC3.voltage
    meanC3         = np.mean(traceC3voltage)
    if meanC3 < ceil:
        tag = 0
        for k in range(len(traceC4time)):
            if traceC4time[k] >left and traceC4time[k] < right:
                if traceC4voltage[k] > ceil_hminus:
                    tag = 1
        if tag == 1 :
            return 'HMINUS_MIXING',fname
        else:
            return 'BACKGD_MIXING',fname
    else:
        tag2 = 0
        for k in range(len(traceC4time)):
            if traceC4time[k] >left and traceC4time[k] < right:
                if traceC4voltage[k] > ceil_hminus:
                    tag2 = 1
        if tag2 == 1 :
            return 'HMINUS_BACKGROUND',fname
        else:
            return 'BACKGD_BACKGROUND',fname

def plot_tif(root,date,threshold_multiplier,min_peaks,patch_radius,min_distance):
    dir = ''#date+'/'#+date+'scope173/'
    hminus_mixing = np.loadtxt(root+date+'/Hminus_mixing.txt', dtype=str)
    hminus_mixing = np.char.replace(hminus_mixing, '/'+date+'scope173', '')
    hminus_mixing = np.char.replace(hminus_mixing, 'WF173_', 'PCO-MCP6_exp_1_us_')
    hminus_mixing = np.char.replace(hminus_mixing, '_C4.trc', '.tif')
    backgd_mixing = np.loadtxt(root+date+'/Backgd_mixing.txt', dtype=str)
    backgd_mixing = np.char.replace(backgd_mixing, '/'+date+'scope173', '')
    backgd_mixing = np.char.replace(backgd_mixing, 'WF173_', 'PCO-MCP6_exp_1_us_')
    backgd_mixing = np.char.replace(backgd_mixing, '_C4.trc', '.tif')
    hminus_background = np.loadtxt(root+date+'/Hminus_background.txt', dtype=str)
    hminus_background = np.char.replace(hminus_background, '/'+date+'scope173', '')
    hminus_background = np.char.replace(hminus_background, 'WF173_', 'PCO-MCP6_exp_1_us_')
    hminus_background = np.char.replace(hminus_background, '_C4.trc', '.tif')
    backgd_background = np.loadtxt(root+date+'/Backgd_background.txt', dtype=str)
    backgd_background = np.char.replace(backgd_background, '/'+date+'scope173', '')
    backgd_background = np.char.replace(backgd_background, 'WF173_', 'PCO-MCP6_exp_1_us_')
    backgd_background = np.char.replace(backgd_background, '_C4.trc', '.tif')
    
    dir = root+dir
    sum_image = None
    for tif_file in hminus_mixing:
        try:
            image = imread(os.path.join(dir, tif_file))
            if sum_image is None:
                    sum_image = np.zeros_like(image)
            threshold = filters.threshold_otsu(image)
            peaks     = peak_local_max(image, min_distance=min_distance, threshold_abs=threshold * threshold_multiplier)
            for (y, x) in peaks:
                y1, y2   = max(0, y - patch_radius), min(image.shape[0], y + patch_radius + 1)
                x1, x2   = max(0, x - patch_radius), min(image.shape[1], x + patch_radius + 1)
                sum_image[y1:y2, x1:x2] += image[y1:y2, x1:x2]
        except FileNotFoundError:
            print(f"File not found: {tif_file} — skipping.")
            continue
    sum_image1 = sum_image

    sum_image = None
    for tif_file in hminus_background:
        try:
            image = imread(os.path.join(dir, tif_file))
            if sum_image is None:
                    sum_image = np.zeros_like(image)
            threshold = filters.threshold_otsu(image)
            peaks     = peak_local_max(image, min_distance=min_distance, threshold_abs=threshold * threshold_multiplier)
            for (y, x) in peaks:
                y1, y2   = max(0, y - patch_radius), min(image.shape[0], y + patch_radius + 1)
                x1, x2   = max(0, x - patch_radius), min(image.shape[1], x + patch_radius + 1)
                sum_image[y1:y2, x1:x2] += image[y1:y2, x1:x2]
        except FileNotFoundError:
            print(f"File not found: {tif_file} — skipping.")
            continue
    sum_image2 = sum_image

    sum_image = None
    for tif_file in backgd_mixing:
        try:
            image = imread(os.path.join(dir, tif_file))
            if sum_image is None:
                    sum_image = np.zeros_like(image)
            threshold = filters.threshold_otsu(image)
            peaks     = peak_local_max(image, min_distance=min_distance, threshold_abs=threshold * threshold_multiplier)
            for (y, x) in peaks:
                y1, y2   = max(0, y - patch_radius), min(image.shape[0], y + patch_radius + 1)
                x1, x2   = max(0, x - patch_radius), min(image.shape[1], x + patch_radius + 1)
                sum_image[y1:y2, x1:x2] += image[y1:y2, x1:x2]
        except FileNotFoundError:
            print(f"File not found: {tif_file} — skipping.")
            continue
    sum_image3 = sum_image

    sum_image = None
    for tif_file in backgd_background:
        try:
            image = imread(os.path.join(dir, tif_file))
            if sum_image is None:
                    sum_image = np.zeros_like(image)
            threshold = filters.threshold_otsu(image)
            peaks     = peak_local_max(image, min_distance=min_distance, threshold_abs=threshold * threshold_multiplier)
            for (y, x) in peaks:
                y1, y2   = max(0, y - patch_radius), min(image.shape[0], y + patch_radius + 1)
                x1, x2   = max(0, x - patch_radius), min(image.shape[1], x + patch_radius + 1)
                sum_image[y1:y2, x1:x2] += image[y1:y2, x1:x2]
        except FileNotFoundError:
            print(f"File not found: {tif_file} — skipping.")
            continue
    sum_image4 = sum_image

    return sum_image1, sum_image2, sum_image3, sum_image4


    

if __name__ == "__main__" :
    start = 1
    if start ==1:
        print('1')
        # Example usage
        root = 'Z:/data/'
        date = '25_07_11'
        time_cutoff = datetime(2025,7,11,17,40)
        cutoff_timestamp = time.mktime(time_cutoff.timetuple())
        
        dir_name  = date+'/'+date+'scope173/'
        trc_files = [f for f in os.listdir(root+dir_name+'.') if f.endswith('_C4.trc') and os.path.getmtime(os.path.join(root+dir_name+'.',f)) > cutoff_timestamp]
        
        ceil                 = -0.0020    # between mixing and background
        ceil_hminus          = 0.0010     # between H- and background
        left                 = 0.00010807
        right                = 0.00010827
        threshold_multiplier = 1.4  # you can tune this
        min_peaks            = 1 # minimal number of "dots" (peaks) to keep an image
        patch_radius         = 6   # patch size = (2*radius+1)
        min_distance         = 5
        
        cmap = 'binary'

        hminus_mixing     = []
        backgd_mixing     = []
        hminus_background = []
        backgd_background = []
        
        for w in trange(len(trc_files)):  # Adjust the range as needed
            tag, fname = tagger_event(root+dir_name+trc_files[w], ceil, ceil_hminus, left, right)
            if tag == 'HMINUS_MIXING':
                hminus_mixing.append(fname)
            elif tag == 'BACKGD_MIXING':
                backgd_mixing.append(fname)
            elif tag == 'HMINUS_BACKGROUND':
                hminus_background.append(fname)
            elif tag == 'BACKGD_BACKGROUND':
                backgd_background.append(fname)
        
        np.savetxt(root+date+'/Hminus_mixing.txt', hminus_mixing, fmt='%s')
        np.savetxt(root+date+'/Backgd_mixing.txt', backgd_mixing, fmt='%s')
        np.savetxt(root+date+'/Hminus_background.txt', hminus_background, fmt='%s')
        np.savetxt(root+date+'/Backgd_background.txt', backgd_background, fmt='%s')
        
        #plt.clf()
        image1,image2,image3,image4 = plot_tif(root,date,threshold_multiplier,min_peaks,patch_radius,min_distance)
        # Update the image in the plot
        fig, ax = plt.subplots(2, 2, figsize=(8,8), dpi=100, sharex=True,sharey=True)
        
        ax[0,0].imshow(image1,cmap)
        ax[0,1].imshow(image2,cmap)
        ax[1,0].imshow(image3,cmap)
        ax[1,1].imshow(image4,cmap)

        hminus_mixing = np.loadtxt(root+date+'/Hminus_mixing.txt', dtype=str)
        hminus_background = np.loadtxt(root+date+'/hminus_background.txt', dtype=str)
        backgd_mixing = np.loadtxt(root+date+'/backgd_mixing.txt', dtype=str)
        backgd_background = np.loadtxt(root+date+'/backgd_background.txt', dtype=str)
        ax[0,0].set_title('H- mixing :'+str(int(len(hminus_mixing)))+' shots')
        ax[0,1].set_title('H- background:' + str(len(hminus_background)) + ' shots')
        ax[1,0].set_title('background mixing : ' + str(len(backgd_mixing)) + ' shots')
        ax[1,1].set_title('background background :' + str(len(backgd_background)) + ' shots')
        now = datetime.now()
        print('last update : ', now)

        fig.suptitle(str(now))
        plt.savefig('Z:/data/'+date+'/ALL-TIF.png', dpi=300, bbox_inches='tight')
        
        time.sleep(10)

     

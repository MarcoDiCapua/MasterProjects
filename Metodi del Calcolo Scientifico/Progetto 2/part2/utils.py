from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

def is_grey_scale(img):
    im = img.convert('RGB')
    im = np.array(im)
    if (im[:,:,0] == im[:,:,1]).all() and (im[:,:,1] == im[:,:,2]).all():
        return True
    return False

def check_image(path):
    try:
        img = Image.open(path)
    except:
        return ["",1]
    if img.format != "BMP":
        return ["",2]
    if not is_grey_scale(img):
        return ["",3]
    return [img, 0]

def fill_image(arr, f, r ,c):
    column_end=0
    row_end=0
    for i in range(0,r,f):
        if i+f>r:
            row_end=i
    no_compress_r = r-row_end
    arr[row_end:r,:] = arr[row_end-no_compress_r:row_end,:]
    for j in range(0,c,f):
        if j+f>c:
            column_end=j
    no_compress_c = c-column_end
    arr[:,column_end:c] = arr[:,column_end-no_compress_c:column_end]
    return arr


def image_compress(path, f, d):
    arr = np.array(Image.open(path), dtype=int)
    size = arr.shape
    if len(size) == 3:
        arr = arr[:,:,0]
    r = size[0]
    c = size[1]
    arr_compress = arr.copy()

    for i in range(0,r,f):
        for j in range(0,c,f):
            if i+f<r and j+f<c:
                block = arr[i:i+f, j:j+f]
                freq = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
                for a in range(f):
                    for b in range(f):
                        if a+b>d:
                            freq[a, b] = 0
                compress = idct(idct(freq, axis=0, norm='ortho'), axis=1, norm='ortho')
                arr_compress[i:i+f, j:j+f] = compress
    
    arr_compress = np.where(arr_compress<0, 0, arr_compress)
    arr_compress = np.where(arr_compress>255, 255, arr_compress)
    arr_compress = fill_image(arr_compress, f, r, c)
    if c>r:
        fig = plt.figure(figsize=(15,10), dpi=100)
        ax1=fig.add_subplot(2,1,1)
        ax1.imshow(arr, cmap=plt.cm.gray)
        ax2=fig.add_subplot(2,1,2)
        ax2.imshow(arr_compress, cmap=plt.cm.gray)
        fig.savefig('out.png', format='png')
    else:
        fig = plt.figure(figsize=(15,10), dpi=100)
        ax1=fig.add_subplot(1,2,1)
        ax1.imshow(arr, cmap=plt.cm.gray)
        ax2=fig.add_subplot(1,2,2)
        ax2.imshow(arr_compress, cmap=plt.cm.gray)
        fig.savefig('out.png', format='png')

    #im1 = Image.fromarray(arr.astype(np.uint8))
    #im2 = Image.fromarray(arr_compress.astype(np.uint8))
    #im1.save('out1.png', 'PNG')
    #im2.save('out2.png', 'PNG')
    #im1.save('out1.bmp', 'BMP')
    #im2.save(name, 'BMP')

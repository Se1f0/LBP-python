import numpy as np
import cv2
from matplotlib import pyplot as plt

def lbp(M,i_ref,j_ref):
    ref_value = M[i_ref][j_ref]
    bin_val = ""
    #0
    if M[i_ref-1][j_ref-1] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #1
    if M[i_ref-1][j_ref] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #2
    if M[i_ref-1][j_ref+1] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #3
    if M[i_ref][j_ref+1] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #4
    if M[i_ref+1][j_ref+1] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #5
    if M[i_ref+1][j_ref] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #6
    if M[i_ref+1][j_ref-1] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #7 
    if M[i_ref][j_ref-1] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    dec_val = int(bin_val,2) 
    #print(dec_val)
    return dec_val

img_path = 'img.jpg'
img = cv2.imread(img_path,0)
h,w = img.shape[:2]

img_res = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
height,width = img_res.shape[:2]
#print(height,width)

img_lbp = np.zeros((h,w))

for i in range(1,height-1):
    for j in range(1,width-1):
        img_lbp[i-1][j-1] = lbp(img_res,i,j)
cv2.imwrite('img_lbp.jpg', img_lbp)

# plt.imshow(img_lbp,cmap ="gray")
# plt.show()

# plt.hist(img_lbp.reshape(-1),256,[0,256])
# plt.show()
fig,axs = plt.subplots(2,2)
axs[0,0].imshow(img,cmap ="gray")
axs[0,0].set_title('L\'image original aux niveau de gris')
axs[0,1].hist(img.reshape(-1),256,[0,256])
axs[0,1].set_title('L\'histograme de l\'image original')
axs[1,0].imshow(img_lbp,cmap ="gray")
axs[1,0].set_title('L\'image original aprés LBP')
axs[1,1].hist(img_lbp.reshape(-1),256,[0,256])
axs[1,1].set_title('L\'histograme de l\'image aprés LBP')

wm = plt.get_current_fig_manager()
wm.window.state('zoomed')

plt.show()
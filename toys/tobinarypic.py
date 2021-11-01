import matplotlib.pyplot as plt
import matplotlib.image as Image
import numpy as np
# img_y=torch.squeeze(y).numpy()
# im=np.where(img_y >= 0, 1, 0)
# plt.imshow(img_y,cmap='gray')
#
# plt.subplot(222)
# img_label=label.squeeze().cpu().numpy()
# plt.imshow(img_label)
# plt.subplot(223)
# plt.imshow(im,cmap='gray')
if __name__=='__main__':
    img=Image.imread('D:\\OneDrive\\桌面\\final_0_3.png.png')
    #print(img)
    img_bin=np.where(img>=1.0,1.0,0.0)
    print(img_bin)
    plt.imshow(img_bin,cmap='gray')
    plt.show()
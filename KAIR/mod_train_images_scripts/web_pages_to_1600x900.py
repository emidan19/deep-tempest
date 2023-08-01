import os
from PIL import Image
import numpy as np

lista_dir = os.listdir('./')

h_total = 1600
v_total = 900

h_image = 1280
v_image = 1280

I_pad = 255*np.ones((v_total,h_total,3),dtype=np.uint8)

for i, file in enumerate(lista_dir):

    file_ext = file.split('.')[-1]
    file_name = file.split('.'+file_ext)[0]

    if file_ext!= 'png':
        continue
    
    I = Image.open(file)
    I = np.array(I)[:,:,:3]

    if I.shape[:2] != (1280,1280):
        continue

    if i%2==0:
        I_pad[:, (h_total-h_image)//2:-(h_total-h_image)//2, :] = I[:v_total,:,:]
    else:
        I_pad[:, (h_total-h_image)//2:-(h_total-h_image)//2, :] = I[v_image-v_total:,:,:]

    Image.fromarray(I_pad).save(f'./{file_name}.png')
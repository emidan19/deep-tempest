import os
from PIL import Image
import numpy as np

lista_dir = os.listdir('./')

h_total = 1600
v_total = 900

h_image = 1024
v_image = 768

I_pad = 255*np.ones((v_total,h_total,3),dtype=np.uint8)

for file in lista_dir:

    file_ext = file.split('.')[-1]
    file_name = file.split('.'+file_ext)[0]

    if file_ext != 'jpg':
        continue
    
    I = Image.open(file)
    I = np.array(I)

    I_pad[(v_total-v_image)//2:-(v_total-v_image)//2, (h_total-h_image)//2:-(h_total-h_image)//2, :] = I

    Image.fromarray(I_pad).save(f'./Roboflow_dataset_{file_name}.png')
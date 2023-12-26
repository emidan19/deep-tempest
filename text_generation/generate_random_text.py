import os
from datetime import date
import random
import string
from text_utils import generate_random_txt_img

NUM_IMAGES = 10
NUM_CHARACTERS = 25000
IMG_SHAPE = (1600,900)
TEXT_SIZE = 22

today = date.today()

# Month abbreviation, day and year	
save_path = today.strftime("%b-%d-%Y")

if not os.path.exists(save_path):
    os.mkdir(save_path)
else:
    i = 2
    save_path_tmp = save_path + str(i)
    while os.path.exists(save_path_tmp):
        i+=1
        save_path_tmp = save_path + str(i)

images_name = "generated_text"

for i in range(NUM_IMAGES):

    text = ''.join(random.choices(string.ascii_letters +
                                string.digits, k=NUM_CHARACTERS))
    
    text_color = random.choices(["black","white"], weights=(70, 30), k=1)[0]
    background_color = random.choices(["black","white"], weights=(30, 70), k=1)[0]

    generate_random_txt_img(text, 
                            IMG_SHAPE, 
                            TEXT_SIZE, 
                            text_color, 
                            background_color, 
                            os.path.join(save_path,images_name+str(i)+".png"))


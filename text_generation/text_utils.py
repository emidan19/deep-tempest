from PIL import Image, ImageDraw, ImageFont
import random
import string
from matplotlib import font_manager

def generate_random_txt_img(text, img_shape, text_size, text_color, background_color, save_path):
    # Create white plain image
    imagen = Image.new("RGB", img_shape, background_color)
    dibujo = ImageDraw.Draw(imagen)

    # Compute amount of lines depending on image shape and number of characters
    N_total = len(text)
    N_lines = N_total//img_shape[1]
    N_horizontal = img_shape[0]//(text_size)

    # Get system font types
    system_fonts = font_manager.findSystemFonts()
    # Filter out some non-readable fonts
    ttf_fonts = [font for font in system_fonts if ((".ttf" in font) and ("lohit" not in font) and ("kacst" not in font)) and  ("Navilu" not in font) and ("telu" not in font) and ("lyx" not in font) and ("malayalam" not in font) and ("tlwg" not in font) and ("samyak" not in font) and ("droid" not in font) and ("kalapi" not in font) and ("openoffice" not in font) and ("orya" not in font)]

    # Write over image one font per line
    for iter in range(N_lines):
        rnd_font_index = random.randint(0,len(ttf_fonts)-1)
        random_font = ttf_fonts[rnd_font_index]
        # print(f"Font N {iter}: {random_font}")

        # Load text font and set size
        try:
            fuente = ImageFont.truetype(font=random_font, size=text_size)
        except:
            # Load a fixed font when crashes
            fuente = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationSans-BoldItalic.ttf", size=text_size)

        # Get line text
        texto_linea = text[iter * N_horizontal : (iter+1) * N_horizontal]

        # Adjust text position
        posicion_texto = ((imagen.width - fuente.getsize(texto_linea)[0]) // 2, 
                          int(1.5* iter * text_size)
                          )

        # Write text
        dibujo.text(posicion_texto, texto_linea, font=fuente, fill=text_color)

    # Save image
    imagen.save(save_path)


import os.path
import argparse
import numpy as np
import logging
import json

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

# OCR metrics
# First, must install Tesseract: https://tesseract-ocr.github.io/tessdoc/Installation.html
# Second, install CER/WER and tesseract python wrapper libraries
# pip install fastwer
# pip install pybind11
# pip install pytesseract 
import pytesseract
import fastwer

'''
# -------------------------
# Evaluation metric code 
# --------------------------------------------
# Emilio Martínez (emiliomartinez98@gmail.com) 8/2023
'''

def calculate_cer_wer(img_E, img_H):
    # Transcribe ground-truth image to text
    text_H = pytesseract.image_to_string(img_H).strip().replace('\n',' ')

    # Transcribe estimated image to text
    text_E = pytesseract.image_to_string(img_E).strip().replace('\n',' ')

    cer = fastwer.score_sent(text_E, text_H, char_level=True)
    wer = fastwer.score_sent(text_E, text_H)

    return cer, wer

def main(json_path='options/evaluation.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = json.load(open(json_path))

    # ----------------------------------------
    # configure logger
    # ----------------------------------------

    logger_name = 'evaluation_' + opt['dataroot_H'].split('/')[-2]
    utils_logger.logger_info(logger_name, os.path.join(opt['logpath'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    opt = option.dict_to_nonedict(opt)

    border = opt['scale']

    """  
    # ----------------------------------------
    # Step--2 (load paths)
    # ----------------------------------------
    """
    E_paths = util.get_image_paths(opt['dataroot_E'])
    H_paths = util.get_image_paths(opt['dataroot_H'])

    '''
    # ----------------------------------------
    # Step--4 (evaluate estimated images)
    # ----------------------------------------
    '''
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_loss = 0.0
    avg_edgeJaccard = 0.0
    avg_cer = 0.0
    avg_wer = 0.0
    idx = 0

    for E_path, H_path in zip(E_paths,H_paths):
        idx += 1
        image_name_ext = os.path.basename(H_path)

        ###################
        ### Load images ###
        ###################

        # Load ground-truth image and use mean of channels if is RGB
        img_H = util.imread_uint(H_path, n_channels=3)
        if img_H.ndim == 3:
            img_H = np.mean(img_H, axis=2)
            img_H = img_H.astype('uint8')

        # Load estimated image in grayscale
        img_E = util.imread_uint(E_path, n_channels=1)
        img_E = img_E[:,:,0]

        # ----------------------------------------
        # compute PSNR, SSIM, edgeJaccard and CER
        # ----------------------------------------
        current_psnr = util.calculate_psnr(img_E, img_H)
        current_ssim = util.calculate_ssim(img_E, img_H)
        current_edgeJaccard = util.calculate_edge_jaccard(img_E, img_H)
        current_cer, current_wer = calculate_cer_wer(img_E, img_H)

        logger.info('{:->4d}--> {:>10s} | PSNR = {:<4.2f}dB ; SSIM = {:.3f} ; edgeJaccard = {:.3f} ; CER = {:.3f}% ; WER = {:.3f}%'.format(idx, image_name_ext, current_psnr, current_ssim, current_edgeJaccard, current_cer, current_wer))

        avg_psnr += current_psnr
        avg_ssim += current_ssim
        avg_edgeJaccard += current_edgeJaccard
        avg_cer += current_cer
        avg_wer += current_wer

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_edgeJaccard = avg_edgeJaccard / idx
    avg_loss = avg_loss / idx

    # Average log
    logger.info('[Average metrics] PSNR : {:<4.2f}dB, SSIM = {:.3f} : edgeJaccard = {:.3f} : CER = {:.3f}% : WER = {:.3f}%'.format(current_psnr, current_ssim, current_edgeJaccard, current_cer, current_wer))

if __name__ == '__main__':
    main()

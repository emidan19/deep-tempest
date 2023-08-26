

#------------------------------------------Imports--------------------------------------------------------------#
import numpy as np
import torch
import torch.nn as nn
from torchaudio import transforms
from matplotlib import pyplot as plt
from PIL import Image


#----------------------------------------Model-------------------------------------------------------------------#

class Model_qm(nn.Module):
    def __init__(self):
        super(Model_qm, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

#----------------------------------------Model Instances------------------------------------------------------------#

q_m_XOR = Model_qm()
state_dict_XOR = torch.load('forward/q_m_XOR.pth')
q_m_XOR.load_state_dict(state_dict_XOR)
#input must be tensor with float (each bit)
def q_m_XOR_diff(bits):
    return q_m_XOR(bits)

q_m_XNOR = Model_qm()
state_dict_XNOR = torch.load('forward/q_m_XNOR.pth')
q_m_XNOR.load_state_dict(state_dict_XNOR)
#input must be tensor with float (each bit)
def q_m_XNOR_diff(bits):
    return q_m_XNOR(bits)



#---------------------------------------Definition of functions-------------------------------------------------------#

def q_m_diff(input):

    """
        Calculates XOR or XNOR to consecutive bits and sets the bit number 8 with 1 if XOR was used or 0 if XNOR was used.
        This function is used as part of the TMDS codification algorithm but in a differential way to calculate gradients.
	
    Input:	2-D tensor (Nx8) 'N' 1-D tensor with 8 bits (float)

	Output: 2-D tensor (Nx9) 'N' 1-D tensor with 9 bits (float)

	"""
    #output definition with the shape of amount of 8 bits numbers of the input x 9
    output = torch.zeros((input.shape[0],9), dtype=torch.float32)

    #amount of bits that are considered '1'
    num_1 = (input > 0.5).sum(dim=1)

    #mask used to calculate indexes for assigning the output
    all_ind_mask = torch.ones(input.shape[0], dtype=torch.bool)
    num_1_plus4_mask = (num_1 > 4) 
    num_1_equal4_mask = (num_1 == 4)   
    data0_equal0_mask = (input[:,0] == 0) 
    intersect_ind_mask = (data0_equal0_mask & num_1_equal4_mask) 
    intersect_ind_mask = (intersect_ind_mask | num_1_plus4_mask)

    # indexes where (input[0] == 0 and num_1 == 4) or (num_1 > 4)  
    intersect_ind = intersect_ind_mask.nonzero().squeeze()
    # indexes that are not included in the ones from before  
    rest_ind = (intersect_ind_mask != all_ind_mask).nonzero().squeeze()

    #Output assignments using the calculated indexes
    output[intersect_ind,:8] = q_m_XNOR_diff(input[intersect_ind,:])
    output[intersect_ind,8] = 0
    output[rest_ind,:8] = q_m_XOR_diff(input[rest_ind,:])
    output[rest_ind,8] = 1

    return output


def TMDS_diff(pixel_column_bits,cnt_column):

    """ 
    Using q_m_diff and some extra calculation this function implements the TMDS codification algorithm of HDMI 
    but it's differentiable.
	
    Inputs:	
    
        -pixel_column_bits: 2-D Nx8 tensor that contains all the pixels values from an image column to codificate in 8 bits (float)
        -cnt_column: 1-D Nx1 tensor that computes the unbalance of 1s and 0s in the codification of pixels (float). It's all 0 at the start of a line of an image.

	Outputs: 
    
        -output: 2-D tensor (Nx10) that contains the codification of all the pixels of the column of an image (float)
        -new_cnt: 1-D tensor (Nx1) the unbalance of 1s and 0s used to codificate the next column of an image
	"""

    #flip the bits due to TMDS sending LSB first for all the pixel bits of the input
    inverse_bits = torch.flip(pixel_column_bits, dims = (1,))
    #calculate intermediate output that will be used to calculate final output of the codification
    q_m = q_m_diff(inverse_bits)
    #calculation of the complementary intermediate output
    Neg_q = 1 - q_m
    #output of the codification definition (10 bits wide per pixel)
    output = torch.zeros((pixel_column_bits.shape[0],10),dtype=torch.float32)
    #number of 0s and 1s of the intermediate output in the first 8 bits calculated for all of the column
    num_1 = (q_m[:,:8] > 0.5).sum(dim = 1)
    num_0 = (q_m[:,:8] < 0.5).sum(dim = 1)
    
    #masks used to calculate indexes for assignations in the final output
    all_ind_mask = torch.ones(q_m.shape[0], dtype=torch.bool)
    IndE_mask = (cnt_column == 0) | (num_1 == num_0)
    IndE_ind = ((cnt_column == 0) | (num_1 == num_0)).nonzero().squeeze()
    IndC_mask = ((cnt_column > 0) & (num_1 > num_0)) | ((cnt_column < 0) & (num_0 > num_1))
    IndE_and_q_m_0 = ((IndE_mask) & (q_m[:,8] < 0.5)).nonzero().squeeze()
    IndE_and_q_m_1 = ((IndE_mask) & (q_m[:,8] > 0.5)).nonzero().squeeze()
    NotIndE_and_IndC = (torch.logical_not(IndE_mask) & IndC_mask).nonzero().squeeze()
    NotIndE_and_NotIndC = (torch.logical_not(IndE_mask) & torch.logical_not(IndC_mask)).nonzero().squeeze()
    q_m_mask = (IndE_mask & (q_m[:,8] > 0.5)) | (torch.logical_not(IndC_mask) & torch.logical_not(IndE_mask))
    q_m_ind = ((IndE_mask & (q_m[:,8] > 0.5)) | (torch.logical_not(IndC_mask) & torch.logical_not(IndE_mask))).nonzero().squeeze()
    Neg_q_ind = (q_m_mask != all_ind_mask).nonzero().squeeze()
    
    #assignation of the final output and new_cnt to use for the codification of the next image column on a future call of the TMDS_diff function
    output[q_m_ind,:8] = q_m[q_m_ind,:8]
    output[Neg_q_ind,:8] = Neg_q[Neg_q_ind,:8]
    output[:,8] = q_m[:,8]
    output[IndE_ind,9] = Neg_q[IndE_ind,8]
    output[NotIndE_and_IndC,9] = 1
    output[NotIndE_and_NotIndC,9] = 0
    new_cnt = cnt_column.clone()
    new_cnt[IndE_and_q_m_0] = cnt_column[IndE_and_q_m_0] + num_0[IndE_and_q_m_0] - num_1[IndE_and_q_m_0]
    new_cnt[IndE_and_q_m_1] = cnt_column[IndE_and_q_m_1] + num_1[IndE_and_q_m_1] - num_0[IndE_and_q_m_1]
    new_cnt[NotIndE_and_IndC] = cnt_column[NotIndE_and_IndC] + 2 * q_m[NotIndE_and_IndC,8] + num_0[NotIndE_and_IndC] - num_1[NotIndE_and_IndC]
    new_cnt[NotIndE_and_NotIndC] = cnt_column[NotIndE_and_NotIndC] - 2 * Neg_q[NotIndE_and_NotIndC,8] + num_1[NotIndE_and_NotIndC] - num_0[NotIndE_and_NotIndC]

    return output,new_cnt

def sigmoid(x):
    output = torch.zeros_like(x)
    ind_greater_0 = (x >= 0).nonzero().squeeze()
    ind_smaller_0 = (x < 0).nonzero().squeeze()
    output[ind_greater_0] = 1 / (1 + torch.exp(-x[ind_greater_0]))
    output[ind_smaller_0] = torch.exp(x[ind_smaller_0]) / (1 + torch.exp(x[ind_smaller_0]))

    return output

def Pixel2Bit_diff(pixel):
    """ 
        This functions takes a pixel (0-255) in float and calculates the 8 bit codification of it differetiably using a sigmoid
    Inputs:	

        -pixel: 2-D Nx1 tensor that contains all the pixels values from an image column to codificate in 8 bits (float)


	Outputs: 
    
        -output: 2-D tensor (Nx8) that contains the 8 bit form of all the pixels of the column of an image (float)
	"""
    pixel_aux = pixel.clone()
    output = torch.zeros((pixel.shape[0],8), dtype= torch.float32)
    for i in range(1,9):
        output[:,i-1] = sigmoid(10*(pixel_aux-2**(8-i)+0.5))
        ind_pixel_greater2Pow = (pixel_aux >= 2**(8-i)).nonzero().squeeze()
        pixel_aux[ind_pixel_greater2Pow] = pixel_aux[ind_pixel_greater2Pow] - 2**(8-i)
    return output


def line_degradation(I_line, h_total=1800, v_total=1000, N_harmonic=3, sdr_rate = 50e6, fps=60):
    """  
    Given an image line bitstream (fs=bitrate), it performs the capture effect of HDMI with
    the pair antenna-SDR (using the specified SDR sample rate, resolution setup and pixel frequency harmonic)
    Returns the captured signal at SDR's sample rate and horizontal pixel resolution

    Inputs:
    I_line (torch tensor): input image bitstream line to perform capture degradation
    h_total (int): horizontal resolution (pixels)
    v_total (int): vertical resolution (pixels)
    N_harmonic (int): number of pixel frequency harmonic
    sdr_rate (float): sampling rate of SDR

    Output:
    I_line (complex torch tensor):  degradeted image bitstream line with SDR resampling with 
                                    horizontal image resolution
    """
    I_line_complex = I_line.clone()

    # Compute pixelrate and bitrate
    px_rate = h_total*v_total*fps
    bit_rate = 10*px_rate

    # Continuous samples (interpolate)
    interpolator = int(np.ceil(N_harmonic/5)) # Condition for sampling rate and
    sample_rate = interpolator*bit_rate
    Nsamples = 10*h_total
        
    # Continuous time array
    t_continuous = torch.arange(Nsamples)/sample_rate

    # AM modulation frequency according to pixel harmonic
    harm = N_harmonic*px_rate

    # Harmonic oscilator (including frequency and phase error)
    baseband_exponential = torch.exp(2j*np.pi*harm*t_continuous)
    # Baseband representation
    I_line_baseband = I_line_complex * baseband_exponential

    # AM modulation and SDR sampling
    resampler = transforms.Resample(sample_rate, sdr_rate)

    # Reshape signal to the image size
    I_line_out = nn.functional.interpolate(resampler(torch.real(I_line_baseband)),h_total) + 1j*nn.functional.interpolate(resampler(torch.imag(I_line_baseband)),h_total)
    

    return I_line_out

def forward(img):

    """ 
        This functions takes and image and implements the model of degradation used for HDMI codification, bit by bit transmition by using pulses, reception of certain
        harmonic with a SDR (sampling,baseband modulation and LPF). Currently no AWGN is added.
    Inputs:	

        -img: rows x columns image tensor (float)


	Outputs: 
    
        -img_out: rows x columns image tensor (complex) with the degradation
	"""
    #constants definitions
    rows = img.shape[0]
    columns = img.shape[1]
    #cnt and pixels columns to use for TMDS cod
    cnt_column = torch.zeros(rows)
    pixels_column = torch.zeros(rows)
    pixel_column_bits = torch.zeros((rows,8),dtype = torch.float32)
    bits_cod_column =  torch.zeros((rows,10), dtype = torch.float32)
    #codification of all the image
    img_cod = torch.zeros((rows,columns*10),dtype = torch.complex64)
    #output image definition (complex)
    img_out = torch.zeros((rows,columns),dtype=torch.complex64)
    #TMDS codification of all the image doing it by columns
    for j in range(columns):
        pixels_column = img[:,j]
        pixel_column_bits = Pixel2Bit_diff(pixels_column)
        bits_cod_column,cnt_column = TMDS_diff(pixel_column_bits, cnt_column)
        img_cod[:,j*10:(j+1)*10] = bits_cod_column
    #apply the degradation to the TMDS bits by rows
    for i in range(rows):
        img_out[i,:] = line_degradation(img_cod[i,:].unsqueeze(0).unsqueeze(0), h_total=columns, v_total=rows, N_harmonic=3, sdr_rate = 50e6, fps=60)
    return img_out

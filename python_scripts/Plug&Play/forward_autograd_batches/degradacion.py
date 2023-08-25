import numpy as np
import torch
import torch.nn as nn
from torchaudio import transforms
from matplotlib import pyplot as plt
from PIL import Image


# Definir una red neuronal simple
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
    
q_m_XOR = Model_qm()
state_dict_XOR = torch.load('forward_autograd/q_m_XOR.pth')
q_m_XOR.load_state_dict(state_dict_XOR)
#input must be tensor with float (each bit)
def q_m_XOR_diff(bits):
    return q_m_XOR(bits)

q_m_XNOR = Model_qm()
state_dict_XNOR = torch.load('forward_autograd/q_m_XNOR.pth')
q_m_XNOR.load_state_dict(state_dict_XNOR)
#input must be tensor with float (each bit)
def q_m_XNOR_diff(bits):
    return q_m_XNOR(bits)

def q_m_diff(input):
    output = torch.zeros((input.shape[0],9), dtype=torch.float32)
    all_ind_mask = torch.ones(input.shape[0], dtype=torch.bool)
    num_1 = (input > 0.5).sum(dim=1)
    #print(f"num_1:{num_1}")
    num_1_plus4_mask = (num_1 > 4) # mask donde la suma de 1 es mayor a 4
    #print(f"num_1_plus4_mask: {num_1_plus4_mask}")
    num_1_equal4_mask = (num_1 == 4) # mask donde la suma de 1 es 4
    #print(f"num_1_equal4_mask:{num_1_equal4_mask}")                      
    data0_equal0_mask = (input[:,0] == 0) # mask donde el primer valor de la entrada es 0
    #print(f"data0_equal0_mask:{data0_equal0_mask}")  
    intersect_ind_mask = (data0_equal0_mask & num_1_equal4_mask) # mask donde el primer valor de la entrada es 0 y la suma de 1 es 4
    #print(f"intersect_ind_mask 1:{intersect_ind_mask}")   
    intersect_ind_mask = (intersect_ind_mask | num_1_plus4_mask)  # mask donde: (el primer valor de la entrada es 0 y la suma de 1 es 4) OR (la suma es mayor a 4)
    #print(f"intersect_ind_mask 2:{intersect_ind_mask}")   
    intersect_ind = intersect_ind_mask.nonzero().squeeze()  # posiciones donde: (el primer valor de la entrada es 0 y la suma de 1 es 4) OR (la suma es mayor a 4)
    #print(f"intersect_ind:{intersect_ind}")   
    rest_ind = (intersect_ind_mask != all_ind_mask).nonzero().squeeze() #resto de indices que no cumplen condiciones anteriores
    #print(f"rest_ind:{rest_ind}")

    output[intersect_ind,:8] = q_m_XNOR_diff(input[intersect_ind,:])
    output[intersect_ind,8] = 0
    output[rest_ind,:8] = q_m_XOR_diff(input[rest_ind,:])
    output[rest_ind,8] = 1

    return output

#input must be float (each bit) example: [0.0 1.0 1.0 1.0 0.0 0.0 0.0 1.0], 2.0
#output are tensors
def TMDS_diff(pixel_column_bits,cnt_column):
    bits_inversos = torch.flip(pixel_column_bits, dims = (1,))  #ahora es en la dim = 1 y no 0
    q_m = q_m_diff(bits_inversos)
    #print(f"qm_diff:{torch.round(q_m)}")
    output = torch.zeros((pixel_column_bits.shape[0],10),dtype=torch.float32)
    num_1 = (q_m[:,:8] > 0.5).sum(dim = 1)
    num_0 = (q_m[:,:8] < 0.5).sum(dim = 1)
    #print(f"num1:{num_1}")
    #print(f"num0:{num_0}")
    all_ind_mask = torch.ones(q_m.shape[0], dtype=torch.bool)
    IndE_mask = (cnt_column == 0) | (num_1 == num_0)
    #print(f"IndE_mask:{IndE_mask}")
    IndE_ind = ((cnt_column == 0) | (num_1 == num_0)).nonzero().squeeze()
    #print(f"IndE_ind:{IndE_ind}")
    IndC_mask = ((cnt_column > 0) & (num_1 > num_0)) | ((cnt_column < 0) & (num_0 > num_1))
    #print(f"IndC_mask:{IndC_mask}")
    Neg_q = 1 - q_m
    #print(f"Neg_qm_diff:{torch.round(Neg_q)}")
    IndE_and_q_m_0 = ((IndE_mask) & (q_m[:,8] < 0.5)).nonzero().squeeze()
    #print(f"IndE_and_q_m_0:{IndE_and_q_m_0}")
    IndE_and_q_m_1 = ((IndE_mask) & (q_m[:,8] > 0.5)).nonzero().squeeze()
    #print(f"IndE_and_q_m_1:{IndE_and_q_m_1}")
    NotIndE_and_IndC = (torch.logical_not(IndE_mask) & IndC_mask).nonzero().squeeze()
    #print(f"NotIndE_and_IndC:{NotIndE_and_IndC}")
    NotIndE_and_NotIndC = (torch.logical_not(IndE_mask) & torch.logical_not(IndC_mask)).nonzero().squeeze()
    #print(f"NotIndE_and_NotIndC:{NotIndE_and_NotIndC}")
    q_m_mask = (IndE_mask & (q_m[:,8] > 0.5)) | (torch.logical_not(IndC_mask) & torch.logical_not(IndE_mask))
    #print(f"q_m_mask:{q_m_mask}")
    q_m_ind = ((IndE_mask & (q_m[:,8] > 0.5)) | (torch.logical_not(IndC_mask) & torch.logical_not(IndE_mask))).nonzero().squeeze()
    #print(f"q_m_ind:{q_m_ind}")
    Neg_q_ind = (q_m_mask != all_ind_mask).nonzero().squeeze()
    #print(f"Neg_q_ind:{Neg_q_ind}")
    output[q_m_ind,:8] = q_m[q_m_ind,:8]
    output[Neg_q_ind,:8] = Neg_q[Neg_q_ind,:8]
    output[:,8] = q_m[:,8]
    output[IndE_ind,9] = Neg_q[IndE_ind,8]
    new_cnt = cnt_column.clone()
    new_cnt[IndE_and_q_m_0] = cnt_column[IndE_and_q_m_0] + num_0[IndE_and_q_m_0] - num_1[IndE_and_q_m_0]
    new_cnt[IndE_and_q_m_1] = cnt_column[IndE_and_q_m_1] + num_1[IndE_and_q_m_1] - num_0[IndE_and_q_m_1]
    #print(f"new_cnt:{new_cnt}")
    output[NotIndE_and_IndC,9] = 1
    new_cnt[NotIndE_and_IndC] = cnt_column[NotIndE_and_IndC] + 2 * q_m[NotIndE_and_IndC,8] + num_0[NotIndE_and_IndC] - num_1[NotIndE_and_IndC]
    #print(f"new_cnt:{new_cnt}")
    output[NotIndE_and_NotIndC,9] = 0
    new_cnt[NotIndE_and_NotIndC] = cnt_column[NotIndE_and_NotIndC] - 2 * Neg_q[NotIndE_and_NotIndC,8] + num_1[NotIndE_and_NotIndC] - num_0[NotIndE_and_NotIndC]
    #print(f"new_cnt:{new_cnt}")

    return output,new_cnt

def sigmoid(x):
    output = torch.zeros_like(x)
    ind_greater_0 = (x >= 0).nonzero().squeeze()
    ind_smaller_0 = (x < 0).nonzero().squeeze()
    output[ind_greater_0] = 1 / (1 + torch.exp(-x[ind_greater_0]))
    output[ind_smaller_0] = torch.exp(x[ind_smaller_0]) / (1 + torch.exp(x[ind_smaller_0]))

    return output

def Pixel2Bit_diff(pixel):
    pixel_aux = pixel.clone()
    output = torch.zeros((pixel.shape[0],8), dtype= torch.float32)
    for i in range(1,9):
        output[:,i-1] = sigmoid(10*(pixel_aux-2**(8-i)+0.5))  # 0.5 para ajustar la sigmoidal
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
    rows = img.shape[0]
    columns = img.shape[1]
    cnt_column = torch.zeros(rows)
    pixels_column = torch.zeros(rows)
    pixel_column_bits = torch.zeros((rows,8),dtype = torch.float32)
    bits_cod_column =  torch.zeros((rows,10), dtype = torch.float32)
    img_cod = torch.zeros((rows,columns*10),dtype = torch.complex64)
    img_out = torch.zeros((rows,columns),dtype=torch.complex64)
    for j in range(columns):
        pixels_column = img[:,j]
        pixel_column_bits = Pixel2Bit_diff(pixels_column)
        bits_cod_column,cnt_column = TMDS_diff(pixel_column_bits, cnt_column)
        img_cod[:,j*10:(j+1)*10] = bits_cod_column
    for i in range(rows):
        img_out[i,:] = line_degradation(img_cod[i,:].unsqueeze(0).unsqueeze(0), h_total=columns, v_total=rows, N_harmonic=3, sdr_rate = 50e6, fps=60)
    return img_out

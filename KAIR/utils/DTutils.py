import numpy as np
from scipy import signal
from numba import jit, uint8, int8, prange

def rgb_entropy(I):
    """ 
    Calculates image entropy with unweighted sum of histogram RGB channels

    Inputs:
    - I (3D array): RGB image

    Returns
    - entropy (float): image color entropy
    """
    # Number of grey-scale bins
    hist_size = 256
    
    # Channels histograms
    hist_R, _ = np.histogram(I[:,:,0], bins=hist_size)
    hist_G, _ = np.histogram(I[:,:,1], bins=hist_size)
    hist_B, _ = np.histogram(I[:,:,2], bins=hist_size)

    # Histogram sum and normalize
    hist_chs_sum = hist_R + hist_G + hist_B
    hist_chs_sum = hist_chs_sum/np.sum(hist_chs_sum)
    
    # Find entropy
    entropy = 0
    for i in range (0,256):
        if hist_chs_sum[i] != 0:
            entropy -= (hist_chs_sum[i]/hist_size)*np.log(hist_chs_sum[i]/hist_size)

    return entropy

def is_natural_patch(patch, entropy_thrs = 0.035):
  """  
  Label an image patch as natural or artifitial based on it's color entropy

  Inputs:
  - patch (3D array): RGB image patch
  - entropy_thrs (float, default 0.035): threshold that decides if its natural or not

  Returns:
  - is_natural (bool): True if entropy of patch is above threshold, False otherwise
  """

  entropy = rgb_entropy(patch)

  is_natural = entropy > entropy_thrs

  return is_natural

def autocorr(x):
	"""Compute autocorrelation function of 1-D array

	Input:
	x:	1-D array

	Output:
	autocorr:	autocorrelation function of x
	"""

	# Use FFT method, which has more computing efectiveness for 1-D numpy arrays
	autocorr = signal.correlate(x,x,mode='full', method= 'fft')

	# Fix some shifts due FFT
	half_idx =int(autocorr.size/2)
	max_ind = np.argmax(autocorr[half_idx:])+half_idx
	autocorr = autocorr[max_ind:]
	# Normalise output
	return autocorr/autocorr[0]


def uint8_to_binarray(integer):
  """Convert integer into fixed-length 8-bit binary array. LSB in [0].
  Extended and modified code from https://github.com/projf/display_controller/blob/master/model/tmds.py
  """

  b_array = [int(i) for i in reversed(bin(integer)[2:])]
  b_array += [0]*(8-len(b_array))
  return b_array

def uint16_to_binarray(integer):
  """Convert integer into fixed-length 16-bit binary array. LSB in [0].
  Extended and modified code from https://github.com/projf/display_controller/blob/master/model/tmds.py
  """
  b_array = [int(i) for i in reversed(bin(integer)[2:])]
  b_array += [0]*(16-len(b_array))
  return b_array

def binarray_to_uint(binarray):
	
  array = binarray[::-1]
  num = array[0]
  for n in range(1,len(binarray)):
    num = (num << 1) + array[n]

  return num

def TMDS_pixel_rare (pix):
  """8bit pixel TMDS coding

  Inputs: 
  - pix: 8-bit pixel
  - cnt: 0's and 1's balance. Default in 0 (balanced)

  Outputs:
  - pix_out: TDMS coded 16-bit pixel (only 10 useful)
  - cnt: 0's and 1's balance updated with new pixel coding

  """ 
  # Convert 8-bit pixel to binary list D
  D = uint8_to_binarray(pix)

  # Initialize output q
  qm = [D[0]]

  # 1's unbalanced condition at current pixel
  N1_D = np.sum(D)

  if N1_D>4 or (N1_D==4 and not(D[0])):

    # XNOR of consecutive bits
    for k in range(1,8):
      qm.append( not(qm[k-1] ^ D[k]) )
    qm.append(0)

  else:
    # XOR of consecutive bits
    for k in range(1,8):
      qm.append( qm[k-1] ^ D[k] )
    qm.append(1)

  qm.append(np.random.choice([0,1]))

  

  # Return the TMDS coded pixel as uint and 0's y 1's balance
  return binarray_to_uint(qm)

@jit(nopython=True)
def TMDS_pixel_numba(pix:uint8, cnt:int8)->tuple:
    
  D = np.zeros(8, dtype=np.uint8)
  for i in range(8):
      D[i] = (pix >> i) & 1

  qm = np.zeros(9, dtype=np.uint8)
  qm[0] = D[0]

  N1_D = np.sum(D)

  if N1_D > 4 or (N1_D == 4 and not D[0]):
      for k in range(1, 8):
          qm[k] = not (qm[k-1] ^ D[k])
      qm[8] = 0
  else:
      for k in range(1, 8):
          qm[k] = qm[k-1] ^ D[k]
      qm[8] = 1

  qout = np.zeros(10, dtype=np.uint8)
  N1_qm = np.sum(qm[:8])
  N0_qm = 8 - N1_qm

  if cnt == 0 or N1_qm == 4:
      
      qout[9] = not(qm[8])
      qout[8] = qm[8]
      if qm[8]:
        qout[:8] = qm[:8]  
      else: 
        qout[:8] = np.logical_not(qm[:8])

      if not qm[8]:
          cnt += N0_qm - N1_qm
      else:
          cnt += N1_qm - N0_qm

  else:
      
      if (cnt > 0 and N1_qm > 4) or (cnt < 0 and N1_qm < 4):
          qout[9] = 1
          qout[8] = qm[8]
          qout[:8] = np.logical_not(qm[:8])
          cnt += 2*qm[8] + N0_qm - N1_qm
      else:
          qout[9] = 0
          qout[8] = qm[8]
          qout[:8] = qm[:8]
          cnt += -2*(not(qm[8])) + N1_qm - N0_qm

  # Convert binary array to unsigned int
  pix_tmds = 0
  for bit in qout[::-1]:
      pix_tmds = (pix_tmds << 1) | bit

  # Return the TMDS coded pixel as uint and 0's y 1's balance
  return pix_tmds, cnt


def TMDS_pixel (pix,cnt=0):
  """8bit pixel TMDS coding

  Inputs: 
  - pix: 8-bit pixel
  - cnt: 0's and 1's balance. Default in 0 (balanced)

  Outputs:
  - pix_out: TDMS coded 16-bit pixel (only 10 useful)
  - cnt: 0's and 1's balance updated with new pixel coding

  """ 
  # Convert 8-bit pixel to binary list D
  D = uint8_to_binarray(pix)

  # Initialize output q
  qm = [D[0]]

  # 1's unbalanced condition at current pixel
  N1_D = np.sum(D)

  if N1_D>4 or (N1_D==4 and not(D[0])):

    # XNOR of consecutive bits
    for k in range(1,8):
      qm.append( not(qm[k-1] ^ D[k]) )
    qm.append(0)

  else:
    # XOR of consecutive bits
    for k in range(1,8):
      qm.append( qm[k-1] ^ D[k] )
    qm.append(1)

  # Initialize output qout
  qout = qm.copy()

  # Unbalanced condition with previous and current pixels
  N1_qm = np.sum(qm[:8])
  N0_qm = 8 - N1_qm

  if cnt==0 or N1_qm==4:

    qout.append(not(qm[8]))
    qout[8] = qm[8]
    qout[:8]=qm[:8] if qm[8] else np.logical_not(qm[:8])

    if not(qm[8]):
      cnt += N0_qm - N1_qm 
    else:
      cnt += N1_qm - N0_qm 

  else:

    if (cnt>0 and N1_qm>4) or (cnt<0 and N1_qm<4):
      qout.append(1)
      qout[8] = qm[8]
      qout[:8] = np.logical_not(qm[:8])
      cnt += 2*qm[8] + N0_qm - N1_qm
    else:
      qout.append(0)
      qout[8] = qm[8]
      qout[:8] = qm[:8]
      cnt += -2*(not(qm[8])) + N1_qm - N0_qm

  # Return the TMDS coded pixel as uint and 0's y 1's balance
  return binarray_to_uint(qout), cnt

@jit(parallel=True)
def TMDS_encoding_original (I, blanking = False):
  """TMDS image coding

  Inputs: 
  - I: 2-D image array
  - blanking: Boolean that specifies if horizontal and vertical blanking is applied

  Output:
  - I_c: TDMS coded 16-bit image (only 10 useful)

  """ 

  # Create "ghost dimension" if I is gray-scale image (not RGB)
  if len(I.shape)!= 3:
    I = I[:, :, np.newaxis]
    chs = 1
  else:    
    chs = 3

  # Get image resolution
  v_in, h_in = I.shape[:2]
  
  if blanking:
    # Get blanking resolution for input image
    
    v = (v_in==1080)*1125 + (v_in==720)*750   + (v_in==600)*628  + (v_in==480)*525
    h = (h_in==1920)*2200 + (h_in==1280)*1650 + (h_in==800)*1056 + (h_in==640)*800 

    vdiff = v - v_in
    hdiff = h - h_in

    # Create image with blanking and change type to uint16
    # Assuming the blanking corresponds to 10bit number [0, 0, 1, 0, 1, 0, 1, 0, 1, 1] (LSB first)
    I_c = 852*np.ones((v,h,chs)).astype('uint16')
    
  else:
    v_diff = 0
    h_diff = 0
    I_c = np.zeros((v_in,h_in,chs)).astype('uint16')

  # Iterate over channels and pixels
  for c in prange(chs):
    for i in range(v_in):
      cnt=[0,0,0]
      for j in range(h_in):
        # Get pixel and code it TMDS between blanking
        pix = I[i,j,c]
        I_c[i + v_diff//2 , j + h_diff//2, c], cnt[c] = TMDS_pixel_numba (pix,cnt[c])

  return I_c

def TMDS_pixel_cntdiff (pix,cnt=0):
  """8bit pixel TMDS coding

  Inputs: 
  - pix: 8-bit pixel
  - cnt: 0's and 1's balance. Default in 0 (balanced)

  Outputs:
  - pix_out: TDMS coded 16-bit pixel (only 10 useful)
  - cntdiff: balance difference given by the actual coded pixel

  """ 
  # Convert 8-bit pixel to binary list D
  D = uint8_to_binarray(pix)

  # Initialize output q
  qm = [D[0]]

  # 1's unbalanced condition at current pixelo
  N1_D = np.sum(D)

  if N1_D>4 or (N1_D==4 and not(D[0])):

    # XNOR of consecutive bits
    for k in range(1,8):
      qm.append( not(qm[k-1] ^ D[k]) )
    qm.append(0)

  else:
    # XOR of consecutive bits
    for k in range(1,8):
      qm.append( qm[k-1] ^ D[k] )
    qm.append(1)

  # Initialize output qout
  qout = qm.copy()

  # Unbalanced condition with previous and current pixels
  N1_qm = np.sum(qm[:8])
  N0_qm = 8 - N1_qm

  if cnt==0 or N1_qm==4:

    qout.append(not(qm[8]))
    qout[8]=qm[8]
    qout[:8]=qm[:8] if qm[8] else [not(val) for val in qm[:8]]

    if not(qm[8]):
      cnt_diff = N0_qm - N1_qm 
    else:
      cnt_diff = N1_qm - N0_qm 

  else:

    if (cnt>0 and N1_qm>4) or (cnt<0 and N1_qm<4):
      qout.append(1)
      qout[8]=qm[8]
      qout[:8] = [not(val) for val in qm[:8]]
      cnt_diff = 2*qm[8] +N0_qm -N1_qm
    else:
      qout.append(0)
      qout[8]=qm[8]
      qout[:8] = qm[:8]
      cnt_diff = -2*(not(qm[8])) + N1_qm - N0_qm

  # Return the TMDS coded pixel as uint and 0's y 1's balance difference
  uint_out = binarray_to_uint(qout)
  return uint_out, cnt_diff


### Create TMDS LookUp Tables for fast encoding (3 times faster than the other implementation)
byte_range = np.arange(256)
# Initialize pixel coding and cnt-difference arrays
TMDS_pix_table = np.zeros((256,3),dtype='uint16')
TMDS_rare_pix_table = np.zeros((256),dtype='uint16')
TMDS_cntdiff_table = np.zeros((256,3),dtype='int8')

for byte in byte_range:
  p0,p_null, p1 = TMDS_pixel_cntdiff(byte,-1),TMDS_pixel_cntdiff(byte,0),TMDS_pixel_cntdiff(byte,1) # 0's and 1's unbalance respect.
  TMDS_pix_table[byte,0] = p0[0]
  TMDS_pix_table[byte,1] = p_null[0]
  TMDS_pix_table[byte,2] = p1[0]
  TMDS_cntdiff_table[byte,0] = p0[1]
  TMDS_cntdiff_table[byte,1] = p_null[1]
  TMDS_cntdiff_table[byte,2] = p1[1]

  TMDS_rare_pix_table[byte] = TMDS_pixel_rare(byte)

def pixel_fastencoding(pix,cnt_prev=0):
  """8bit pixel TMDS fast coding

  Inputs: 
  - pix: 8-bit pixel
  - cnt: 0's and 1's balance. Default in 0 (balanced)

  Outputs:
  - pix_out: TDMS coded 16-bit pixel (only 10 useful)
  - cnt: 0's and 1's balance updated with new pixel coding

  """ 
  balance_idx = int(np.sign(cnt_prev))+1
  pix_out = TMDS_pix_table[pix,balance_idx]
  cnt     = cnt_prev + TMDS_cntdiff_table[pix,balance_idx]

  return  pix_out, cnt

def TMDS_blanking (h_total, v_total, h_active, v_active, h_front_porch, v_front_porch, h_back_porch, v_back_porch):
  
  # Initialize blanking image
  img_blank = np.zeros((v_total,h_total))

  # Get the total blanking on vertical an horizontal axis
  h_blank = h_total - h_active
  v_blank = v_total - v_active
  
  # (C1,C0)=(0,0) region
  img_blank[:v_front_porch,:h_front_porch] = 0b1101010100
  img_blank[:v_front_porch,h_blank-h_back_porch:] = 0b1101010100
  img_blank[v_blank-v_back_porch:v_blank,:h_front_porch] = 0b1101010100
  img_blank[v_blank-v_back_porch:v_blank,h_blank-h_back_porch:] = 0b1101010100
  img_blank[v_blank:,:h_blank] = 0b1101010100

  # (C1,C0)=(0,1) region
  img_blank[:v_front_porch,h_front_porch:h_blank-h_back_porch] = 0b0010101011
  img_blank[v_blank-v_back_porch:,h_front_porch:h_blank-h_back_porch] = 0b0010101011

  # (C1,C0)=(1,0) region
  img_blank[v_front_porch:v_blank-v_back_porch,:h_front_porch] = 0b0101010100
  img_blank[v_front_porch:v_blank-v_back_porch,h_blank-h_back_porch:] = 0b0101010100

  # (C1,C0)=(1,1) region
  img_blank[v_front_porch:v_blank-v_back_porch,v_front_porch:h_blank-h_back_porch] = 0b1010101011

  return(img_blank)

def TMDS_encoding (I, blanking = False):
  """TMDS image coding

  Inputs: 
  - I: 2D/3D image array (v_size, h_size, channels)
  - blanking: Boolean that specifies if horizontal and vertical blanking is applied or not

  Output:
  - I_c: 3D TDMS coded 16-bit (only 10 useful) image array 

  """ 

  # Create "ghost dimension" if I is gray-scale image (not RGB)
  if len(I.shape)!= 3:
    I = I[:, :, np.newaxis]
    chs = 1
  else:    
    chs = 3

  # Get image resolution
  v_in, h_in = I.shape[:2]
  
  if blanking:
    # Get blanking resolution for input image
    
    v = (v_in==1080)*1125 + (v_in==900)*1000  + (v_in==720)*750   + (v_in==600)*628  + (v_in==480)*525
    h = (h_in==1920)*2200 + (h_in==1600)*1800 + (h_in==1280)*1650 + (h_in==800)*1056 + (h_in==640)*800 

    v_diff = v - v_in
    h_diff = h - h_in

    v_front_porch = (v_in==1080)*4 + (v_in==900)*1  + (v_in==720)*5   + (v_in==600)*1  + (v_in==480)*2
    v_back_porch = (v_in==1080)*36 + (v_in==900)*96  + (v_in==720)*20   + (v_in==600)*23  + (v_in==480)*25

    h_front_porch = (h_in==1920)*88 + (h_in==1600)*24 + (h_in==1280)*110 + (h_in==800)*40 + (h_in==640)*8 
    h_back_porch = (h_in==1920)*148 + (h_in==1600)*96 + (h_in==1280)*220 + (h_in==800)*88 + (h_in==640)*40 

    # Create image with blanking and change type to uint16
    # Assuming the blanking corresponds to 10bit number 
    # [0, 0, 1, 0, 1, 0, 1, 0, 1, 1] (LSB first) for channels R and G
    I_c = 852*np.ones((v,h,chs)).astype('uint16')
    if chs==3:
      I_c[:,:,2] = TMDS_blanking(h_total=h, v_total=v, h_active=h_in, v_active=v_in, 
                    h_front_porch=h_front_porch, v_front_porch=v_front_porch, h_back_porch=h_back_porch, v_back_porch=v_back_porch)
    
  else:
    v_diff = 0
    h_diff = 0
    I_c = np.zeros((v_in,h_in,chs)).astype('uint16')

  # Iterate over channels and pixels
  for c in range(chs):
    for i in range(v_in):
        cnt = [0,0,0]
        for j in range(h_in):
            # Get pixel and code it TMDS between blanking
            pix = I[i,j,c]
            I_c[i + v_diff, j + h_diff, c], cnt[c] = pixel_fastencoding (pix,cnt[c])

  return I_c

def TMDS_encoding_rare (I, blanking = False):
  """TMDS image coding

  Inputs: 
  - I: 2-D image array
  - blanking: Boolean that specifies if horizontal and vertical blanking is applied

  Output:
  - I_c: TDMS coded 16-bit image (only 10 useful)

  """ 

  # Create "ghost dimension" if I is gray-scale image (not RGB)
  if len(I.shape)!= 3:
    I = I[:, :, np.newaxis]
    chs = 1
  else:    
    chs = 3

  # Get image resolution
  v_in, h_in = I.shape[:2]

  # Get image resolution
  v_in, h_in = I.shape[:2]
  
  if blanking:
    # Get blanking resolution for input image
    
    v = (v_in==1080)*1125 + (v_in==720)*750   + (v_in==600)*628  + (v_in==480)*525
    h = (h_in==1920)*2200 + (h_in==1280)*1650 + (h_in==800)*1056 + (h_in==640)*800 

    vdiff = v - v_in
    hdiff = h - h_in

    # Create image with blanking and change type to uint16
    # Assuming the blanking corresponds to 10bit number [0, 0, 1, 0, 1, 0, 1, 0, 1, 1] (LSB first)

    I_c = 852*np.ones((v,h,chs)).astype('uint16')
    
  else:
    v_diff = 0
    h_diff = 0
    I_c = I.copy()
    I_c = I_c.astype('uint16')

  # Iterate over channels and pixels
  for c in range(chs):
    for i in range(v_in):
      for j in range(h_in):
        # Get pixel and code it TMDS between blanking
        pix = I[i,j,c]
        I_c[i + v_diff//2 , j + h_diff//2, c] = TMDS_rare_pix_table[pix]

  return I_c

def DecTMDS_pixel (pix):
  """10-bit pixel TMDS decoding

  Inputs: 
  - pix: 16-bit pixel (only 10 first bits useful)

  Output:
  - pix_out: 8-bit TMDS decoded pixel

  """ 


  D = uint16_to_binarray(pix)[:10]

  if D[9]:
    D[:8] = np.logical_not(D[:8])

  Q = D.copy()[:8]

  if D[8]:
    for k in range(1,8):
      Q[k] = D[k] ^ D[k-1]
  else:
    for k in range(1,8):
      Q[k] = not(D[k] ^ D[k-1])

  # Return pixel as uint
  return binarray_to_uint(Q)

def TMDS_decoding (Ic):
  """Image TMDS decoding

  Inputs: 
  - Ic: TMDS coded image

  Output:
  - Idec: 8-bit decoded image

  """ 

  # Create "ghost dimension" if gray-scale image (not RGB)
  if len(Ic.shape)!= 3:
    Ic = Ic.reshape(Ic.shape[0],Ic.shape[1],1)

  Idec = Ic.copy()
  # Get image dimensions
  Nx, Ny, Nz = Ic.shape

  # Iterate over channels and pixels
  for c in np.arange(Nz):
    for i in np.arange(Nx):
      for j in np.arange(Ny):

        # Get pixel and use TMDS decoding
        pix = Ic[i,j,c]
        Idec[i,j,c] = DecTMDS_pixel (pix)

  return Idec


def TMDS_serial(I):
  '''
  Serialize an image as an 1D binary array given a 10bit pixel value.

  Inputs: 
  - I: TMDS image to serialize. Pixel values must be between 0 and 1023

  Output:
  - Iserials: 1D binary array per image channel which represents 
              the voltage value to be transmitted

  '''
  assert np.min(I)>=0 and np.max(I)<= 1023, "Pixel values must be between 0 and 1023"

  # Initialize lists per channel
  Iserials = []
  n_rows,n_columns, n_channels = I.shape
  
  # Iterate over pixel
  for c in range(n_channels):
    channel_list = []
    for i in range(n_rows):
      for j in range(n_columns):
        # Get pixel value and cast it as binary string
        binstring = bin(I[i,j,c])[2:]
        # Fill string with 0's to get length 10
        binstring = '0'*(10-len(binstring))+binstring
        # Re-order string for LSB first
        binstring = binstring[::-1]
        binarray  = list(binstring)
        # Extend the bit stream
        channel_list.extend(binarray)

    Iserials.append(channel_list)

  # Digital to analog value mapping: [0,1]-->[-A,A] (A=1)
  Iserials = np.sum(2*np.array(Iserials,dtype='int32') - 1, axis=0)

  del(channel_list)

  return Iserials

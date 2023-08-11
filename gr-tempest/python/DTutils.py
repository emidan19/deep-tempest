import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import cv2 as cv
from scipy.spatial import distance_matrix

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
    I = np.repeat(I[:, :, np.newaxis], 3, axis=2).astype('uint8')
    
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
  for c in range(chs):
    for i in range(v_in):
      cnt=[0,0,0]
      for j in range(h_in):
        # Get pixel and code it TMDS between blanking
        pix = I[i,j,c]
        I_c[i + v_diff//2 , j + h_diff//2, c], cnt[c] = TMDS_pixel (pix,cnt[c])

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
TMDS_cntdiff_table = np.zeros((256,3),dtype='int8')

for byte in byte_range:
  p0,p_null, p1 = TMDS_pixel_cntdiff(byte,-1),TMDS_pixel_cntdiff(byte,0),TMDS_pixel_cntdiff(byte,1) # 0's and 1's unbalance respect.
  TMDS_pix_table[byte,0] = p0[0]
  TMDS_pix_table[byte,1] = p_null[0]
  TMDS_pix_table[byte,2] = p1[0]
  TMDS_cntdiff_table[byte,0] = p0[1]
  TMDS_cntdiff_table[byte,1] = p_null[1]
  TMDS_cntdiff_table[byte,2] = p1[1]

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
    # Gray-scale image
    I = np.repeat(I[:, :, np.newaxis], 3, axis=2).astype('uint8')
    chs = 1
  else:
    # RGB image
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
def remove_outliers(I, radius=3, threshold=20):
    """  
    Replaces a pixel by the median of the pixels in the surrounding if it deviates from the median by more than a certain value (the threshold).
    """

    # Copy input
    I_output = I.copy()

    # Apply median filter
    I_median = cv.medianBlur(I,radius)

    # Replace with median value where difference with median exceedes threshold
    where_replace = np.abs(I-I_median) > threshold
    I_output[where_replace] = I_median[where_replace]

    return I_output

def adjust_dynamic_range(I):

    I_output = I.astype('float32')
    I_output = (I_output - I_output.min()) / (I_output.max() - I_output.min())
    I_output = 255 * I_output
    return I_output.astype('uint8')

def find_intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    # Return as row-column coordinates
    return [y0, x0]
def apply_blanking_shift(I, h_active=1600, v_active=900, 
                         h_blanking=200,v_blanking=100, 
                         pad_len=300, debug=False):
    """  
    Find
    """
    
    if debug:
        # Show original image
        plt.figure(figsize=(12,10))
        plt.title(f'Original image ')
        plt.imshow(I)
        plt.axis('off')
        plt.show()

    # Color to RGB
    I_gray = cv.cvtColor(I,cv.COLOR_BGR2GRAY)
    # I_gray = cv.medianBlur(I_gray,3)
    I_gray = cv.GaussianBlur(I_gray,ksize=(5,5),sigmaX=3,sigmaY=3)
    # I_gray = cv.blur(I_gray,(3,3))

    # Wrap-padding image to get whole blanking pattern
    pad_size_gray = ((pad_len,0),(pad_len,0))
    pad_size = ((pad_len,0),(pad_len,0),(0,0))
    I_gray = np.pad(I_gray,pad_size_gray,'wrap')
    I = np.pad(I,pad_size,'wrap')

    # Edge-detector with Canny filter with empirical parameters
    I_edge = cv.Canny(I_gray,40,50,apertureSize = 3)    # sin blur
    

    if debug:
        # Show padded image and edges
        plt.figure()
        plt.title(f'Original image wrap-padded {pad_len} pixels up and left sided')
        plt.imshow(I)
        plt.axis('off')
        plt.show()
        plt.figure()
        plt.title('Canny edged image')
        plt.imshow(I_edge,cmap='gray')
        plt.axis('off')
        plt.show()

    # Hough Transform resolution and minimum votes threshold
    theta_step = np.pi/2
    rho_step = 1
    # votes_thrs = 255    # sin blur
    votes_thrs = 80
    rho_max = np.sqrt(I_gray.shape[0]**2 + I_gray.shape[1]**2)

    # Find Hough Transform
    lines = cv.HoughLines(I_edge, rho_step, theta_step, votes_thrs, None, 0, 0)

    if debug:
        # Show detected lines
        I_lines = I_gray.copy()
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + rho_max*(-b)), int(y0 + rho_max*(a)))
            pt2 = (int(x0 - rho_max*(-b)), int(y0 - rho_max*(a)))

            cv.line(I_lines, pt1, pt2, (255,0,0), 1, cv.LINE_AA)

        plt.figure()
        plt.title('All detected lines')
        plt.imshow(I_lines)
        plt.axis('off')
        plt.show()

    # Angle and rho arrays
    lines_angles = lines[:,0,1]
    lines_rhos = lines[:,0,0]

    # Find unique lines angles detected
    unique_angles = np.unique(lines_angles)
    blankings = [h_blanking, v_blanking]

    # Initiate blanking lines variable
    blanking_lines = []

    for angle, blanking in zip(unique_angles, blankings):
        
        # Keep lines with certain angle
        angle_lines = lines[lines[:,0,1]==angle]
        angle_lines = angle_lines[:,0,:]

        # Keep the pair with rho distance that equals blanking
        # First, compute the distance matrix over all rhos:
        rho_angle_lines = np.array(angle_lines[:,0]).reshape(-1,1)

        rho_distances = np.abs(np.abs(distance_matrix(rho_angle_lines, rho_angle_lines)) - blanking)

        # Find minimum index
        pair_lines_idx = np.unravel_index(rho_distances.argmin(), rho_distances.shape)

        # Add blanking limit line
        for line in pair_lines_idx:
            blanking_lines.append(angle_lines[line])

    # List to array
    blanking_lines = np.array(blanking_lines)

    # Show blanking limit lines
    if debug:    
        I_lines = I_gray.copy()

        for line in blanking_lines:
            rho = line[0]
            theta = line[1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + rho_max*(-b)), int(y0 + rho_max*(a)))
            pt2 = (int(x0 - rho_max*(-b)), int(y0 - rho_max*(a)))

            cv.line(I_lines, pt1, pt2, (255,0,0), 3, cv.LINE_AA)

        plt.figure(figsize=(12,10))
        plt.title('Blanking limit lines')
        plt.imshow(I_lines)
        plt.axis('off')
        plt.show()

    # Initialize top-left corner blanking lines to find intersection
    # These lines statisfies to be the ones with bigger rho value
    blanking_start = []
    unique_angles = np.unique(blanking_lines[:,1])
    for angulo in unique_angles:
        angle_lines = blanking_lines[blanking_lines[:,1]==angulo]
        max_rho_line = angle_lines[np.argmax(angle_lines[:,0])]
        blanking_start.append(max_rho_line)

    if len(blanking_start)!=2:
        return -1

    if debug:
        I_lines = I_gray.copy()
        for line in blanking_start:
            rho = line[0]
            theta = line[1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + rho_max*(-b)), int(y0 + rho_max*(a)))
            pt2 = (int(x0 - rho_max*(-b)), int(y0 - rho_max*(a)))

            cv.line(I_lines, pt1, pt2, (255,0,0), 3, cv.LINE_AA)
            
        plt.figure(figsize=(12,10))
        plt.title('Blanking start lines')
        plt.imshow(I_lines)
        plt.axis('off')
        plt.show()
    
    # Find blanking start lines intersection coordinates
    x_shift, y_shift = find_intersection(blanking_start[0],blanking_start[1])

    # Adjust to active image only (remove all blanking)
    I_shift = I[pad_len:,pad_len:]
    I_shift = np.roll(I_shift, -x_shift+pad_len, axis=0)
    I_shift = np.roll(I_shift, -y_shift+pad_len, axis=1)
    I_shift = I_shift[:v_active,:h_active]

    if debug:
        plt.figure(figsize=(12,10))
        plt.title('Blanking removed image')
        plt.imshow(I_shift)
        plt.axis('off')
        plt.show()
    
    return I_shift

def preprocess_raw_capture(I, h_active, v_active, 
                           h_blanking, v_blanking, debug=False):
    """  
    Center raw captured image, filter noise and adjust the contrast
    """

    # Center image. Returns -1 if no sufficient lines where detected
    # In the latter case, use image as is without centering
    is_centered = True
    I_shift_fix = apply_blanking_shift(I,
                                       h_active=h_active, v_active=v_active,
                                       h_blanking=h_blanking, v_blanking=v_blanking,
                                       debug=debug
                                       )
    if np.shape(I_shift_fix) == ():
        I_shift_fix = I.copy()
        is_centered = False
    
    # Remove outliers with median thresholding heuristic
    # Default: radius=3, threshold=20
    I_no_outliers = remove_outliers(I_shift_fix)

    # Stretch dynamic range to [0,255]
    I_out = adjust_dynamic_range(I_no_outliers)

    if debug:
        plt.figure(figsize=(12,10))
        ax0 = plt.subplot(3,1,1)
        ax0.imshow(I_shift_fix, interpolation='none')
        ax0.set_title('Centered image'*is_centered + 'Image'*(~is_centered))
        ax0.axis('off')
        ax1 = plt.subplot(3,1,2, sharex=ax0, sharey=ax0)
        ax1.imshow(I_no_outliers, interpolation='none')
        ax1.set_title('Outliers removed')
        ax1.axis('off')
        ax1 = plt.subplot(3,1,3, sharex=ax0, sharey=ax0)
        ax1.imshow(I_out, interpolation='none')
        ax1.set_title('Contrast adjusted')
        ax1.axis('off')
        plt.show()

    return I_out
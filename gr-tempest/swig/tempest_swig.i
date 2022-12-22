/* -*- c++ -*- */

#define TEMPEST_API

%include "gnuradio.i"           // the common stuff

//load generated python docstrings
%include "tempest_swig_doc.i"

%{
#include "tempest/sampling_synchronization.h"
#include "tempest/framing.h"
#include "tempest/Hsync.h"
#include "tempest/normalize_flow.h"
#include "tempest/fine_sampling_synchronization.h"
#include "tempest/sync_detector.h"
#include "tempest/frame_drop.h"
#include "tempest/fft_peak_fine_sampling_sync.h"
#include "tempest/infer_screen_resolution.h"
#include "tempest/ssamp_correction.h"
%}

%include "tempest/sampling_synchronization.h"
GR_SWIG_BLOCK_MAGIC2(tempest, sampling_synchronization);
%include "tempest/framing.h"
GR_SWIG_BLOCK_MAGIC2(tempest, framing);
%include "tempest/Hsync.h"
GR_SWIG_BLOCK_MAGIC2(tempest, Hsync);
%include "tempest/normalize_flow.h"
GR_SWIG_BLOCK_MAGIC2(tempest, normalize_flow);
%include "tempest/fine_sampling_synchronization.h"
GR_SWIG_BLOCK_MAGIC2(tempest, fine_sampling_synchronization);

%include "tempest/sync_detector.h"
GR_SWIG_BLOCK_MAGIC2(tempest, sync_detector);
%include "tempest/frame_drop.h"
GR_SWIG_BLOCK_MAGIC2(tempest, frame_drop);

%include "tempest/fft_peak_fine_sampling_sync.h"
GR_SWIG_BLOCK_MAGIC2(tempest, fft_peak_fine_sampling_sync);
%include "tempest/infer_screen_resolution.h"
GR_SWIG_BLOCK_MAGIC2(tempest, infer_screen_resolution);
%include "tempest/ssamp_correction.h"
GR_SWIG_BLOCK_MAGIC2(tempest, ssamp_correction);

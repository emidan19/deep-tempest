/* -*- c++ -*- */
/*
 * Copyright 2021 gr-tempest
 *    Pablo Bertrand    <pablo.bertrand@fing.edu.uy>
 *    Felipe Carrau     <felipe.carrau@fing.edu.uy>
 *    Victoria Severi   <maria.severi@fing.edu.uy>
 *   Adapted to 3.10 by Santiago Fernandez <santiago.fernandez.rovira@fing.edu.uy>
 
 *    
 *    Instituto de Ingeniería Eléctrica, Facultad de Ingeniería,
 *    Universidad de la República, Uruguay.
 * 
 * This software is based on Martin Marinov's TempestSDR.
 * In particular, gaussianblur() function is entirely his, and
 * find_best_beta and find_shift are implementations of
 * findbestfit and findthesweetspot, respectively.
 * 
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_SYNC_DETECTOR_IMPL_H
#define INCLUDED_SYNC_DETECTOR_IMPL_H

#include <gnuradio/tempest/sync_detector.h>

namespace gr {
  namespace tempest {

    class sync_detector_impl : public sync_detector
    {
     private:
      //Input data
      int d_hdisplay;
      int d_vdisplay;
      int d_hblanking;
      int d_vblanking;
      int d_Htotal;
      int d_Vtotal;

      //Blanking variables
      int d_blanking_size_h;
      int d_blanking_size_v;
      int d_blanking_index_h;
      int d_blanking_index_v;
      int d_working_index_h;

      //Flags
      int d_frame_average_complete;
      int d_frame_wait_for_blanking;
      int d_frame_output;


      //Counters
      int d_frame_height_counter; 
      int d_blanking_wait_counter; 
      int d_output_counter;

      // Control flag, and its mutex
      gr::thread::mutex d_mutex;
      bool d_start_sync_detect;   

      //Arrays
      float * d_data_h;
      float * d_avg_h_line;
      float * d_avg_v_line;

      //Functions
      void find_best_beta (const float *data, const int total_line_size, const double total_sum, const int blanking_size, double *beta, int *beta_index);
      void find_shift (int *blanking_index, int *blanking_size, float *data, const int total_line_size, int min_blanking_size, double lowpasscoeff);
      void gaussianblur(float * data, int size);
      float calculate_gauss_coeff(float N, float i);
      void set_ena_msg(pmt::pmt_t msg);

      //Fixed parameters
      float d_LOWPASS_COEFF_V;
      float d_LOWPASS_COEFF_H;
      float d_GAUSSIAN_ALPHA;

     public:
      sync_detector_impl(int hscreen, int vscreen, int hblanking, int vblanking);
      ~sync_detector_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

    };

  } // namespace sync_detector
} // namespace gr

#endif /* INCLUDED_SYNC_DETECTOR_IMPL_H */

/* -*- c++ -*- */
/**
 * Copyright 2021
 *    Pablo Bertrand    <pablo.bertrand@fing.edu.uy>
 *    Felipe Carrau     <felipe.carrau@fing.edu.uy>
 *    Victoria Severi   <maria.severi@fing.edu.uy>
 *   Adapted to 3.10 by Santiago Fernandez <santiago.fernandez.rovira@fing.edu.uy>
 *    
 *    Instituto de Ingeniería Eléctrica, Facultad de Ingeniería,
 *    Universidad de la República, Uruguay.
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
 * 
 * @file frame_drop_impl.h
 * 
 * @brief Block that uses the interpolation ratio to count
 * full frames in samples and discard some of them in order
 * to reduce the processing needed by other blocks to do the
 * interpolation.
 *
 * gr-tempest
 *
 * @date September 19, 2021
 * @author  Pablo Bertrand   <pablo.bertrand@fing.edu.uy>
 * @author  Felipe Carrau    <felipe.carrau@fing.edu.uy>
 * @author  Victoria Severi  <maria.severi@fing.edu.uy>
 */

/**********************************************************
 * Constant and macro definitions
 **********************************************************/

#ifndef INCLUDED_TEMPEST_FRAME_DROP_IMPL_H
#define INCLUDED_TEMPEST_FRAME_DROP_IMPL_H

/**********************************************************
 * Include statements
 **********************************************************/

#include <gnuradio/filter/mmse_fir_interpolator_cc.h>
#include <random>
#include <gnuradio/tempest/frame_drop.h>
#include <stdio.h>
#include <math.h>

namespace gr {
  namespace tempest {

    class frame_drop_impl : public frame_drop
    {
     private:

      /**********************************************************
       * Data declarations
       **********************************************************/

      //Counters
      int d_display_counter;
      int d_frames_counter;
      int d_sample_counter;
      
      //Fixed parameter
      int d_discarded_amount_per_frame;

      int d_Htotal; 
      int d_Vtotal; 
      int d_correct_sampling;
      uint32_t d_required_for_interpolation;
      float d_max_deviation;

      std::geometric_distribution<int> d_dist;
      std::minstd_rand d_gen;
      
      float d_proba_of_updating;
      int d_next_update;

      //double d_actual_samp_rate;
      double d_samp_inc_rem;
      double d_new_interpolation_ratio_rem;
      double d_samp_phase;

      float d_alpha_samp_inc; 
      int d_max_deviation_px; 
      int d_peak_line_index;

      float d_alpha_corr;
      double d_last_freq ;

      bool d_start_frame_drop;

      // the interpolating filter        
      gr::filter::mmse_fir_interpolator_cc d_inter; 
      gr::thread::mutex d_mutex;

      /**********************************************************
       * Private function prototypes
       **********************************************************/
      /**
        * @brief Function that processes the ratio received as PMT
        * message from other blocks and assignes it to a variable.
        *  
        * @param pmt_t msg: Message received from autocorrelation
        * block during runtime.
        */
      void set_ratio_msg(pmt::pmt_t msg);
      //---------------------------------------------------------
     
     public:
      frame_drop_impl(int Htotal, int Vtotal, int correct_sampling, float max_deviation, float update_proba, double actual_samp_rate);
      ~frame_drop_impl();

      /**********************************************************
       * Public function prototypes
       **********************************************************/
      /**
        * @brief Used to establish the amount of samples required
        * for a full work iteration.
        *  
        */
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);
      //---------------------------------------------------------
      /**
        * @brief Function that emulates the calculation made by
        * Fine Sampling Synchronization of the samples required
        * for obtained an interpolated result of a certain size.
        *  
        * @param int size: amount of samples desired to result 
        * from the interpolation process, if the was one.
        */
      void get_required_samples(int size);
      //---------------------------------------------------------
      /**
        * @brief While keeping count of the processed samples, the
        * function decides whether each one belongs to a frame that
        * it wants to display in the output or not. To do this, it
        * runs the calculation for the required samples to 
        * interpolate a full frame and counts how many times it is
        * achieved.
        *  
        */
      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
      //---------------------------------------------------------
    };

  } // namespace tempest
} // namespace gr

#endif /* INCLUDED_TEMPEST_FRAME_DROP_IMPL_H */


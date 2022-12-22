/* -*- c++ -*- */
/*
 * Copyright 2020
 *   Federico "Larroca" La Rocca <flarroca@fing.edu.uy>
 *
 *   Instituto de Ingenieria Electrica, Facultad de Ingenieria,
 *   Universidad de la Republica, Uruguay.
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
 */

#ifndef INCLUDED_TEMPEST_SAMPLING_SYNCHRONIZATION_IMPL_H
#define INCLUDED_TEMPEST_SAMPLING_SYNCHRONIZATION_IMPL_H

#include <tempest/sampling_synchronization.h>
#include <gnuradio/filter/mmse_fir_interpolator_cc.h>
#include <random>

namespace gr {
  namespace tempest {

    class sampling_synchronization_impl : public sampling_synchronization
    {
     private:
         int d_Htotal;
         //the number of lines to consider for correlation
         int d_no_of_lines_for_corr; 
         // the maximum number of pixels I'll search for the peak 
         int d_max_deviation_px; 
         // the maximum number of pixels I'll search for the peak in percentaje
         float d_max_deviation; 
         // the estimated interpolation I'd have to use to have a correct sampling - 1
         double d_samp_inc_remainder;
         // a manual correction assigned by the user at run-time
         double d_samp_inc_correction; 
         double d_samp_phase;
         float d_new_interpolation_ratio;

         float d_alpha_samp_inc;

         // to accelerate the process, I'll only update the interpolation
         // once every a random number of iterations with probability d_portion
         std::uniform_real_distribution<float> d_dist;
         std::minstd_rand d_gen;
         float d_proba_of_updating;


        // the correlation
        gr_complex * d_historic_corr;
        float * d_abs_historic_corr;
        gr_complex * d_current_corr;
        gr_complex * d_in_conj; 
        float d_alpha_corr;
        
        // the interpolating filter        
        gr::filter::mmse_fir_interpolator_cc d_inter; 

        void update_interpolation_ratio(const float * datain, const int datain_length);

        int interpolate_input(const gr_complex * in, gr_complex * out, int size);

     public:
      sampling_synchronization_impl(int Htotal, double manual_correction);
      ~sampling_synchronization_impl();

      void set_Htotal(int Htotal);

      void set_manual_correction(double correction);

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

    };

  } // namespace tempest
} // namespace gr

#endif /* INCLUDED_TEMPEST_SAMPLING_SYNCHRONIZATION_IMPL_H */


/* -*- c++ -*- */
/* 
 * Copyright 2020 <+YOU OR YOUR COMPANY+>.
 *   Adapted to 3.10 by Santiago Fernandez <santiago.fernandez.rovira@fing.edu.uy>
 
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
 */

#ifndef INCLUDED_TEMPEST_HSYNC_IMPL_H
#define INCLUDED_TEMPEST_HSYNC_IMPL_H

#include <gnuradio/tempest/Hsync.h>

namespace gr {
  namespace tempest {

    class Hsync_impl : public Hsync
    {
     private:
         int d_delay;
         int d_Htotal;

         int d_line_start;

         // the number of consecutive aligns in shorter ranges (to check whether we are locked)
         int d_consecutive_aligns;
         int d_consecutive_aligns_threshold;
         int d_shorter_range_size;
         int d_max_aligns;

         int d_line_found; 
         // true means I'm sure where the line may be
         int d_line_locked;
         int d_out;
         int d_consumed;

         // For peak detector
        float d_threshold_factor_rise;
        float d_avg_alpha;
        float d_avg_max;
        float d_avg_min; 
      
        // the correlation
        gr_complex * d_corr;
        float * d_abs_corr;


     /*!
      * \brief Initializes the parameters used in the peak_detect_process. 
      *
      * \param threshold_factor_rise The algorithm keeps an average of minimum and maximum value. A peak is considered valid
      * when it's bigger than avg_max - threshold_factor_rise(avg_max-avg_min). 
      * \param alpha The parameter used to update both the average maximum and minimum (exponential filter, or single-root iir). 
      *
      */ 
      void peak_detect_init(float threshold_factor_rise, float alpha);
     
      /*!
       * \brief Given datain and its length, the method return the peak position 
       */ 
      int peak_detect_process(const float * datain, const int datain_length, int * peak_pos);

      /*!
       * \brief Calculates the conjugate multipliplication of the signal and its delayed version, 
       *  and outputs the position of its absolute maximum. 
       *
       * Given the input, it calculates the resulting likelihood function between indices lookup_stop and lookup_start. 
       * It returns the beginning of the maximum 1-sample correlation, and an exponential modulated with minus the estimated 
       * frequency error (in the pointer derot). to_consume and to_out was used as indicators of whether the peak was 
       * correctly found or not. Now  the return value is used (either true or false). 
       *
       */
      int max_corr_sync(const gr_complex * in, int lookup_start, int lookup_stop, int * max_corr_pos);


     public:
      Hsync_impl(int Htotal, int delay);
      ~Hsync_impl();

      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      // Where all the action really happens
      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
      
      void set_Htotal_and_delay(int Htotal, int delay);

    };

  } // namespace tempest
} // namespace gr

#endif /* INCLUDED_TEMPEST_HSYNC_IMPL_H */


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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "normalize_flow_impl.h"
#include <volk/volk.h>

namespace gr {
  namespace tempest {

    normalize_flow::sptr
    normalize_flow::make(float min, float max, int window, float alpha_avg, float update_proba)
    {
      return gnuradio::get_initial_sptr
        (new normalize_flow_impl(min, max, window, alpha_avg, update_proba));
    }

    /*
     * The private constructor
     */
    normalize_flow_impl::normalize_flow_impl(float min, float max, int window, float alpha_avg, float update_proba)
      : gr::sync_block("normalize_flow",
              gr::io_signature::make(1, 1, sizeof(float)),
              gr::io_signature::make(1, 1, sizeof(float))),
              d_dist(0, 1),
              d_gen(std::random_device{}())
    {
        d_min = min; 
        d_max = max; 
        d_win = window;
        d_alpha_avg = alpha_avg;

        set_output_multiple(d_win);

        d_current_max = d_max;
        d_current_min = d_min;
        
        d_proba_of_updating = update_proba;
    }

    void normalize_flow_impl::set_min_max(float min, float max){
        d_min = min; 
        d_max = max;
        printf("[TEMPEST] setting min and max to %f and %f in the normalize flow block.\n",min, max);
    }

    float normalize_flow_impl::compute_max(const float * datain, const int datain_length){
      uint16_t peak_index = 0;
      uint32_t d_datain_length = (uint32_t)datain_length;

      volk_32f_index_max_16u(&peak_index, &datain[0], d_datain_length); 
      return datain[peak_index];

    }

    /*
     * Our virtual destructor.
     */
    normalize_flow_impl::~normalize_flow_impl()
    {
    }

    int
    normalize_flow_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
    {
      const float *in = (const float *) input_items[0];
      float *out = (float *) output_items[0];

   

      if(d_dist(d_gen)<d_proba_of_updating){
          float max = compute_max(&in[0], noutput_items);
          d_current_max = (1-d_alpha_avg)*d_current_max + d_alpha_avg*max;

          d_minus_datain = new float[noutput_items];
          volk_32f_s32f_multiply_32f(&d_minus_datain[0], &in[0], -1, noutput_items);
          float minusmin = compute_max(&d_minus_datain[0], noutput_items);
          d_current_min = (1-d_alpha_avg)*d_current_min - d_alpha_avg*minusmin;
      }

      // There's no scalar add in VOLK!!!
      for (int i=0; i<noutput_items; i++){
          out[i] = (in[i]-d_current_min)/(d_current_max-d_current_min)*(d_max-d_min) + d_min; 
      }


      // Tell runtime system how many output items we produced.
      return noutput_items;
    }

  } /* namespace tempest */
} /* namespace gr */


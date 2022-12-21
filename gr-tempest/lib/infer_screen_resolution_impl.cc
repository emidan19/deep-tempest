/* -*- c++ -*- */
/*
 * Copyright 2020
 *   Federico "Larroca" La Rocca <flarroca@fing.edu.uy>
 *   Adapted to 3.10 by Santiago Fernandez <santiago.fernandez.rovira@fing.edu.uy>
 
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
#include "infer_screen_resolution_impl.h"

#define N 64
#define lowpasscoeff 0.6 // TODO: check this for every resolution.
#define MAX_PERIOD 0.0000284

namespace gr {
  namespace tempest {

    infer_screen_resolution::sptr
    infer_screen_resolution::make(int sample_rate, int fft_size, float refresh_rate, bool automatic_mode)
    {
      return gnuradio::get_initial_sptr
        (new infer_screen_resolution_impl(sample_rate, fft_size, refresh_rate, automatic_mode));
    }


    /*
     * The private constructor
     */
    infer_screen_resolution_impl::infer_screen_resolution_impl(int sample_rate, int fft_size, float refresh_rate, bool automatic_mode)
      : gr::block("infer_screen_resolution",
              gr::io_signature::make(1, 1, sizeof(float)),
              gr::io_signature::make(1, 1, sizeof(float)))
    {
      d_start_fft_peak_finder = 1;
      
      //Received parameters
      d_sample_rate = sample_rate;
      d_fft_size = fft_size;
      d_mode = automatic_mode;

      //Search values
      d_search_skip = 0;
      d_search_margin = d_fft_size;
      d_vtotal_est = 800;
      d_peak_1 = 0;
      d_peak_2 = 0;
      
      //Parameters to publish
      d_refresh_rate = refresh_rate;
      d_refresh_rate_est = 0;    
      d_Hblank = 0;
      d_Vblank = 0;

      d_start = true;

      d_ratio = 0;
      d_accumulator = 0.0f;
      //d_real_line = 827;  
      d_real_line = 827.076923077;  

      //Counters
      d_work_counter = 1;
      d_i = 0;


    }

    /*
     * Our virtual destructor.
     */
    infer_screen_resolution_impl::~infer_screen_resolution_impl()
    {
    }

    void
    infer_screen_resolution_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items;
    }


    //---------------------------------------------------------

    void infer_screen_resolution_impl::set_refresh_rate(float refresh_rate)
    {
      gr::thread::scoped_lock l(d_mutex);

      //If the refresh rate changed, parameters are reset with callback
      d_refresh_rate = refresh_rate;
      d_search_skip = d_sample_rate/(d_refresh_rate+0.2);
      d_refresh_rate_est = refresh_rate;
      printf("[TEMPEST] Setting refresh to %i in infer block.\n", refresh_rate);
    }
    
    //---------------------------------------------------------

    int
    infer_screen_resolution_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const float *in = (const float *) input_items[0];
      float *out = (float *) output_items[0];

      gr::thread::scoped_lock l(d_mutex);

      if(!d_start_fft_peak_finder)
      {
                consume_each(noutput_items);
                return noutput_items;
      }
      else
      {
                /////////////////////////////
                //      RATIO SEARCH       //
                /////////////////////////////
                /* 
                    If we receive a full d_fft_size from the fft_autocorrelation block
                  we process the full d_fft_size vector to find the best two peaks, peak_1, peak_2.
                    We should consume d_fft_size data from the in[0] to the in[d_fft_size] 
                  and compute the distance between peak_1 and peak_2 in samples.
                    The value of d_accumulator should be the moving average of 
                  the distance between peak_2 and peak_1. 
                    So we divide the thing over N and repeat the measurement N times.
                */ 
                if(d_sample_counter > 1)
                {
                      if(d_mode)
                      {
                          // Automatic mode.
                          uint32_t one_full_frame_in_samples = floor( (1.0/d_refresh_rate) * d_sample_rate );
                          d_search_margin = d_fft_size;
                          d_search_skip = 0;
                          d_peak_1 = calculate_peak_index_relative_to_search_skip(
                                                  in, 
                                                  d_search_skip, 
                                                  d_search_margin
                                                );
                          d_search_skip = d_peak_1 + one_full_frame_in_samples - floor((0.004)*d_sample_rate);
                          d_search_margin = 200 + floor((0.004)*5*d_sample_rate);

                          d_peak_2 = calculate_peak_index_relative_to_search_skip(
                                                    in, 
                                                    d_search_skip, 
                                                    d_search_margin
                                                  );

                          d_accumulator += (long double)(d_peak_2-d_peak_1)/(long double)(N);
                      }
                      else
                      {
                          // Semi-automatic mode.
                          d_peak_1 = nitems_written(0) + 0;
                          d_search_skip = d_sample_rate / (d_refresh_rate + 0.2);
                          d_search_margin = 10000;

                          d_peak_2 = calculate_peak_index_relative_to_search_skip(
                                                    in, 
                                                    d_search_skip, 
                                                    d_search_margin
                                                  );

                          d_accumulator = d_peak_2;
                      }
                      
                      if(d_work_counter%N == 0)
                      {
                                    uint32_t  yt_index = 0, yt_aux = 0;
                                    double fv = (double)d_sample_rate/(double)d_accumulator;

                                    // Lower the variation of the received refresh rate:
                                    d_refresh_rate_est = ((long) round(fv * lowpasscoeff + (1.0 - lowpasscoeff) * (d_refresh_rate_est)));
                                  
                                    /////////////////////////////
                                    //     HEIGHT SEARCH       //
                                    /////////////////////////////

                                    int yt_largo = (int)d_sample_rate*(MAX_PERIOD);

                                    volk_32f_index_max_32u(&yt_index, &in[(d_peak_2)+5], yt_largo);
                                    // The peak search begins a few samples later to avoid repeating the previous result

                                    double yt = (double)d_sample_rate / (double)((yt_index+5)*fv);
                                    // The same sample movement is compensated

                                    if (d_flag)  
                                    {
                                      if (yt < 1225 && yt > 350)
                                        d_vtotal_est = ((int) round(yt * lowpasscoeff + (1.0 - lowpasscoeff) * (d_vtotal_est)));
                                      
                                    }
                                    else 
                                    {
                                      if (yt < 1225 && yt > 350)
                                      {
                                        d_vtotal_est = yt;
                                        d_flag = true;

                                      }
                                    }
                                    //printf(" yt instant \t %lf \t  yt estimate \t %ld \t \n ", yt, d_vtotal_est);
                                      
                                    /////////////////////////////
                                    //    UPDATE RESULTS       //
                                    /////////////////////////////

                                    int last_result = d_Vvisible;

                                    search_table(d_refresh_rate_est); 

                                    if (last_result == d_Vvisible) {
                                      d_i++;
                                    } else {
                                      d_i=0;
                                    }

                                    if (d_i == 15) {
                                      printf(" Hdisplay \t %ld \t Px \t\t Vdisplay \t %ld \t Px \t\t Hsize \t %ld \t Px \t\t Vsize \t %ld \t Px \t\t Refresh Rate \t %f \t Hz \t \n ", d_Hvisible,d_Vvisible,d_Hsize,d_Vsize,d_refresh_rate);
                                      d_i=0;
                                    }                

                                    d_accumulator = 0;
                                    d_work_counter = 0;
                                    d_sample_counter = 0;         
                      }
                }
                

      } 
      memcpy(&out[0], &in[0], noutput_items*sizeof(float)); // el tag deberia hacerse repetidamente aca con d_peak_1 d_peak_2

      d_work_counter++;   
      d_sample_counter+=noutput_items;
      
      consume_each (noutput_items);

      add_item_tag(
        0, 
        nitems_written(0) + d_peak_1, 
        pmt::mp("peak_1"), 
        pmt::PMT_T
      ); 
      add_item_tag(
        0, 
        nitems_written(0) + d_peak_2, 
        pmt::mp("peak_2"), pmt::PMT_T
      );
      return noutput_items; 
                
    }

  } /* namespace tempest */
} /* namespace gr */


/* -*- c++ -*- */
/**
 * Copyright 2021
 *    Pablo Bertrand    <pablo.bertrand@fing.edu.uy>
 *    Felipe Carrau     <felipe.carrau@fing.edu.uy>
 *    Victoria Severi   <maria.severi@fing.edu.uy>
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
 * @file fft_peak_fine_sampling_sync_impl.cc
 *
 * gr-tempest
 *
 * @date September 19, 2021
 * @author  Pablo Bertrand   <pablo.bertrand@fing.edu.uy>
 * @author  Felipe Carrau    <felipe.carrau@fing.edu.uy>
 * @author  Victoria Severi  <maria.severi@fing.edu.uy>
 */

/**********************************************************
 * Include statements
 **********************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "fft_peak_fine_sampling_sync_impl.h"
#include <thread>
#include <volk/volk.h>
#include <math.h>

/**********************************************************
 * Constant and macro definitions
 **********************************************************/


uint32_t calculate_peak_index_relative_to_search_skip(const float *in, uint32_t search_skip, uint32_t search_range)
{
    uint32_t peak_index;
    volk_32f_index_max_32u(&peak_index, &in[search_skip], search_range);   
    return peak_index += search_skip;                                                

}

#define N 64

namespace gr {
  namespace tempest {

    fft_peak_fine_sampling_sync::sptr
    fft_peak_fine_sampling_sync::make(int sample_rate, int size, int refresh_rate, int Vvisible, int Hvisible, bool automatic_mode)
    {
        return gnuradio::get_initial_sptr
          (new fft_peak_fine_sampling_sync_impl(sample_rate, size, refresh_rate, Vvisible, Hvisible, automatic_mode));
    }
    
    /**********************************************************
     * Function bodies
     **********************************************************/
    /*
     * The private constructor
     */
    fft_peak_fine_sampling_sync_impl::fft_peak_fine_sampling_sync_impl(int sample_rate, int size, int refresh_rate, int Vvisible, int Hvisible, bool automatic_mode)
      : gr::block("fft_peak_fine_sampling_sync",
              gr::io_signature::make(1, 1, sizeof(float)),
              gr::io_signature::make(1, 1, sizeof(float)))
    {
        d_start_fft_peak_finder = 1;
        
        //Received parameters
        d_sample_rate = sample_rate;
        d_fft_size = size;

        //Search values
        d_search_skip = 0;
        d_search_margin = d_fft_size;
        d_vtotal_est = 0;
        d_peak_1 = 0;
        d_peak_2 = 0;
        
        //Parameters to publish
        d_refresh_rate = refresh_rate;
        d_refresh_rate_est = 0;    
        d_Hvisible = Hvisible;
        d_Vvisible = Vvisible;
        d_Hblank = 0;
        d_Vblank = 0;

        d_start = true;

        d_ratio = 0;
        d_accumulator = 0.0f;
        //d_real_line = 827;  
        d_real_line = 827.076923077;  

        //Counters
        d_work_counter = 1;
        
        //PMT ports
        message_port_register_out(pmt::mp("en"));
        message_port_register_out(pmt::mp("ratio"));
        message_port_register_out(pmt::mp("rate"));
        
        message_port_register_in(pmt::mp("en"));

        // PMT handlers
        set_msg_handler(pmt::mp("en"),   [this](const pmt::pmt_t& msg) {fft_peak_fine_sampling_sync_impl::set_ena_msg(msg); });

        set_history(d_fft_size);

        printf("[TEMPEST] Welcome to gr-tempest. Once the sampling is being corrected properly and the vertical lines of the target monitor are indeed vertical, please hit the Stop button to stop autocorrelation calculation and begin the vertical and horizontal synchronization .\n");
    }

    //---------------------------------------------------------
    /*
     * Our virtual destructor.
     */
    fft_peak_fine_sampling_sync_impl::~fft_peak_fine_sampling_sync_impl()
    {
    }

    void fft_peak_fine_sampling_sync_impl::set_ena_msg(pmt::pmt_t msg)
    {
        gr::thread::scoped_lock l(d_mutex); 
        if (pmt::is_bool(msg)) {
            bool en = pmt::to_bool(msg);
            d_start_fft_peak_finder = !en;
            printf("FFT peak finder. Ratio calculation stopped.\n");
        } else {
            GR_LOG_WARN(d_logger,
                        "FFT peak finder: Non-PMT type received, expecting Boolean PMT\n");
            if(d_start_fft_peak_finder)
            {
              printf("FFT peak finder. Ratio calculation stopped.\n");
              d_start_fft_peak_finder = 0;
            }  
            else
            {
              printf("FFT peak finder. Ratio calculation restarted.\n");
              d_start_fft_peak_finder = 1;
            }
            /*
              Stop FFT_autocorrelation block.
            */
            message_port_pub(
              pmt::mp("en"), 
              pmt::from_bool(d_start_fft_peak_finder)
            );

            /* Hardware USRP rate command: */
            
            double new_freq = ((double)1.0/((double)(d_ratio + 1))) * (double)d_sample_rate;

            message_port_pub(pmt::mp("rate"), pmt::cons(pmt::mp("rate"), pmt::from_double(new_freq))); 
            d_sample_rate = new_freq;
            new_freq = d_ratio;
            message_port_pub(
              pmt::mp("ratio"), pmt::cons(
                                  pmt::mp("ratio"), 
                                  pmt::from_double(new_freq)
                                )
            );
            
        }
    }

    //---------------------------------------------------------

    void fft_peak_fine_sampling_sync_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
        ninput_items_required[0] = noutput_items; // Funny Obs: Leaving ninput_items_required uninitialized creates chaos.
    }

    //---------------------------------------------------------


    //---------------------------------------------------------
    int fft_peak_fine_sampling_sync_impl::general_work (int noutput_items,
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
                uint32_t one_full_frame_in_samples = floor( (1.0/d_refresh_rate) * d_sample_rate );
                
                d_search_margin = d_fft_size;
                d_search_skip = 0;
                d_peak_1 = calculate_peak_index_relative_to_search_skip(
                                        in, 
                                        d_search_skip, 
                                        d_search_margin
                                      );
 
                d_search_skip = d_peak_1 + one_full_frame_in_samples - floor((0.001)*d_sample_rate);
                d_search_margin = 200 + floor((0.001)*4*d_sample_rate);
                d_peak_2 = calculate_peak_index_relative_to_search_skip(
                                          in, 
                                          d_search_skip, 
                                          d_search_margin
                                        );

                d_accumulator += (long double)(d_peak_2-d_peak_1)/(long double)(N);
                
                if(d_work_counter%N == 0)
                {
                              long double ratio = (long double)(d_accumulator)/(long double)(d_Hvisible*d_Vvisible);

                              d_ratio = (ratio-1);
                              
                              /* 
                                Send ratio message to the interpolator. 
                              */
                              double new_freq = d_ratio;
                              message_port_pub(
                                pmt::mp("ratio"), pmt::cons(
                                                    pmt::mp("ratio"), 
                                                    pmt::from_double(new_freq)
                                                  )
                              );
                              printf("\r\n[FFT_peak_finder] Ratio = \t %Lf. \t d_accumulator = \t %Lf. \t \r\n ", ratio, d_accumulator);  
                              
                              d_accumulator = 0;
                              d_work_counter = 0;
                              /* 
                                  Maybe sleep for a few milliseconds here.
                              */
                              long period_ms = (1000);
                              //boost::this_thread::sleep(  boost::posix_time::milliseconds(static_cast<long>(period_ms)) );
                           
                
                           
                              d_sample_counter = 0;         
                }
                
                

      } 
      memcpy(&out[0], &in[0], noutput_items*sizeof(float)); // el tag deberia hacerse repetidamente aca con d_peak_1 d_peak_2

      d_work_counter++;   
      d_sample_counter+=noutput_items;
      
                          
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
      add_item_tag(
        0, 
        nitems_written(0) + d_search_skip, 
        pmt::mp("Search1"), 
        pmt::PMT_T
      ); 
      add_item_tag(
        0, 
        nitems_written(0) + d_search_skip + d_search_margin, 
        pmt::mp("Search2"), pmt::PMT_T
      );
      consume_each (noutput_items);

      return noutput_items; 

    }

  } /* namespace tempest */
} /* namespace gr */


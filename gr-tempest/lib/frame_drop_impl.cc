/* -*- c++ -*- */
/**
 * Copyright 2021
 *    Pablo Bertrand    <pablo.bertrand@fing.edu.uy>
 *    Felipe Carrau     <felipe.carrau@fing.edu.uy>
 *    Victoria Severi   <maria.severi@fing.edu.uy>
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
 * @file frame_drop_impl.cc
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

#include <stdio.h>
#include <math.h>
#include <gnuradio/io_signature.h>
#include "frame_drop_impl.h"
#include <volk/volk.h>
#include <random>
#include <vector>

namespace gr 
{
  namespace tempest 
  {

    frame_drop::sptr
    frame_drop::make(int Htotal, int Vtotal, int correct_sampling, float max_deviation, float update_proba, double actual_samp_rate)
    {
      return gnuradio::get_initial_sptr
        (new frame_drop_impl(Htotal, Vtotal, correct_sampling, max_deviation, update_proba, actual_samp_rate));
    }

    /**********************************************************
     * Function bodies
     **********************************************************/
    /*
     * The private constructor
     */
    frame_drop_impl::frame_drop_impl(int Htotal, int Vtotal, int correct_sampling, float max_deviation, float update_proba, double actual_samp_rate)
      : gr::block("frame_drop",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex))),
      d_inter(gr::filter::mmse_fir_interpolator_cc()),
      d_dist(update_proba),
      d_gen(std::random_device{}()) 
    {
      set_relative_rate(1);

      //Counters
      d_sample_counter = 0; 
      d_display_counter = 0;
      d_frames_counter = 0;

      //Fixed values
      d_discarded_amount_per_frame = 15;
      d_proba_of_updating = update_proba;
      d_max_deviation = max_deviation;
      d_alpha_samp_inc = 1e-1;
      d_last_freq = 0;
      d_samp_phase = 0; 
      d_alpha_corr = 1e-6; 

      d_start_frame_drop = 0;
      
      d_Htotal = Htotal; 
      d_Vtotal = Vtotal; 
      d_required_for_interpolation = d_Htotal*d_Vtotal;
      d_max_deviation_px = (int)std::ceil(d_Htotal*d_max_deviation);

      set_history(d_Vtotal*(d_Htotal+d_max_deviation_px)+1);

      d_peak_line_index = 0;
      d_samp_inc_rem = 0;
      d_new_interpolation_ratio_rem = 0;
      d_next_update = 0;

      // PMT port
      message_port_register_in(pmt::mp("ratio"));

      // PMT handler
      set_msg_handler(pmt::mp("ratio"),[this](const pmt::pmt_t& msg) {frame_drop_impl::set_ratio_msg(msg); });

    }

    //---------------------------------------------------------
    /*
     * Our virtual destructor.
     */
    frame_drop_impl::~frame_drop_impl()
    {
    }

    //---------------------------------------------------------

    void frame_drop_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
        int ninputs = ninput_items_required.size ();
        for (int i = 0; i < ninputs; i++)
        {
            ninput_items_required[i] = (int)ceil((noutput_items + 1) * (2+d_samp_inc_rem)) + d_inter.ntaps() ;
        }
    }

    //---------------------------------------------------------

    void frame_drop_impl::set_ratio_msg(pmt::pmt_t msg){
      if(pmt::is_pair(msg)) {
          pmt::pmt_t key = pmt::car(msg);
          pmt::pmt_t val = pmt::cdr(msg);
          if(pmt::eq(key, pmt::string_to_symbol("ratio"))) {
              if(pmt::is_number(val)) {
                  d_new_interpolation_ratio_rem = (double)pmt::to_double(val);
                  printf("Frame dropper: interpolation ratio received = %f \n", d_new_interpolation_ratio_rem);
              }
          }
      }
    } 

    //---------------------------------------------------------

    void frame_drop_impl::get_required_samples(int size)
    {
      uint32_t ii = 0, incr;
      int oo = 0;
      double s, f;
      
      //d_samp_inc_rem = (1-d_alpha_samp_inc)*d_samp_inc_rem + d_alpha_samp_inc*d_new_interpolation_ratio_rem;

      while(oo < size) {
        s = d_samp_phase + d_samp_inc_rem + 1;
        f = floor(s);
        incr = (uint32_t)f;
        d_samp_phase = s - f;
        ii += incr;

        oo++;
      }
      d_required_for_interpolation = ii;
    }

    //---------------------------------------------------------

    int frame_drop_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex *in = (const gr_complex *) input_items[0];
      gr_complex *out = (gr_complex *) output_items[0];

      int consumed = 0, out_amount = 0, aux;
      
      d_samp_inc_rem = (1-d_alpha_samp_inc)*d_samp_inc_rem + d_alpha_samp_inc*d_new_interpolation_ratio_rem;

      for (int i=0; i<noutput_items; i++){

        d_sample_counter++;

        if (d_sample_counter <= d_required_for_interpolation){
          out[i]=in[i];
          out_amount++;
        }

        if (d_sample_counter == (d_discarded_amount_per_frame*d_required_for_interpolation)){
          consumed = i;
          d_sample_counter = 0;

          add_item_tag(0, nitems_written(0)+i, pmt::mp("trigger"), pmt::PMT_T);
          break;
        } 

        if (d_sample_counter%d_required_for_interpolation==0){
          get_required_samples(d_Htotal*d_Vtotal);
        }
      }

      consumed == 0 ? consumed += noutput_items : consumed += 0;

      // Tell runtime system how many input items we consumed on
      // each input stream.
      consume_each (consumed);

      // Tell runtime system how many output items we produced.
      return out_amount;
    
    }    
    
  } /* namespace tempest */
} /* namespace gr */



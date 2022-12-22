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

#include <volk/volk.h>
#include <random>
#include "ssamp_correction_impl.h"

namespace gr {
  namespace tempest {

    ssamp_correction::sptr
    ssamp_correction::make(int Htotal, int Vtotal, int correct_sampling, float max_deviation)
    {
      return gnuradio::get_initial_sptr
        (new ssamp_correction_impl(Htotal, Vtotal, correct_sampling, max_deviation));
    }


    /*
     * The private constructor
     */
    ssamp_correction_impl::ssamp_correction_impl(int Htotal, int Vtotal, int correct_sampling, float max_deviation)
      : gr::block("ssamp_correction",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex))),
            d_inter(gr::filter::mmse_fir_interpolator_cc()),
            d_gen(std::random_device{}())
    {
      set_relative_rate(1);
      d_correct_sampling = correct_sampling; 
      d_max_deviation = max_deviation;
      d_Htotal = Htotal; 
      d_Vtotal = Vtotal; 
      d_max_deviation_px = (int)std::ceil(d_Htotal*d_max_deviation);
      
      set_history(d_Vtotal*(d_Htotal+d_max_deviation_px)+1);

      d_peak_line_index = 0;
      d_samp_inc_rem = 0;
      d_stop_fine_sampling_synch = 0;
      d_new_interpolation_ratio_rem = 0;
      
      d_current_line_corr = new gr_complex[2*d_max_deviation_px + 1];
      d_historic_line_corr = new gr_complex[2*d_max_deviation_px + 1];
      d_abs_historic_line_corr = new float[2*d_max_deviation_px + 1];

      /** Note: d_current_frame_corr[i] and derivatives will keep the correlation between pixels 
          px[t] and px[t+Htotal*Vtotal+i]. Since a single pixel de-alignment with the next line will 
          mean d_Vtotal pixels de-alignments with the next frame, these arrays are much bigger. 
          However, instead of always calculating the whole of them, I'll only calculate around those
          indicated by the max in the d_abs_historic_line_corr. 
      */
      d_current_frame_corr = new gr_complex[2*(d_max_deviation_px+1)*d_Vtotal + 1];
      d_historic_frame_corr = new gr_complex[2*(d_max_deviation_px+1)*d_Vtotal + 1];
      d_abs_historic_frame_corr = new float[2*(d_max_deviation_px+1)*d_Vtotal + 1];
      
      d_alpha_samp_inc = 1e-1;
      d_samp_phase = 0; 
      d_alpha_corr = 1e-2; 
      d_next_update = 0;

      for (int i = 0; i<2*d_max_deviation_px+1; i++){
          d_historic_line_corr[i] = 0;
          d_abs_historic_line_corr[i] = 0;
      }
      for (int i = 0; i<2*d_max_deviation_px*d_Vtotal+1; i++){
          d_historic_frame_corr[i] = 0;
          d_abs_historic_frame_corr[i] = 0;
      }

      printf("[TEMPEST] Construction of ssamp_correction_impl with Htotal=%i and Vtotal=%i.\n", Htotal, Vtotal);

      //VOLK alignment as recommended by GNU Radio's Manual. It has a similar effect 
      //than set_output_multiple(), thus we will generally get multiples of this value
      //as noutput_items. 
      const int alignment_multiple = volk_get_alignment() / sizeof(gr_complex);
      set_alignment(std::max(1, alignment_multiple));

      // PMT ports
      message_port_register_in(pmt::mp("ratio"));
      message_port_register_in(pmt::mp("en"));

      // PMT handlers 
      //              : Port                Lambda function
      set_msg_handler(  pmt::mp("ratio"),   [this](const pmt::pmt_t& msg) {ssamp_correction_impl::set_ratio_msg(msg); });
      set_msg_handler(  pmt::mp("en"),      [this](const pmt::pmt_t& msg) {ssamp_correction_impl::set_ena_msg(msg); });
    }


    void 
    ssamp_correction_impl::set_ena_msg(pmt::pmt_t msg)
    {
      if (pmt::is_bool(msg)) {
          bool en = pmt::to_bool(msg);
          gr::thread::scoped_lock l(d_mutex);
          d_stop_fine_sampling_synch = !en;
          printf("ssamp_correction_impl Received Sampling Stop.\n");
      } else {
          GR_LOG_WARN(d_logger,
                      "ssamp_correction_impl Received : Non-PMT type received, expecting Boolean PMT\n");
      }
    }

    void 
    ssamp_correction_impl::set_ratio_msg(pmt::pmt_t msg)
    {
      if(pmt::is_pair(msg)) {
          // saca el primero de la pareja
          pmt::pmt_t key = pmt::car(msg);
          // saca el segundo
          pmt::pmt_t val = pmt::cdr(msg);
          if(pmt::eq(key, pmt::string_to_symbol("ratio"))) {
              if(pmt::is_number(val)) {
                  d_new_interpolation_ratio_rem = (double)pmt::to_double(val);
                  printf("ssamp_correction_impl Received : interpolation ratio  = %f \n", d_new_interpolation_ratio_rem);
              }
          }
      }
    }

    /*
     * Our virtual destructor.
     */
    ssamp_correction_impl::~ssamp_correction_impl()
    {
      delete [] d_current_line_corr;
      delete [] d_historic_line_corr;
      delete [] d_abs_historic_line_corr;
      delete [] d_current_frame_corr;
      delete [] d_historic_frame_corr;
      delete [] d_abs_historic_frame_corr;
    }

    void
    ssamp_correction_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      int ninputs = ninput_items_required.size ();
      // make sure we receive at least Hsize+max_deviation+taps_to_interpolate
      for (int i = 0; i < ninputs; i++)
      {
          ninput_items_required[i] = (int)ceil((noutput_items + 1) * (2+d_samp_inc_rem)) + d_inter.ntaps() ;
      }

    }

    void 
    ssamp_correction_impl::set_Htotal_Vtotal(int Htotal, int Vtotal){
      // If the resolution's changed, I reset the whole block
      /**
       * @fcarraustewart
       *  FIXME: This callback double taps onChange 
       * of any of these two variables, Htotal Vtotal,
       *  This happens on all our testXXX.grc files.
       *  Solution .grc file should not have a link 
       * between these two variables?
       */
      d_Htotal = Htotal; 
      d_Vtotal = Vtotal; 
      
      d_max_deviation_px = (int)std::ceil(d_Htotal*d_max_deviation);
      printf("ssamp_correction_impl d_max_deviation_px: %i\n", d_max_deviation_px);
      set_history(d_Vtotal*(d_Htotal+d_max_deviation_px)+1);

      d_peak_line_index = 0;
      d_samp_inc_rem = 0;
      d_stop_fine_sampling_synch = 0;
      d_new_interpolation_ratio_rem = 0;

      /**
       * < Calling delete for each of the following pointers 
       * and assigning them nullptr right here. Before running the new operator.
       * 
       * */
      delete [] d_current_line_corr;
      delete [] d_historic_line_corr;
      delete [] d_abs_historic_line_corr;
      delete [] d_current_frame_corr;
      delete [] d_historic_frame_corr;
      delete [] d_abs_historic_frame_corr;

      d_current_line_corr = new gr_complex[2*d_max_deviation_px + 1];
      d_historic_line_corr = new gr_complex[2*d_max_deviation_px + 1];
      d_abs_historic_line_corr = new float[2*d_max_deviation_px + 1];
      d_current_frame_corr = new gr_complex[2*(d_max_deviation_px+1)*d_Vtotal + 1];
      d_historic_frame_corr = new gr_complex[2*(d_max_deviation_px+1)*d_Vtotal + 1];
      d_abs_historic_frame_corr = new float[2*(d_max_deviation_px+1)*d_Vtotal + 1];
      //I'll estimate the new sampling synchronization asap
      d_next_update = 0;

      for (int i = 0; i<2*d_max_deviation_px+1; i++){
          d_historic_line_corr[i] = 0;
          d_abs_historic_line_corr[i] = 0;
      }
      for (int i = 0; i<2*d_max_deviation_px*d_Vtotal+1; i++){
          d_historic_frame_corr[i] = 0;
          d_abs_historic_frame_corr[i] = 0;
      }

      printf("[TEMPEST] Construction of ssamp_correction_impl with Htotal=%i and Vtotal=%i.\n", Htotal, Vtotal);

    }
    
    int 
    ssamp_correction_impl::interpolate_input(const gr_complex * in, 
                        gr_complex * out, 
                        int size)
    {
      int ii = 0; // input index
      int oo = 0; // output index

      double s, f; 
      int incr; 
      while(oo < size) {
          out[oo++] = d_inter.interpolate(&in[ii], d_samp_phase);

          s = d_samp_phase + d_samp_inc_rem + 1;
          f = floor(s);
          incr = (int)f;
          d_samp_phase = s - f;
          ii += incr;
      }

      // return how many inputs we required to generate d_cp_length+d_fft_length outputs 
      return ii;
    }

    int
    ssamp_correction_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex *in = (const gr_complex *) input_items[0];
      gr_complex *out = (gr_complex *) output_items[0];
      gr::thread::scoped_lock l(d_mutex);
      int required_for_interpolation = noutput_items; 
      
      d_samp_inc_rem = d_new_interpolation_ratio_rem;
      required_for_interpolation = interpolate_input(&in[0], &out[0], noutput_items);
      consume_each (required_for_interpolation);
      return noutput_items;
    }

  } /* namespace tempest */
} /* namespace gr */


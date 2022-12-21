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
#include "sampling_synchronization_impl.h"
#include <volk/volk.h>
#include <random>

namespace gr {
  namespace tempest {

    sampling_synchronization::sptr
    sampling_synchronization::make(int Htotal, double manual_correction)
    {
      return gnuradio::get_initial_sptr
        (new sampling_synchronization_impl(Htotal, manual_correction));
    }


    /*
     * The private constructor
     */
    sampling_synchronization_impl::sampling_synchronization_impl(int Htotal, double manual_correction)
      : gr::block("sampling_synchronization",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex))), 
              d_inter(gr::filter::mmse_fir_interpolator_cc()),
              d_dist(0, 1),
              d_gen(std::random_device{}())

    {
        set_relative_rate(1);
        d_Htotal = Htotal;

        //VOLK alignment as recommended by GNU Radio's Manual. It has a similar effect 
        //than set_output_multiple(), thus we will generally get multiples of this value
        //as noutput_items. 
        const int alignment_multiple = volk_get_alignment() / sizeof(gr_complex);
        set_alignment(std::max(1, alignment_multiple));

        d_max_deviation = 0.10;
        d_max_deviation_px = (int)std::ceil(Htotal*d_max_deviation);
        d_samp_inc_remainder = 0;
        d_samp_inc_correction = manual_correction;
        d_samp_phase = 0; 
        d_alpha_samp_inc = 1e-3;


        d_current_corr = new gr_complex[2*d_max_deviation_px + 1];
        d_historic_corr = new gr_complex[2*d_max_deviation_px + 1];
        d_abs_historic_corr = new float[2*d_max_deviation_px + 1];
        for (int i = 0; i<2*d_max_deviation_px+1; i++){
            d_historic_corr[i] = 0;
            d_abs_historic_corr[i] = 0;
        }
        d_alpha_corr = 1e-6; 

        d_proba_of_updating = 0.01;

    }

    /*
     * Our virtual destructor.
     */
    sampling_synchronization_impl::~sampling_synchronization_impl()
    {
        delete [] d_current_corr;
        delete [] d_historic_corr;
        delete [] d_abs_historic_corr;
    }

    void
    sampling_synchronization_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
           int ninputs = ninput_items_required.size ();
           // make sure we receive at least Hsize+max_deviation+taps_to_interpolate
           for (int i = 0; i < ninputs; i++)
           {
               //ninput_items_required[i] = ( d_cp_length + d_fft_length ) * (noutput_items + 1) ;
               ninput_items_required[i] = (int)ceil((noutput_items + 1) * (1+d_samp_inc_remainder)) + d_inter.ntaps() + d_Htotal+d_max_deviation_px ;
           }
    }

    void sampling_synchronization_impl::set_Htotal(int Htotal){
        d_Htotal = Htotal;
        d_max_deviation_px = (int)std::ceil(Htotal*d_max_deviation);
        printf("[TEMPEST] Setting Htotal to %i in sampling synchronization block.\n", Htotal);
    }
    
    void sampling_synchronization_impl::set_manual_correction(double correction){

        d_samp_inc_correction = correction;

    }

    void sampling_synchronization_impl::update_interpolation_ratio(const float * datain, const int datain_length){
            uint16_t peak_index = 0;
            uint32_t d_datain_length = (uint32_t)(datain_length);
            volk_32f_index_max_16u(&peak_index, datain, datain_length); 

            // the new interpolation ratio is how far the peak is from d_Htotal.
            d_new_interpolation_ratio = ((float)(peak_index-d_max_deviation_px + d_Htotal))/(float)d_Htotal;
            //for (int i=0; i<datain_length; i++)
            //    printf("datain[%i]=%f\n", i, datain[i]);
            //printf("peak_index: %i\n",peak_index);
            //printf("new_interpolation_ratio: %f\n",d_new_interpolation_ratio);
            //printf("d_samp_inc: %.20f\n",d_samp_inc_remainder+1);

            //d_samp_inc = (1-d_alpha_samp_inc)*d_samp_inc - d_alpha_samp_inc* d_new_interpolation_ratio;
            d_samp_inc_remainder = d_samp_inc_remainder - d_alpha_samp_inc*(d_samp_inc_remainder+1 - d_new_interpolation_ratio);

    }

    int sampling_synchronization_impl::interpolate_input(const gr_complex * in, gr_complex * out, int size){
        int ii = 0; // input index
        int oo = 0; // output index

        double s, f; 
        int incr; 
        while(oo < size) {
            out[oo++] = d_inter.interpolate(&in[ii], d_samp_phase);

            s = d_samp_phase + d_samp_inc_remainder + d_samp_inc_correction + 1;
            f = floor(s);
            incr = (int)f;
            d_samp_phase = s - f;
            ii += incr;
        }

        // return how many inputs we required to generate d_cp_length+d_fft_length outputs 
        return ii;
    }

    int
    sampling_synchronization_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {

    	const gr_complex *in = (const gr_complex *) input_items[0];
        gr_complex *out = (gr_complex *) output_items[0];

        d_in_conj = new gr_complex[noutput_items]; 
        volk_32fc_conjugate_32fc(&d_in_conj[0], &in[0], noutput_items);
        
        if(d_dist(d_gen)<d_proba_of_updating){
            for (int i=0; i<noutput_items; i++){
                // I calculate the correlation between the current sample and the past samples, 
                // and update the historic values
                volk_32fc_s32fc_multiply_32fc(&d_current_corr[0], &in[i+d_Htotal-d_max_deviation_px], d_in_conj[i]*d_alpha_corr, 2*d_max_deviation_px+1);

                volk_32fc_s32fc_multiply_32fc(&d_historic_corr[0], &d_historic_corr[0], (1-d_alpha_corr), 2*d_max_deviation_px+1);
                volk_32fc_x2_add_32fc(&d_historic_corr[0], &d_historic_corr[0], &d_current_corr[0], 2*d_max_deviation_px+1);
                volk_32fc_magnitude_squared_32f(&d_abs_historic_corr[0], &d_historic_corr[0], 2*d_max_deviation_px+1);

                //for (int dev=0; dev<2*d_max_deviation_px+1; dev++)
                //    printf("d_current_corr[%i]=%f+j %f\n", dev, std::real(d_current_corr[dev]), std::imag(d_current_corr[dev]));
                update_interpolation_ratio(d_abs_historic_corr, 2*d_max_deviation_px+1);
            }
        }

        //printf("d_samp_inc: %f\n", d_samp_inc);

        int required_for_interpolation = 0;
        required_for_interpolation = interpolate_input(&in[0], &out[0],noutput_items);
        
        // Tell runtime system how many input items we consumed on
        // each input stream.
        consume_each (required_for_interpolation);

        // Tell runtime system how many output items we produced.
        return noutput_items;


    }

  } /* namespace tempest */
} /* namespace gr */


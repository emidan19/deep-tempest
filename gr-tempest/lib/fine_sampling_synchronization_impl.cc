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
#include "fine_sampling_synchronization_impl.h"
#include <volk/volk.h>
#include <random>

namespace gr {
    namespace tempest {

        fine_sampling_synchronization::sptr
            fine_sampling_synchronization::make(int Htotal, int Vtotal, int correct_sampling, float max_deviation, float update_proba)
            {
                return gnuradio::get_initial_sptr
                    (new fine_sampling_synchronization_impl(Htotal, Vtotal, correct_sampling, max_deviation, update_proba));
            }

        /*
         * The private constructor
         */
        fine_sampling_synchronization_impl::fine_sampling_synchronization_impl(int Htotal, int Vtotal, int correct_sampling, float max_deviation, float update_proba)
            : gr::block("fine_sampling_synchronization",
                    gr::io_signature::make(1, 1, sizeof(gr_complex)),
                    gr::io_signature::make(1, 1, sizeof(gr_complex))),
            d_inter(gr::filter::mmse_fir_interpolator_cc()),
            d_dist(update_proba),
            d_gen(std::random_device{}())
        {
            set_relative_rate(1);

            d_correct_sampling = correct_sampling; 
            d_proba_of_updating = update_proba;

            d_max_deviation = max_deviation;
            set_Htotal_Vtotal(Htotal, Vtotal);

            d_alpha_samp_inc = 1e-1;
            
            d_samp_phase = 0; 
            d_alpha_corr = 1e-6; 

            //VOLK alignment as recommended by GNU Radio's Manual. It has a similar effect 
            //than set_output_multiple(), thus we will generally get multiples of this value
            //as noutput_items. 
            const int alignment_multiple = volk_get_alignment() / sizeof(gr_complex);
            set_alignment(std::max(1, alignment_multiple));


            //set_output_multiple(d_Htotal);

        }

        /*
         * Our virtual destructor.
         */
        fine_sampling_synchronization_impl::~fine_sampling_synchronization_impl()
        {
            delete [] d_current_line_corr;
            delete [] d_historic_line_corr;
            delete [] d_abs_historic_line_corr;
            delete [] d_current_frame_corr;
            delete [] d_historic_frame_corr;
            delete [] d_abs_historic_frame_corr;
        }

        void
            fine_sampling_synchronization_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
            {
                int ninputs = ninput_items_required.size ();
                // make sure we receive at least Hsize+max_deviation+taps_to_interpolate
                for (int i = 0; i < ninputs; i++)
                {
                    //ninput_items_required[i] = (int)ceil((noutput_items + 1) * (1+d_samp_inc)) + d_inter.ntaps() + d_Htotal+d_max_deviation_px ;
                    ninput_items_required[i] = (int)ceil((noutput_items + 1) * (2+d_samp_inc_rem)) + d_inter.ntaps() ;
                }

            }

        void fine_sampling_synchronization_impl::set_Htotal_Vtotal(int Htotal, int Vtotal){
            // If the resolution's changed, I reset the whole block
            
            d_Htotal = Htotal; 
            d_Vtotal = Vtotal; 
            //d_max_deviation = max_deviation; 
            d_max_deviation_px = (int)std::ceil(d_Htotal*d_max_deviation);
            printf("d_max_deviation_px: %i\n", d_max_deviation_px);
            //set_history(d_Vtotal*d_Htotal+2*d_max_deviation_px+2);
            set_history(d_Vtotal*(d_Htotal+d_max_deviation_px)+1);

            d_peak_line_index = 0;
            d_samp_inc_rem = 0;
            d_new_interpolation_ratio_rem = 0;

            /** @fcarraustewart: FIXME > Memory leak when using this as a callback ? 
             * in the context where we already constructed our block
             * invoking this function would leave 6 dangling pointers right here, we 
             * should somehow memory manage this correctly, on first thought:
             * */
            /**
             * < Calling delete for each of the following pointers 
             * and assigning them nullptr right here. Before running the new operator.
             * 
             * */

            // d_current_line_corr[i] and derivatives will keep the correlation between pixels 
            // px[t] and px[t+Htotal+i]
            d_current_line_corr = new gr_complex[2*d_max_deviation_px + 1];
            d_historic_line_corr = new gr_complex[2*d_max_deviation_px + 1];
            d_abs_historic_line_corr = new float[2*d_max_deviation_px + 1];

            // d_current_frame_corr[i] and derivatives will keep the correlation between pixels 
            // px[t] and px[t+Htotal*Vtotal+i]. Since a single pixel de-alignment with the next line will 
            // mean d_Vtotal pixels de-alignments with the next frame, these arrays are much bigger. 
            // However, instead of always calculating the whole of them, I'll only calculate around those
            // indicated by the max in the d_abs_historic_line_corr. 
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

            printf("[TEMPEST] Setting Htotal to %i and Vtotal to %i in fine sampling synchronization block.\n", Htotal, Vtotal);

        }

        int fine_sampling_synchronization_impl::interpolate_input(const gr_complex * in, gr_complex * out, int size){
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


        void fine_sampling_synchronization_impl::estimate_peak_line_index(const gr_complex * in, int in_size)
        {

            // TODO a proper programmer would have done this repeated piece of code in a separate function. too lazy...
            gr_complex * d_in_conj = new gr_complex[in_size]; 
            volk_32fc_conjugate_32fc(&d_in_conj[0], &in[0], in_size);

            for (int i=0; i<in_size; i++){
                //d_current_corr[:] = in[i+H*V-max : i+H*V+max]*conj(in[i])
                volk_32fc_s32fc_multiply_32fc(&d_current_line_corr[0], &in[i+d_Htotal-d_max_deviation_px], d_in_conj[i], 2*d_max_deviation_px+1);
                volk_32fc_s32fc_multiply_32fc(&d_historic_line_corr[0], &d_historic_line_corr[0], (1-d_alpha_corr), 2*d_max_deviation_px+1);
                volk_32fc_x2_add_32fc(&d_historic_line_corr[0], &d_historic_line_corr[0], &d_current_line_corr[0], 2*d_max_deviation_px+1);
                volk_32fc_magnitude_squared_32f(&d_abs_historic_line_corr[0], &d_historic_line_corr[0], 2*d_max_deviation_px+1);
            }
            uint16_t peak_index = 0;
            uint32_t d_datain_length = (uint32_t)(2*d_max_deviation_px+1);
            volk_32f_index_max_16u(&peak_index, d_abs_historic_line_corr, 2*d_max_deviation_px+1); 

            d_peak_line_index = (peak_index-d_max_deviation_px);
            delete [] d_in_conj;

        }


        void fine_sampling_synchronization_impl::update_interpolation_ratio(const gr_complex * in, int in_size)
        {
            gr_complex * d_in_conj = new gr_complex[in_size]; 
            volk_32fc_conjugate_32fc(&d_in_conj[0], &in[0], in_size);

            int corrsize = 2*d_Vtotal+1;
            int offset = (d_max_deviation_px+d_peak_line_index)*d_Vtotal;
            int offset_in = d_peak_line_index*d_Vtotal;
            for (int i=0; i<in_size; i++){
                //d_current_corr[offset:offset+corrsize] = in[i+H*V+d_peak_line_index*V - V: i+H*V+d_peak_line_index*V + V]*conj(in[i])
                volk_32fc_s32fc_multiply_32fc(&d_current_frame_corr[offset], &in[i+d_Htotal*d_Vtotal+offset_in-d_Vtotal], d_in_conj[i], corrsize);
                volk_32fc_s32fc_multiply_32fc(&d_historic_frame_corr[offset], &d_historic_frame_corr[offset], (1-d_alpha_corr), corrsize);
                volk_32fc_x2_add_32fc(&d_historic_frame_corr[offset], &d_historic_frame_corr[offset], &d_current_frame_corr[offset], corrsize);
                volk_32fc_magnitude_squared_32f(&d_abs_historic_frame_corr[offset], &d_historic_frame_corr[offset], corrsize);
            }
            uint16_t peak_index = 0;
            uint32_t d_datain_length = (uint32_t)(corrsize);
            volk_32f_index_max_16u(&peak_index, &d_abs_historic_frame_corr[offset], corrsize); 

            // the new interpolation ratio is how far the peak is from d_Vtotal*d_Htotal.
            d_new_interpolation_ratio_rem = ((double)(peak_index+offset_in-d_Vtotal))/(double)(d_Vtotal*d_Htotal);

            //printf("d_peak_line_index: %i, peak_index: %i\n", d_peak_line_index, peak_index-d_Vtotal);
            delete [] d_in_conj;
            
        }

        int
            fine_sampling_synchronization_impl::general_work (int noutput_items,
                    gr_vector_int &ninput_items,
                    gr_vector_const_void_star &input_items,
                    gr_vector_void_star &output_items)
            {
                const gr_complex *in = (const gr_complex *) input_items[0];
                gr_complex *out = (gr_complex *) output_items[0];


                //if(d_dist(d_gen)<d_proba_of_updating){
                d_next_update -= noutput_items;
                if(d_next_update <= 0){
                    estimate_peak_line_index(in, noutput_items);
                    // If noutput_items is too big, I only use a single line
                    //update_interpolation_ratio(in, std::min(noutput_items,d_Htotal));
                    update_interpolation_ratio(in, noutput_items);

                    if (d_next_update<=-10*d_Htotal){
                        d_next_update = d_dist(d_gen);
                    }
                }
                int required_for_interpolation = noutput_items; 
                
                //printf("d_next_update: %i\n",d_next_update);
                if (d_correct_sampling){
                    d_samp_inc_rem = (1-d_alpha_samp_inc)*d_samp_inc_rem + d_alpha_samp_inc*d_new_interpolation_ratio_rem;
                    // d_samp_inc_rem = d_samp_inc_rem - d_alpha_samp_inc*(d_samp_inc_rem+1 - new_interpolation_ratio);
                    required_for_interpolation = interpolate_input(&in[0], &out[0], noutput_items);
                }
                else
                {
                    memcpy(&out[0], &in[0], noutput_items*sizeof(gr_complex));
                }

                //memcpy(out, in, noutput_items*sizeof(gr_complex));

                // Tell runtime system how many input items we consumed on
                // each input stream.
                consume_each (required_for_interpolation);

                // Tell runtime system how many output items we produced.
                return noutput_items;
            }

    } /* namespace tempest */
} /* namespace gr */



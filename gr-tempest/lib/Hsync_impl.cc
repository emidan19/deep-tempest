/* -*- c++ -*- */
/* 
* Copyright 2020
*   Federico "Larroca" La Rocca <flarroca@fing.edu.uy>
*   Adapted to 3.10 by Santiago Fernandez <santiago.fernandez.rovira@fing.edu.uy>
* 
*   Instituto de Ingenieria Electrica, Facultad de Ingenieria,
*   Universidad de la Republica, Uruguay.
* 
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "Hsync_impl.h"
#include <volk/volk.h>
#include <math.h>
#include <gnuradio/math.h>


namespace gr {
    namespace tempest {

        Hsync::sptr
            Hsync::make(int Htotal, int delay)
            {
                return gnuradio::get_initial_sptr
                    (new Hsync_impl(Htotal, delay));
                //return
               //gnuradio::make_block_sptr<Hsync>(Htotal, delay);
            }

        /*
         * The private constructor
         */
        Hsync_impl::Hsync_impl(int Htotal, int delay)
            : gr::block("Hsync",
                    gr::io_signature::make(1, 1, sizeof(gr_complex)),
                    gr::io_signature::make(1, 1, sizeof(gr_complex)))
        {
            set_relative_rate(1);
            set_Htotal_and_delay(Htotal, delay);

            //VOLK alignment as recommended by GNU Radio's Manual. It has a similar effect 
            //than set_output_multiple(), thus we will generally get multiples of this value
            //as noutput_items. 
            const int alignment_multiple = volk_get_alignment() / sizeof(gr_complex);
            set_alignment(std::max(1, alignment_multiple));

           
           //d_peak_epsilon = 0;
            //peak_detect_init(0.2, 0.25, 30, 0.0005);
            // peak_detect_init(0.8, 0.9, 30, 0.9);
            peak_detect_init(0.5, 0.9);
            

        }

        /*
         * Our virtual destructor.
         */
        Hsync_impl::~Hsync_impl()
        {
            delete [] d_corr;
            delete [] d_abs_corr;
        }

        void Hsync_impl::set_Htotal_and_delay(int Htotal, int delay){

            d_delay = delay;
            d_Htotal = Htotal;

            printf("[TEMPEST] Setting Htotal to %i and delay to %i in Hsync block.\n",Htotal, delay);
            
            d_consecutive_aligns = 0;
            d_line_locked = 0;

            //I'll generate complete lines per call to the block
            set_output_multiple(d_Htotal);

            d_corr = new gr_complex[d_Htotal + d_delay];
            if (d_corr == NULL)
                std::cout << "cannot allocate memory: d_corr" << std::endl;
            d_abs_corr = new float[d_Htotal + d_delay];
            if (d_abs_corr == NULL)
                std::cout << "cannot allocate memory: d_abs_corr" << std::endl;
 
            d_consecutive_aligns = 0;
            d_consecutive_aligns_threshold = 10;
            d_shorter_range_size = d_Htotal/50;
            d_max_aligns = d_Htotal/10;
 
        }

        void 
            Hsync_impl::peak_detect_init(float threshold_factor_rise, float alpha)
            {
                d_avg_alpha = alpha;
                d_threshold_factor_rise = threshold_factor_rise;
                d_avg_max = - (float)INFINITY;
                d_avg_min =   (float)INFINITY;
            }


        int 
            Hsync_impl::peak_detect_process(const float * datain, const int datain_length, int * peak_pos)
            {
                uint16_t peak_index = 0;
                uint32_t d_datain_length = (uint32_t)datain_length;
                bool success = true;

                volk_32f_index_max_16u(&peak_index, &datain[0], d_datain_length); 

                if (datain_length>=d_Htotal){
                    float min = datain[(peak_index + d_delay/2) % d_Htotal];
                    if(d_avg_min==(float)INFINITY){
                        d_avg_min = min;
                    }
                    else 
                    {
                        d_avg_min = d_avg_alpha*min + (1-d_avg_alpha)*d_avg_min;
                    }

                }

                if (d_avg_max==-(float)INFINITY)
                {
                    // I initialize the d_avg with the first value. 
                    d_avg_max = datain[peak_index];
                }
                else if ( datain[ peak_index ] > d_avg_max - d_threshold_factor_rise*(d_avg_max-d_avg_min) ) 
                {
                    d_avg_max = d_avg_alpha * datain[ peak_index ] + (1 - d_avg_alpha) * d_avg_max;
                }
                else
                {
                    // if the peak is not large enough, we declare this a failure
                    success = false; 
                }

                //printf("d_avg_max: %f\n",d_avg_max);
                //printf("d_avg_min: %f\n",d_avg_min);
                //printf("datain[%i]: %f\n",peak_index, datain[peak_index]);
                //printf("sucess: %d\n",success);
                
                //We now check whether peaks are also present at multiples of d_delay. This may 
                //happen with border that by chance are at the same distance as the d_delay. In this
                //case we will ignore the result of this peak_process, as we don't know whether samples 
                //were lost (and the line has actually changed). 
                //for(int i=(peak_index % d_delay); i<datain_length; i=i+d_delay){
                //    if((datain[ i ] > d_avg_max - d_threshold_factor_rise*(d_avg_max-d_avg_min)) && i!=peak_index  ){
                //        // i.e. it could also be a peak according to our criteria, we thus declare this a failure
                //        success = false;
                //    }
                //}


                if (success){
                    *peak_pos = peak_index;
                }
                return (success);

            }

        int
            Hsync_impl::max_corr_sync(const gr_complex * in, int lookup_start, int lookup_stop, int * max_corr_pos)
            {

                assert(lookup_start >= lookup_stop);
                //assert(lookup_stop >= (d_Htotal - 1));

                int size;

                // I calculate the 1-sample correlation
                size = lookup_start - lookup_stop;
                //volk_32fc_x2_multiply_conjugate_32fc(&d_corr[0], &in[lookup_start], &in[lookup_start - d_delay], size);
                volk_32fc_x2_multiply_conjugate_32fc(&d_corr[0], &in[lookup_stop+d_delay], &in[lookup_stop], size);

                // I calculate its magnitude
                size = lookup_start - lookup_stop;
                volk_32fc_magnitude_32f(&d_abs_corr[0], &d_corr[0], size);

                int peak = 0;
                bool found_peak = true; 

                // Find peaks of lambda
                // We have found an end of symbol at peak_pos[0] + CP + FFT
                if ((found_peak = peak_detect_process(&d_abs_corr[0], (lookup_start - lookup_stop), &peak)))
                {
                    *max_corr_pos = peak + lookup_stop;
                }
                
                //found_peak = peak_detect_process(&d_abs_corr[0], (lookup_start - lookup_stop), &peak);
                //*max_corr_pos = peak + lookup_stop;

                
                return (found_peak);


           }


        void
            Hsync_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
            {
                int ninputs = ninput_items_required.size ();
                // make sure we receive at least (d_delay + dHtotal + 1)
                for (int i = 0; i < ninputs; i++)
                {
                    ninput_items_required[i] = ( d_delay + 2*d_Htotal + 1) * (noutput_items/d_Htotal +1) ;
                    //printf("division: %i\n",noutput_items/d_Htotal);
                    //printf("noutput_items: %i\n",noutput_items);      
                    //printf("ninput_items_required[%i]: %i\n",i,ninput_items_required[i]);      
                }
            }

        int
            Hsync_impl::general_work (int noutput_items,
                    gr_vector_int &ninput_items,
                    gr_vector_const_void_star &input_items,
                    gr_vector_void_star &output_items)
            {
                const gr_complex *in = (const gr_complex *) input_items[0];
                gr_complex *out = (gr_complex *) output_items[0];

                int low, size;

                d_consumed = 0;
                d_out = 0;

                //printf("is_unaligned():%s\n",is_unaligned() ? "True":"False");
                //

                for (int line = 0; line < noutput_items/d_Htotal ; line++) {
                    //////////////////////
                    // NOW I'M TESTING THE SIMPEST ALGORITHM
                    //////////////////////
                    //d_line_locked = false;
                    //printf("d_consecutive_aligns: %i\n",d_consecutive_aligns);      
                    if (!d_line_locked)
                    {
                        // If we are here it means that we have no idea where the max_corr may be. We thus 
                        // search it thoroughly

                        d_line_found = max_corr_sync(&in[d_consumed], d_Htotal+d_delay, d_delay, &d_line_start);
                        d_line_locked = d_line_found; 
                        
                        //printf("d_line_locked (no idea): %i\n",d_line_locked);
                        //printf("d_line_start (no idea): %i\n",d_line_start);      
                    }
                    else
                    {
                        //If we are here it means that we are sure where the line is, and 
                        //now thus only search near it. 

                        
                        d_line_found = max_corr_sync(&in[d_consumed], d_line_start + d_shorter_range_size, std::max(d_line_start - d_shorter_range_size, 0), &d_line_start);
                        if(!d_line_found){
                        //printf("d_line_found (search near): %i\n",d_line_found);      
                        //printf("d_line_start (search near): %i\n",d_line_start);      
                        }
                        
                        if (d_line_found)
                        {
                            d_consecutive_aligns = std::min(d_max_aligns,d_consecutive_aligns + 1);
                        }
                        else
                        {
                            d_consecutive_aligns = std::max(0,d_consecutive_aligns - 1);
                            // We may have not found the max corr because the smaller search range was too small. It may 
                            // happen either because we lost samples (in which case we should search for the line more 
                            // thoroughly), or because in this particular line the correlation signal was too small.
                            // To avoid confusing these two, we will only change the line position when several of 
                            // these situations happen in a row. 

                            if(d_consecutive_aligns<d_consecutive_aligns_threshold){ 
                                d_line_found = max_corr_sync(&in[d_consumed], d_Htotal+d_delay, d_delay, &d_line_start );
                                
                                // since we are no longer sure where the line starts, I change this to false
                                d_line_locked = false;
                                d_consecutive_aligns = 0;

                            }
                        }
                    }


                    memcpy(&out[line*d_Htotal], &in[d_consumed+d_line_start], d_Htotal*sizeof(gr_complex));

                    d_out = d_out + d_Htotal; 

                    d_consumed += std::max(d_line_start,1);
                    //d_consumed += d_Htotal;
                }

                // Tell runtime system how many input items we consumed on
                // each input stream.
                consume_each(d_consumed);

                // Tell runtime system how many output items we produced.
                return (d_out);
            }


    } /* namespace tempest */
} /* namespace gr */


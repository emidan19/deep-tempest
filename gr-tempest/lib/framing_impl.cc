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
#include "framing_impl.h"

namespace gr {
  namespace tempest {

    framing::sptr
    framing::make(int Htotal, int Vtotal, int Hdisplay, int Vdisplay)
    {
      return gnuradio::get_initial_sptr
        (new framing_impl(Htotal, Vtotal, Hdisplay, Vdisplay));
    }

    /*
     * The private constructor
     */
    framing_impl::framing_impl(int Htotal, int Vtotal, int Hdisplay, int Vdisplay)
      : gr::block("framing",
              gr::io_signature::make(1, 1, sizeof(float)),
              gr::io_signature::make(1, 1, sizeof(float)))
    {
        d_Htotal = Htotal; 
        d_Vtotal = Vtotal; 
        d_Hdisplay = Hdisplay; 
        d_Vdisplay = Vdisplay; 
        d_zeros = new float[d_Hdisplay];
        for (int i=0; i<d_Hdisplay; i++){
            d_zeros[i] = 0;
        }

        d_current_line = 0;

        set_output_multiple(Hdisplay);
    }

    /*
     * Our virtual destructor.
     */
    framing_impl::~framing_impl()
    {
    }


    void framing_impl::set_Htotal_and_Vtotal(int Htotal, int Vtotal){

        d_Htotal = Htotal;
        d_Vtotal = Vtotal;
        d_current_line = 0;
        
        printf("[TEMPEST] Setting Htotal to %i and Vtotal to %i in Framing block.\n",Htotal, Vtotal);

    }

    void
    framing_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
        int ninputs = ninput_items_required.size ();
        // make sure we receive at least a complete line to output
        for (int i = 0; i < ninputs; i++)
        {
            ninput_items_required[i] = d_Htotal*(noutput_items/d_Hdisplay);
       }

    }

    int
    framing_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
        const float *in = (const float *) input_items[0];
        float *out = (float *) output_items[0];

        int d_out = 0;
        int d_consumed = 0;

        for(int line=0; line<noutput_items/d_Hdisplay; line++){
            if(d_current_line<d_Vdisplay){
                // This is a line I should copy
                d_out = d_out + d_Hdisplay;

                if (d_current_line<d_Vtotal){
                    memcpy(&out[line*d_Hdisplay], &in[line*d_Htotal], std::min(d_Hdisplay,d_Htotal)*sizeof(float)); 
                }
                else{
                    memcpy(&out[line*d_Hdisplay], &d_zeros[0], d_Hdisplay*sizeof(float)); 
                }

            }
            if(d_current_line<d_Vtotal){
                // these are lines correspoding to an input
                d_consumed += d_Htotal;
            }
            //printf("line: %i, d_current_line: %i, d_Vdisplay: %i, d_Hdisplay: %i, d_Htotal, %i, d_consumed: %i, noutput_items: %i, d_out: %i\n", line, d_current_line, d_Vdisplay, d_Hdisplay, d_Htotal, d_consumed, noutput_items, d_out);

            d_current_line = (d_current_line+1)%std::max(d_Vdisplay,d_Vtotal);
        }

        // Do <+signal processing+>
        // Tell runtime system how many input items we consumed on
        // each input stream.
        consume_each (d_consumed);

        // Tell runtime system how many output items we produced.
        return d_out;
    }

  } /* namespace tempest */
} /* namespace gr */


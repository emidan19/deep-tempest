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

#ifndef INCLUDED_TEMPEST_FRAMING_IMPL_H
#define INCLUDED_TEMPEST_FRAMING_IMPL_H

#include <tempest/framing.h>

namespace gr {
  namespace tempest {

    class framing_impl : public framing
    {
     private:
         int d_Htotal; 
         int d_Vtotal; 
         int d_Hdisplay; 
         int d_Vdisplay; 
         int d_current_line; 
         float * d_zeros;

     public:
      framing_impl(int Htotal, int Vtotal, int Hdisplay, int Vdisplay);
      ~framing_impl();

      void set_Htotal_and_Vtotal(int Htotal, int Vtotal);

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
    };

  } // namespace tempest
} // namespace gr

#endif /* INCLUDED_TEMPEST_FRAMING_IMPL_H */


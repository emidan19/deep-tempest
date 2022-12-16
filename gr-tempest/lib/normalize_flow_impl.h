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

#ifndef INCLUDED_TEMPEST_NORMALIZE_FLOW_IMPL_H
#define INCLUDED_TEMPEST_NORMALIZE_FLOW_IMPL_H

#include <gnuradio/tempest/normalize_flow.h>
#include <random>

namespace gr {
  namespace tempest {

    class normalize_flow_impl : public normalize_flow
    {
     private:
         float d_min; 
         float d_max; 
         int d_win; 
         float d_alpha_avg; 

         float d_current_max;
         float d_current_min;

         float * d_minus_datain; 

         float compute_max(const float * datain, const int datain_length);

         // to accelerate the process, I'll only update the interpolation
         // once every a random number of iterations with probability d_portion
         std::uniform_real_distribution<float> d_dist;
         std::minstd_rand d_gen;
         float d_proba_of_updating;

     public:
      normalize_flow_impl(float min, float max, int window, float alpha_avg, float update_proba);
      ~normalize_flow_impl();

      // Where all the action really happens
      int work(int noutput_items,
         gr_vector_const_void_star &input_items,
         gr_vector_void_star &output_items);

      void set_min_max(float min, float max);

    };

  } // namespace tempest
} // namespace gr

#endif /* INCLUDED_TEMPEST_NORMALIZE_FLOW_IMPL_H */


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

#ifndef INCLUDED_TEMPEST_FRAME_DROP_H
#define INCLUDED_TEMPEST_FRAME_DROP_H

#include <gnuradio/tempest/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace tempest {

    /*!
     * \brief <+description of block+>
     * \ingroup tempest
     *
     */
    class TEMPEST_API frame_drop : virtual public gr::block
    {
     public:
      typedef std::shared_ptr<frame_drop> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of tempest::frame_drop.
       *
       * To avoid accidental use of raw pointers, tempest::frame_drop's
       * constructor is in a private implementation
       * class. tempest::frame_drop::make is the public interface for
       * creating new instances.
       */
      static sptr make(int Htotal, int Vtotal, int correct_sampling, float max_deviation, float update_proba, double actual_samp_rate);
    };

  } // namespace tempest
} // namespace gr

#endif /* INCLUDED_TEMPEST_FRAME_DROP_H */


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

#ifndef INCLUDED_TEMPEST_NORMALIZE_FLOW_H
#define INCLUDED_TEMPEST_NORMALIZE_FLOW_H

#include <gnuradio/tempest/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
  namespace tempest {

    /*!
     * \brief Normalizes a flow between min and max. To be less CPU-intensive, it randomly chooses when to update the signal's maximum.
     * \ingroup tempest
     *
     */
    class TEMPEST_API normalize_flow : virtual public gr::sync_block
    {
     public:
      typedef std::shared_ptr<normalize_flow> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of tempest::normalize_flow.
       *
       * To avoid accidental use of raw pointers, tempest::normalize_flow's
       * constructor is in a private implementation
       * class. tempest::normalize_flow::make is the public interface for
       * creating new instances.
       */
      static sptr make(float min, float max, int window, float alpha_avg, float update_proba);
      
      virtual void set_min_max(float min, float max) = 0;
    };

  } // namespace tempest
} // namespace gr

#endif /* INCLUDED_TEMPEST_NORMALIZE_FLOW_H */


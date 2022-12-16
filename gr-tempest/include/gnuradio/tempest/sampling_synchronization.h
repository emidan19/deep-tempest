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

#ifndef INCLUDED_TEMPEST_SAMPLING_SYNCHRONIZATION_H
#define INCLUDED_TEMPEST_SAMPLING_SYNCHRONIZATION_H

#include <gnuradio/tempest/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace tempest {

    /*!
     * \brief It correlates the signal and estimates what's the most probable line length. It then corrects the sampling rate so that the line length's is Htotal.  
     * \ingroup tempest
     *
     */
    class TEMPEST_API sampling_synchronization : virtual public gr::block
    {
     public:
      typedef std::shared_ptr<sampling_synchronization> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of tempest::sampling_synchronization.
       *
       * To avoid accidental use of raw pointers, tempest::sampling_synchronization's
       * constructor is in a private implementation
       * class. tempest::sampling_synchronization::make is the public interface for
       * creating new instances.
       */
      static sptr make(int Htotal, double manual_correction);

      virtual void set_Htotal(int Htotal) = 0;

      virtual void set_manual_correction(double correction) = 0;
 
    };

  } // namespace tempest
} // namespace gr

#endif /* INCLUDED_TEMPEST_SAMPLING_SYNCHRONIZATION_H */


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

#ifndef INCLUDED_TEMPEST_FFT_PEAK_FINE_SAMPLING_SYNC_H
#define INCLUDED_TEMPEST_FFT_PEAK_FINE_SAMPLING_SYNC_H

#include <tempest/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace tempest {

    /*!
     * \brief <+description of block+>
     * \ingroup tempest
     *
     */
    class TEMPEST_API fft_peak_fine_sampling_sync : virtual public gr::block
    {
     public:
      typedef boost::shared_ptr<fft_peak_fine_sampling_sync> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of tempest::fft_peak_fine_sampling_sync.
       *
       * To avoid accidental use of raw pointers, tempest::fft_peak_fine_sampling_sync's
       * constructor is in a private implementation
       * class. tempest::fft_peak_fine_sampling_sync::make is the public interface for
       * creating new instances.
       */
      static sptr make(int sample_rate, int size, int refresh_rate, int Vvisible, int Hvisible, bool automatic_mode);
    };

  } // namespace tempest
} // namespace gr

#endif /* INCLUDED_TEMPEST_FFT_PEAK_FINE_SAMPLING_SYNC_H */


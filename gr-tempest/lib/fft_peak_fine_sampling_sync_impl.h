/* -*- c++ -*- */
/**
 * Copyright 2021
 *    Pablo Bertrand    <pablo.bertrand@fing.edu.uy>
 *    Felipe Carrau     <felipe.carrau@fing.edu.uy>
 *    Victoria Severi   <maria.severi@fing.edu.uy>
 *    
 *    Instituto de Ingeniería Eléctrica, Facultad de Ingeniería,
 *    Universidad de la República, Uruguay.
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
 * @file fft_peak_fine_sampling_sync_impl.h
 * 
 * @brief Block that uses the signal autocorrelation to
 * continuously calculate the interpolation ratio required,
 * for as long as it takes to properly correct sampling.
 *
 * gr-tempest
 *
 * @date September 19, 2021
 * @author  Pablo Bertrand   <pablo.bertrand@fing.edu.uy>
 * @author  Felipe Carrau    <felipe.carrau@fing.edu.uy>
 * @author  Victoria Severi  <maria.severi@fing.edu.uy>
 */

/**********************************************************
 * Constant and macro definitions
 **********************************************************/

#ifndef INCLUDED_TEMPEST_FFT_PEAK_FINE_SAMPLING_SYNC_IMPL_H
#define INCLUDED_TEMPEST_FFT_PEAK_FINE_SAMPLING_SYNC_IMPL_H
#define lowpasscoeff 0.1
#define MAX_PERIOD 0.0000284
/**********************************************************
 * Include statements
 **********************************************************/

#include <tempest/fft_peak_fine_sampling_sync.h>

namespace gr {
  namespace tempest {

    class fft_peak_fine_sampling_sync_impl : public fft_peak_fine_sampling_sync
    {
     private:

      /**********************************************************
       * Data declarations
       **********************************************************/

      gr::thread::mutex d_mutex;
      bool d_start_fft_peak_finder = 1;

      //Received parameters
      int d_sample_rate;
      int d_fft_size;


      double d_ratio;
      long double d_accumulator;
      long double d_real_line;
      bool d_start;


      //Search values
      uint32_t d_search_skip;
      uint32_t d_search_margin;
      uint32_t d_vtotal_est;
      uint32_t d_peak_1;
      uint32_t d_peak_2;

      //Results to publish
      float d_refresh_rate;

      //Counters
      uint32_t d_work_counter;
      uint32_t d_sample_counter;

      //Search values
      float d_refresh_rate_est;
      bool d_flag;

      //Resolution results
      int d_Hsize;
      int d_Vsize;
      int d_Vvisible;
      int d_Hvisible;
      long d_Hblank;
      long d_Vblank;
      
      void search_table(float fv_estimated)
      {
            if (fv_estimated<65)
            {
              d_refresh_rate=60;
              if (d_vtotal_est<576.5)
              {
                d_Vvisible=480;
                d_Vsize=525;
                d_Hvisible=640;
                d_Hsize=800;
                          }
              if (d_vtotal_est>576.5 && d_vtotal_est<687)
              { 
                //800x600
                d_Hvisible=800;
                d_Hsize=1056;
                d_Vvisible=600;
                d_Vsize=628;
              }
              if (d_vtotal_est>687 && d_vtotal_est<776)
              { 
                //1280x720
                d_Hvisible=1280;
                d_Hsize=1664;
                d_Vvisible=720;
                d_Vsize=746;
              }
              if (d_vtotal_est>776 && d_vtotal_est<869)
              { 
                //1024x768
                d_Hvisible=1024;
                d_Hsize=1344;
                d_Vvisible=768;
                d_Vsize=806;
              }
              if (d_vtotal_est>869 && d_vtotal_est<966)
              { 
                //1600x900
                d_Hvisible=1600;
                d_Hsize=2128;
                d_Vvisible=900;
                d_Vsize=932;
              }
              if (d_vtotal_est>966 && d_vtotal_est<1033)
              { 
                //1280x960
                d_Hvisible=1280;
                d_Hsize=1800;
                d_Vvisible=960;
                d_Vsize=1000;
              }
              if (d_vtotal_est>1033 && d_vtotal_est<1077.5)
              { 
                //1280x1024
                d_Hvisible=1280;
                d_Hsize=1688;
                d_Vvisible=1024;
                d_Vsize=1066;
              }
              if (d_vtotal_est>1077.5 && d_vtotal_est<1107)
              { 
                //1680x1050
                d_Hvisible=1680;
                d_Hsize=2240;
                d_Vvisible=1050;
                d_Vsize=1089;
              }
              if (d_vtotal_est>1107 && d_vtotal_est<1300)
              { 
                //1920x1080
                d_Hvisible=1920;
                d_Hsize=2200;
                d_Vvisible=1080;
                d_Vsize=1125;
              }
            }
            if (fv_estimated>65 && fv_estimated<72.5)
            {
              d_refresh_rate=(int)70.1;
              if (d_vtotal_est>400 && d_vtotal_est<500)
              {
                d_Hvisible=720;
                d_Hsize=900;
                d_Vvisible=400;
                d_Vsize=449;
              }
            }
            if (fv_estimated>72.5 && fv_estimated<80)
            {
              d_refresh_rate=75;
              if (d_vtotal_est<562.5) 
              {
                //640x480
                d_Hvisible=640;
                d_Hsize=840;
                d_Vvisible=480;
                d_Vsize=500;
              }
              if (d_vtotal_est>562.5 && d_vtotal_est<646)
              {
                //800x600
                d_Hvisible=800;
                d_Hsize=1056;
                d_Vvisible=600;
                d_Vsize=625;
              }
              if (d_vtotal_est>646 && d_vtotal_est<733.5)
              {
                //832x624
                d_refresh_rate=(int)74.6;
                d_Hvisible=832;
                d_Hsize=1152;
                d_Vvisible=624;
                d_Vsize=667;
              }
              if (d_vtotal_est>733.5 && d_vtotal_est<850)
              {
                //1024x768
                d_refresh_rate=(int)75.1;
                d_Hvisible=1024;
                d_Hsize=1312;
                d_Vvisible=768;
                d_Vsize=800;
              }
              if (d_vtotal_est>850 && d_vtotal_est<983)
              {
                //1152x864
                d_Hvisible=1152;
                d_Hsize=1600;
                d_Vvisible=864;
                d_Vsize=900;
              }
              if (d_vtotal_est>983 && d_vtotal_est<1250)
              {
                //1280x1024
                d_Hvisible=1280;
                d_Hsize=1688;
                d_Vvisible=1024;
                d_Vsize=1066;
              }
            }
      };

      /**********************************************************
       * Private function prototypes
       **********************************************************/
      /**
        * @brief Function that processes the enable received as PMT
        * message from other blocks and assignes it to a variable.
        *  
        * @param pmt_t msg: Message received from autocorrelation
        * block during runtime.
        */
      void set_ena_msg(pmt::pmt_t msg);
      //---------------------------------------------------------
     public:
      fft_peak_fine_sampling_sync_impl(int sample_rate, int size, int refresh_rate, int Vvisible, int Hvisible, bool automatic_mode);
      ~fft_peak_fine_sampling_sync_impl();

      /**********************************************************
       * Public function prototypes
       **********************************************************/
      /**
        * @brief Used to establish the amount of samples required
        * for a full work iteration.
        *  
        */
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);
      //---------------------------------------------------------
      /**
        * @brief Main function that carries out the peak searches
        * in the autocorrelation signal and uses the results to
        * calculate the interpolation ratio required and prints 
        * it to a PMT port for other blocks to use it as it is 
        * updated in runtime.
        *  
        */
      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
      //---------------------------------------------------------
    };

  } // namespace tempest
} // namespace gr

#endif /* INCLUDED_TEMPEST_FFT_PEAK_FINE_SAMPLING_SYNC_IMPL_H */


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

#ifndef INCLUDED_TEMPEST_INFER_SCREEN_RESOLUTION_IMPL_H
#define INCLUDED_TEMPEST_INFER_SCREEN_RESOLUTION_IMPL_H

#include <gnuradio/tempest/infer_screen_resolution.h>
//#include <thread>
#include <volk/volk.h>
#include <gnuradio/math.h>

namespace gr {
  namespace tempest {

    class infer_screen_resolution_impl : public infer_screen_resolution
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
      int d_i;
      
      //Search values
      float d_refresh_rate_est;
      bool d_flag;
      bool d_mode;

      //Resolution results
      int d_Hsize;
      int d_Vsize;
      int d_Vvisible;
      int d_Hvisible;
      long d_Hblank;
      long d_Vblank;
      

      uint32_t calculate_peak_index_relative_to_search_skip(const float *in, uint32_t search_skip, uint32_t search_range)
      {
          uint32_t peak_index;
          volk_32f_index_max_32u(&peak_index, &in[search_skip], search_range);   
          return peak_index += search_skip;                                                
      };

      void search_table(float fv_estimated)
      {
            if (fv_estimated<65 && fv_estimated>55)
            {
              d_refresh_rate=60;
              if (d_vtotal_est<576.5)
              {
                d_Vvisible=480;
                d_Vsize=525;
                d_Hvisible=640;
                d_Hsize=800;
              }
              else if (d_vtotal_est>576.5 && d_vtotal_est<687)
              { 
                //800x600
                d_Hvisible=800;
                d_Hsize=1056;
                d_Vvisible=600;
                d_Vsize=628;
              }
              else if (d_vtotal_est>687 && d_vtotal_est<776)
              { 
                //1280x720
                d_Hvisible=1280;
                d_Hsize=1664;
                d_Vvisible=720;
                d_Vsize=746;
              }
              else if (d_vtotal_est>776 && d_vtotal_est<869)
              { 
                //1024x768
                d_Hvisible=1024;
                d_Hsize=1344;
                d_Vvisible=768;
                d_Vsize=806;
              }
              else if (d_vtotal_est>869 && d_vtotal_est<966)
              { 
                //1600x900
                d_Hvisible=1600;
                d_Hsize=2128;
                d_Vvisible=900;
                d_Vsize=932;
              }
              else if (d_vtotal_est>966 && d_vtotal_est<1033)
              { 
                //1280x960
                d_Hvisible=1280;
                d_Hsize=1800;
                d_Vvisible=960;
                d_Vsize=1000;
              }
              else if (d_vtotal_est>1033 && d_vtotal_est<1077.5)
              { 
                //1280x1024
                d_Hvisible=1280;
                d_Hsize=1688;
                d_Vvisible=1024;
                d_Vsize=1066;
              }
              else if (d_vtotal_est>1077.5 && d_vtotal_est<1107)
              { 
                //1680x1050
                d_Hvisible=1680;
                d_Hsize=2240;
                d_Vvisible=1050;
                d_Vsize=1089;
              }
              else if (d_vtotal_est>1107 && d_vtotal_est<1300)
              { 
                //1920x1080
                d_Hvisible=1920;
                d_Hsize=2200;
                d_Vvisible=1080;
                d_Vsize=1125;
              }
            }
            else if (fv_estimated>65 && fv_estimated<72.5)
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
            else if (fv_estimated>72.5 && fv_estimated<80)
            {
              d_refresh_rate=75;
              if (d_vtotal_est<562) 
              {
                //640x480
                d_Hvisible=640;
                d_Hsize=840;
                d_Vvisible=480;
                d_Vsize=500;
              }
              else if (d_vtotal_est>=562 && d_vtotal_est<646)
              {
                //800x600
                d_Hvisible=800;
                d_Hsize=1056;
                d_Vvisible=600;
                d_Vsize=625;
              }
              else if (d_vtotal_est>=646 && d_vtotal_est<733)
              {
                //832x624
                d_refresh_rate=(int)74.6;
                d_Hvisible=832;
                d_Hsize=1152;
                d_Vvisible=624;
                d_Vsize=667;
              }
              else if (d_vtotal_est>=733 && d_vtotal_est<850)
              {
                //1024x768
                d_refresh_rate=(int)75.1;
                d_Hvisible=1024;
                d_Hsize=1312;
                d_Vvisible=768;
                d_Vsize=800;
              }
              else if (d_vtotal_est>850 && d_vtotal_est<983)
              {
                //1152x864
                d_Hvisible=1152;
                d_Hsize=1600;
                d_Vvisible=864;
                d_Vsize=900;
              }
              else if (d_vtotal_est>983 && d_vtotal_est<1250)
              {
                //1280x1024
                d_Hvisible=1280;
                d_Hsize=1688;
                d_Vvisible=1024;
                d_Vsize=1066;
              }
            }
      };

     public:
      infer_screen_resolution_impl(int sample_rate, int fft_size, float refresh_rate, bool automatic_mode);
      ~infer_screen_resolution_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      void set_refresh_rate(float refresh_rate);
    
      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

    };

  } // namespace tempest
} // namespace gr

#endif /* INCLUDED_TEMPEST_INFER_SCREEN_RESOLUTION_IMPL_H */


/* -*- c++ -*- */
/*
 * Copyright 2021 gr-tempest
 *    Pablo Bertrand    <pablo.bertrand@fing.edu.uy>
 *    Felipe Carrau     <felipe.carrau@fing.edu.uy>
 *    Victoria Severi   <maria.severi@fing.edu.uy>
 *    
 *    Instituto de Ingeniería Eléctrica, Facultad de Ingeniería,
 *    Universidad de la República, Uruguay.
 * 
 * This software is based on Martin Marinov's TempestSDR.
 * In particular, gaussianblur() function is entirely his, and
 * find_best_beta and find_shift are implementations of
 * findbestfit and findthesweetspot, respectively.
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
#include "sync_detector_impl.h"
#include <volk/volk.h>
#include <gnuradio/math.h>
#include <gnuradio/math.h>

namespace gr {
  namespace tempest {

    sync_detector::sptr
    sync_detector::make(int hscreen, int vscreen, int hblanking, int vblanking)
    {
      return gnuradio::get_initial_sptr
        (new sync_detector_impl(hscreen, vscreen, hblanking, vblanking));
    }

    /*
     * The private constructor
     */
    sync_detector_impl::sync_detector_impl(int hscreen, int vscreen, int hblanking, int vblanking)
      : gr::block("sync_detector",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex)))
    {
      //Fixed parameters
      d_LOWPASS_COEFF_V = 0.2;
      d_LOWPASS_COEFF_H = 0.1;
      d_GAUSSIAN_ALPHA = 1.0;

      //Input data
      d_hdisplay = hscreen;
      d_vdisplay = vscreen;
      d_hblanking = hblanking;
      d_vblanking = vblanking;
      d_Htotal = d_hdisplay + d_hblanking;
      d_Vtotal = d_vdisplay + d_vblanking;

      //Blanking variables
      d_blanking_size_h = d_hblanking;
      d_blanking_size_v = d_vblanking;
      d_blanking_index_h = 0;
      d_blanking_index_v = 0;
      d_working_index_h = 0;

      //Flags
      d_frame_average_complete = 0;
      d_frame_wait_for_blanking = 0;
      d_frame_output = 0;
      d_start_sync_detect = 0;

      //Counters
      d_frame_height_counter = 0; 
      d_blanking_wait_counter = 0; 
      d_output_counter = 0;

      //Arrays
      d_data_h = new float[d_Htotal];
      if (d_data_h == NULL)
        std::cout << "cannot allocate memory: d_data_h" << std::endl;
      d_avg_h_line = new float[d_Htotal];
      if (d_avg_h_line == NULL)
        std::cout << "cannot allocate memory: d_data_h" << std::endl;
      d_avg_v_line = new float[d_Vtotal];
      if (d_avg_v_line == NULL)
        std::cout << "cannot allocate memory: d_data_h" << std::endl;

      // PMT ports
      message_port_register_in(pmt::mp("en"));

      // PMT handlers
      set_msg_handler(pmt::mp("en"), [this](const pmt::pmt_t& msg) {sync_detector_impl::set_ena_msg(msg); });
      
      //Complete lines per call to the block will be generated
      set_output_multiple(2*d_Htotal);
      
      // const int alignment_multiple = volk_get_alignment() / sizeof(gr_complex);
      // set_alignment(std::max(1, alignment_multiple));
    }

    /*
     * Our virtual destructor.
     */
    sync_detector_impl::~sync_detector_impl()
    {
      delete [] d_data_h;
      delete [] d_avg_h_line; 
      delete [] d_avg_v_line;
    }


    void sync_detector_impl::find_best_beta (const float *data, const int total_line_size, const double total_sum, const int blanking_size, double *beta, int *beta_index)
    {
    ////////////////////////////////////////////////////////////////////////////////
    // Function that takes a full line (either horizontal or vertical) and a fixed//
    // blanking size to calculate the medium energy difference between blanking   //
    // and screen (beta) for each possible blanking position. Returns the position// 
    // that granted the best beta and the value of beta itself.                   //
    ////////////////////////////////////////////////////////////////////////////////
    
      const double screen_size_double = total_line_size - blanking_size;
      const double blanking_size_double = blanking_size;      
      //Doubles will be used for accuracy, as well as for beta

      int i;
      double curr_sum = 0.0;
      for (i = 0; i < blanking_size; i++) curr_sum += data[i];  

      //Initial calculation of beta (blanking located on i=0)
      const double beta_0 = (total_sum - curr_sum)/screen_size_double - curr_sum/blanking_size_double;
      *beta = pow(beta_0, 2);
      *beta_index = 0;

      const int screen_size = total_line_size - blanking_size;

      //Then the blanking is moved along the line by removing the first element and adding the next
      //In each case, beta and it's position are only saved if the beta was improved
      for (i = 0; i < (total_line_size - 1); i++) {

        const double data_to_remove = data[i];

        const int to_remove_index = (i < screen_size) ? (i+blanking_size) : (i-screen_size);
        const double data_to_add = data[to_remove_index];

        curr_sum = curr_sum - data_to_remove + data_to_add;
        const double beta_i = pow(((total_sum - curr_sum)/screen_size_double - curr_sum/blanking_size_double),2);

        if (beta_i > *beta) {
          *beta = beta_i;
          *beta_index = i;
        }
      }
    }


    float sync_detector_impl::calculate_gauss_coeff(float N, float i) 
    {
      return(expf(-2.0f*d_GAUSSIAN_ALPHA*d_GAUSSIAN_ALPHA*i*i/(N*N)));
    }


    void sync_detector_impl::gaussianblur(float * data, int size) 
    {
    ////////////////////////////////////////////////////////////////////////////////
    // Marinov's gaussian blur application for a fixed-size array.                //
    ////////////////////////////////////////////////////////////////////////////////

      float norm = 0.0f, c_2 = 0.0f, c_1 = 0.0f, c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;
      if (norm == 0.0f) {
        norm = calculate_gauss_coeff(5,-2) + calculate_gauss_coeff(5, -1) + calculate_gauss_coeff(5, 0) + calculate_gauss_coeff(5, 1) + calculate_gauss_coeff(5, 2);
        c_2 = calculate_gauss_coeff(5, -2) / norm;
        c_1 = calculate_gauss_coeff(5, -1) / norm;
        c0 = calculate_gauss_coeff(5, 0) / norm;
        c1 = calculate_gauss_coeff(5, 1) / norm;
        c2 = calculate_gauss_coeff(5, 2) / norm;
      }

      float p_2, p_1, p0, p1, p2, data_2, data_3, data_4;
      if (size < 5) {
        p_2 = data[0];
        p_1 = data[1 % size];
        p0 = data[2 % size];
        p1 = data[3 % size];
        p2 = data[4 % size];
      } else {
        p_2 = data[0];
        p_1 = data[1];
        p0 = data[2];
        p1 = data[3];
        p2 = data[4];
      }

      data_2 = p0;
      data_3 = p1;
      data_4 = p2;

      int i;
      const int sizem2 = size - 2;
      const int sizem5 = size - 5;
      for (i = 0; i < size; i++) {

        const int idtoupdate = (i < sizem2) ? (i + 2) : (i - sizem2);
        const int nexti = (i < sizem5) ? (i + 5) : (i - sizem5);

        data[idtoupdate] = p_2 * c_2 + p_1 * c_1 + p0 * c0 + p1 * c1 + p2 * c2;
        p_2 = p_1;
        p_1 = p0;
        p0 = p1;
        p1 = p2;

        if (nexti < 2 || nexti >= 5)
          p2 = data[nexti];
        else {
          switch (nexti) {
          case 2:
            p2 = data_2;
            break;
          case 3:
            p2 = data_3;
            break;
          case 4:
            p2 = data_4;
            break;
          }
        }
      }
    }

    void sync_detector_impl::find_shift (int *blanking_index, int *blanking_size, float *data, const int total_line_size, int min_blanking_size, double lowpasscoeff)
    {    
    //////////////////////////////////////////////////////////////////////////////
    // Function that takes the line and runs find_best_beta for some possible   //
    // blanking sizes. Then takes the best location (and size) returned and     //
    // defines the new position for shifting using both the new calculation and //
    // the previous information to prevent sudden big changes.                  //
    //////////////////////////////////////////////////////////////////////////////

      gaussianblur(data, total_line_size);
      
      double total_sum = 0.0;
      for (int i = 0; i < total_line_size; i++) total_sum += data[i];
      int beta_index=0;
      double beta;
      int beta_index_temp;
      double beta_temp;

      //Blanking sizes to test are established
      int max_blanking_size = total_line_size >> 1;
      if (min_blanking_size < 1) min_blanking_size = 1;
      if ((*blanking_size) < min_blanking_size) (*blanking_size) = min_blanking_size;
      else if ((*blanking_size) > max_blanking_size) (*blanking_size) = max_blanking_size;
      int blanking_size_attempt[4] = {((*blanking_size)-4), ((*blanking_size)+4), ((*blanking_size)>>1), ((*blanking_size)<<1)};
      
      //First search for beta using the initial blanking size
      find_best_beta(data, total_line_size, total_sum, (*blanking_size), &beta, &beta_index);

      //Then test the rest of the sizes and save the best one found
      for (int i = 0; i < 4; i++) {
        if (blanking_size_attempt[i]>min_blanking_size && blanking_size_attempt[i]<max_blanking_size && blanking_size_attempt[i]!=(*blanking_size)){
          find_best_beta(data, total_line_size, total_sum, blanking_size_attempt[i], &beta_temp, &beta_index_temp);
          if (beta_temp>beta) {
            beta = beta_temp;
            beta_index = beta_index_temp;
            (*blanking_size) = blanking_size_attempt[i];
          }
        }
      }

      const int half_line_size = total_line_size / 2;

      //Here is defined what would be the shift location (position found plus half of the blanking)
      int raw_index = (beta_index + (*blanking_size) /2) % total_line_size;

      //Checking if the new shift differs excessively from the previous one
      const int raw_diff = raw_index - *blanking_index;
            
      if (raw_diff > half_line_size)
        *blanking_index += total_line_size;

      else if (raw_diff < -half_line_size)
        raw_index += total_line_size;

      //The shift is updated considering both new and previous data (with the corresponding weights)
      *blanking_index = ((int) round(raw_index * lowpasscoeff + (1.0 - lowpasscoeff) * (*blanking_index))) % ((int) total_line_size);
    }

    void 
    sync_detector_impl::set_ena_msg(pmt::pmt_t msg)
    {
        gr::thread::scoped_lock l(d_mutex);
        if (pmt::is_bool(msg)) {
            bool en = pmt::to_bool(msg);
            d_start_sync_detect = !en;
            printf("Sync Detecting Start.\n");
        } else {
            GR_LOG_WARN(d_logger,
                        "Sync Detector: Non-PMT type received, expecting Boolean PMT\n");
        }
    }

    void
    sync_detector_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      int ninputs = ninput_items_required.size ();
      for (int i = 0; i < ninputs; i++){
        ninput_items_required[i] = (2*d_Htotal + 1)*(noutput_items/(2*d_Htotal) +1);
      }
    }


    int
    sync_detector_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex *in = (const gr_complex *) input_items[0];
      gr_complex *out = (gr_complex *) output_items[0];
      
      int delta_h, consumed = 0, out_amount = 0;

      gr::thread::scoped_lock l(d_mutex);
      
      if (d_start_sync_detect==0){
          memcpy(out,in,noutput_items*sizeof(gr_complex));
          out_amount=noutput_items;
          consumed=noutput_items;

      } else {

          for (int line = 0; line < noutput_items/d_Htotal; line++) { 
     
            volk_32fc_magnitude_32f(&d_data_h[0], &in[line*d_Htotal], d_Htotal);

            //From an horizontal line, we obtain the partial value of all elements of the
            //horizontal average and the full value of a single element of the vertical average
            for (int i=0 ;   (i < d_Htotal)  ; i++)
            {
                  d_avg_h_line[i] = d_avg_h_line[i] + (d_data_h[i]/(d_Vtotal));    
                  d_avg_v_line[d_frame_height_counter] += (d_data_h[i]/(d_Htotal));
            }

            d_frame_height_counter ++;

            //When a complete frame is evaluated we have full averages and are ready to find the shift
            if(d_frame_height_counter % d_Vtotal == 0)
              d_frame_average_complete = 1;

            if(d_frame_average_complete)
            {
              //Finding the position that maximizes beta both horizontally and vertically for the frame
              find_shift (&d_blanking_index_h, &d_blanking_size_h,  d_avg_h_line, d_Htotal, d_Htotal*0.05f, d_LOWPASS_COEFF_H);
              find_shift (&d_blanking_index_v, &d_blanking_size_v,  d_avg_v_line, d_Vtotal, d_Vtotal*0.0005f, d_LOWPASS_COEFF_V);          

              //As the information is used, we set up the variables to receive the next frame
              d_frame_average_complete = 0;
              d_frame_height_counter = 0;
              d_blanking_wait_counter = 0;

              for (int i=0; (i<d_Htotal); i++)
              {
                     d_avg_h_line[i] = 0;
              }
              for (int i=0; (i<d_Vtotal); i++)
              {
                     d_avg_v_line[i] = 0;
              }

              //Pass on to the state where the next frame's display is given by the found shifts
              d_frame_wait_for_blanking = 1;
            }

            if (d_frame_wait_for_blanking)
            { 
              //Begin the search for the line that provides the vertical shift found
              d_blanking_wait_counter++;

              if(d_blanking_wait_counter == d_blanking_index_v)
              {
                //When found, first reset variables
                d_frame_wait_for_blanking = 0;
             
                //Consume horizontally according to the shift
                delta_h = d_blanking_index_h - d_working_index_h;
                consumed += delta_h;
                d_working_index_h = d_blanking_index_h;
                
                //If the vertical shift has been made and we are not yet printing, we begin
                if (d_frame_output == 0) 
                {
                  d_frame_output = 1;
                } 
                d_blanking_wait_counter = 0;
              }
            }

            if (d_frame_output)
            { 
              //If we are allowed, print a full line beginning at the horizontal shift found
              //It is worth noticing that a full frame is always printed with the same shift
              memcpy(&out[line*d_Htotal], &in[line*d_Htotal + d_working_index_h], d_Htotal*sizeof(gr_complex));
              out_amount = out_amount + d_Htotal;
              d_output_counter++;

              //After a full frame, printing is disabled to allow further vertical sync
              if (d_output_counter == d_Vtotal) {
                d_frame_output = 0;
                d_output_counter = 0;
              }
            }
            //Consuming the regular amount since odd cases have already been considered
            consumed += d_Htotal;
          }

      }

      consume_each (consumed);
      return (out_amount);
    }

  } /* namespace sync_detector */
} /* namespace gr */


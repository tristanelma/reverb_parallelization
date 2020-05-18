#include "AudioFile.h"

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <omp.h>

using namespace std;

int main(int argc, char **argv){
  if(argc != 6){
    printf("Required args: <input_file> <impulse_response_file> <output_file> <wetness> <num_threads>\nNote: <wetness> is a decimal between 0 and 1.\n");
    return 1;
  }
  else if(atof(argv[4]) < 0 || atof(argv[4]) > 1){
    printf("Note: <wetness> is a decimal between 0 and 1.\n");
    return 1;
  }

  int p = atoi(argv[5]);

  //AudioFile helps with low-level audio data processing
  AudioFile<double> inputAF;
  inputAF.load(argv[1]);
  AudioFile<double> impulseAF;
  impulseAF.load(argv[2]);

  printf("|======================================|\n");
  printf("INPUT FILE: %s\n", argv[1]);
  inputAF.printSummary();
  printf("|======================================|\n");
  printf("IMPULSE RESPONSE FILE: %s\n", argv[2]);
  impulseAF.printSummary();

  int num_input_samples = inputAF.getNumSamplesPerChannel();
  int num_impulse_samples = impulseAF.getNumSamplesPerChannel();
  int num_output_samples = num_input_samples + num_impulse_samples - 1;

  int num_input_channels = inputAF.getNumChannels();
  int num_impulse_channels = impulseAF.getNumChannels();
  //Set num_output_channels to max of input and impulse channels
  int num_output_channels = num_input_channels > num_impulse_channels ? num_input_channels : num_impulse_channels;

  AudioFile<double>::AudioBuffer output_buffer;
  output_buffer.resize(num_output_channels);
  for (int i = 0; i < num_output_channels; i++)
  {
    output_buffer[i].resize(num_output_samples);
  }
  //Initialize the output buffer to all 0's
  //Crucial to ensure correctness with multiply-add implementation below
  for (int i = 0; i < num_output_samples; i++)
  {
    for(int channel = 0; channel < num_output_channels; channel++){
      output_buffer[channel][i] = 0;
    }
  }

  //Increase input gain by 2dB
  for(int i = 0; i < num_input_samples; i++){
    inputAF.samples[0][i] = inputAF.samples[0][i] * 2;
    if(inputAF.samples[0][i] >= 1) inputAF.samples[0][i] = .99;
  }

  //Start timer
  struct timespec start, stop;
  double time;
  if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

  //Convolve
  double input_sample;
  int input_channel;
  int impulse_channel;
  int output_channel;
  double sum = 0;
  int i,j;
  omp_set_num_threads(p);
  for(i=0; i<num_output_samples; i++){
    sum = 0;
    //Print progress
    printf("%f\n", float(i)/num_output_samples);
    #pragma omp parallel shared(p, inputAF, impulseAF, output_buffer, num_input_samples, num_impulse_samples, num_input_channels, num_impulse_channels, num_output_channels, i) private(input_channel, impulse_channel, output_channel, input_sample, sum, j)
    {
      #pragma omp for nowait schedule(static, num_impulse_samples/p)
      for(j=0; j<num_impulse_samples; j++){
        for(output_channel = 0; output_channel < num_output_channels; output_channel++){
          input_channel = output_channel;
          impulse_channel = output_channel;
          if(num_input_channels < 2) input_channel = 0;
          if(num_impulse_channels < 2) impulse_channel = 0;
          if(((i-j) < 0) || ((i-j) >= num_input_samples)) input_sample = 0.0;
          else input_sample = inputAF.samples[input_channel][i-j];
          sum += input_sample * impulseAF.samples[impulse_channel][j];
        }
      }
      #pragma omp atomic
      output_buffer[0][i] += sum;
    }
  }

  //Stop timer
  if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}
  //Print convolution time & performance
  time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec)/1e9;
  long flops = (long)2*num_output_samples*num_impulse_samples*num_output_channels;
  printf("Done.\nNumber of FLOPs = %lu, Execution time = %f sec,\n%lf MFLOPs per sec\n", flops, time, 1/time/1e6*flops);

  //Mix in dry signal with wet signal
  double wetness = double(atof(argv[4]));
  for (int input_index = 0; input_index < num_input_samples; input_index++)
  {
    for(int output_channel = 0; output_channel < num_output_channels; output_channel++){
      int input_channel = output_channel;
      if(num_input_channels < 2) input_channel = 0;
      output_buffer[output_channel][input_index] = double(output_buffer[output_channel][input_index] * wetness);
      output_buffer[output_channel][input_index] += double(inputAF.samples[input_channel][input_index] * (1.0-wetness));
    }
  }

  bool error = inputAF.setAudioBuffer(output_buffer);
  inputAF.setAudioBufferSize(num_output_channels, num_output_samples);
  inputAF.save(argv[3], AudioFileFormat::Wave);
  return 0;
}

#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <stdlib.h>
#include <sndfile.h>
#include <math.h>
#include <stdbool.h>

#define VARIANCE_GAIN 0.997
#define TWO_PI 2.0*M_PI

// RACHEL OF THE FUTURE: THIS IS HOW YOU COMPILE AND RUN THIS:
// $ gcc -o hll hll.c -lsndfile
// $ ./hll
// Sincerely, Rachel of the past.

float* readfile(int *n_samples, float *samplerate, char *filepath) {
    SNDFILE *sf;
    SF_INFO info;
    int num_channels;
    int num_items;
    int f, sr, c;
    int i, j;
    FILE *out;

    info.format = 0;
    sf = sf_open(filepath, SFM_READ, &info);
    if (sf == NULL){
      printf("[HLL] Failed to open the audio file.\n");
      printf("%s\n", filepath);
      exit(-1);
    }

    f = info.frames;
    *samplerate = (float) info.samplerate;
    c = info.channels;
    num_items = f*c;

    if (c > 1) {
      printf("[HLL] Audio is not mono!");
      exit(1);
    }

    float *y = (float *) calloc(num_items, sizeof(float));

    *n_samples = sf_read_float(sf, y, num_items);
    sf_close(sf);

    return y;
}

int hll(float y[], int n_start, int n_samples, float fs, float f_init,
        int contour_num, int direction, float tracking_gain,
        float variance_gain, int n_harmonics, char *output_path, float f_cutoff,
        int min_contour_len, float amp_thresh, float update_thresh) {

  // File where output will be saved
  FILE *out;
  out = fopen(output_path, "a");

  // Initialize tracking arrays
  float a_hat;
  float *tracking_error = malloc(n_harmonics * sizeof(float));

  float complex *d0 = malloc(n_harmonics * sizeof(float complex));
  float complex *d1 = malloc(n_harmonics * sizeof(float complex));
  float complex *d2 = malloc(n_harmonics * sizeof(float complex));
  float complex *d3 = malloc(n_harmonics * sizeof(float complex));

  float complex *u0 = malloc(n_harmonics * sizeof(float complex));
  float complex *u1 = malloc(n_harmonics * sizeof(float complex));
  float complex *u2 = malloc(n_harmonics * sizeof(float complex));
  float complex *u3 = malloc(n_harmonics * sizeof(float complex));

  float *powers = malloc(n_harmonics * sizeof(float));
  float *tracking_error_const = malloc(n_harmonics * sizeof(float));

  float *sigma_hat_sq = malloc(n_harmonics * sizeof(float));

  float f_hat = f_init;

  // Initialize variables for each harmonic
  for (int k=0; k < n_harmonics; k++) {
    tracking_error[k] = 0.0;
    sigma_hat_sq[k] = 1.0;

    powers[k] = (float) (k + 1);
    tracking_error_const[k] = (fs/(TWO_PI))*powers[k];

    d0[k] = y[n_start] + 0.0*I;
    d1[k] = 0.0 + 0.0*I;
    d2[k] = 0.0 + 0.0*I;
    d3[k] = 0.0 + 0.0*I;

    u0[k] = 0.0 + 0.0*I;
    u1[k] = 0.0 + 0.0*I;
    u2[k] = 0.0 + 0.0*I;
    u3[k] = 0.0 + 0.0*I;
  }

  // filter coefficients
  float b[4];
  float a[4];
  if (f_cutoff == 50) {
    b[0] = 4.48699455e-08;
    b[1] = 1.34609837e-07;
    b[2] = 1.34609837e-07;
    b[3] = 4.48699455e-08;
    a[0] = 1.0;
    a[1] = -2.98575244;
    a[2] = 2.9716062;
    a[3] = -0.9858534;
  } else if (f_cutoff == 30) {
    b[0] = 9.71948607e-09;
    b[1] = 2.91584582e-08;
    b[2] = 2.91584582e-08;
    b[3] = 9.71948607e-09;
    a[0] = 1.0;
    a[1] = -2.99145146;
    a[2] = 2.98293941;
    a[3] = -0.99148788;
  } else if (f_cutoff == 20) {
    b[0] = 2.88394643e-09;
    b[1] = 8.65183928e-09;
    b[2] = 8.65183928e-09;
    b[3] = 2.88394643e-09;
    a[0] = 1.0;
    a[1] = -2.99430097;
    a[2] = 2.98861816;
    a[3] = -0.99431717;
  } else {
    printf("[HLL] %f is an invalid f_cutoff value", f_cutoff);
    exit(1);
  }

  // Initialize tracking parameters
  float phi_hat = 0.0;
  float tracking_update = 0.0;
  float complex xi = 0.0 + 0.0*I;
  float var_weight;
  float var_weight_sum;
  float amp_weight_sum;
  float average_var_numerator;
  float avgerage_amp_numerator;
  float avgerage_amplitude;
  bool print_sample;

  const float rad_per_sample = (TWO_PI)/fs;
  const float var_gain_minus_one = 1.0 - variance_gain;

  int counter = 0;
  int n;
  if (direction == 0) {
    n=n_start+1;
  } else {
    n=n_start-1;
  }

  while (n < n_samples && n >=0) {

    print_sample = (n % 256 == 0);

    f_hat += (f_hat/440.0)*(tracking_gain*tracking_update);
    phi_hat += (rad_per_sample*f_hat);

    if (print_sample) {
      fprintf(out, "%d,%d,%f", contour_num, n, f_hat);
    }

    if (isnan(f_hat)) {
      printf("[HLL] hit a NaN\n");
      printf("[HLL] tracking update: %f\n", tracking_update);
      fprintf(out, "[HLL] NAN\n");
      break;
    }

    xi = cexpf(-1.0*I*((float complex) phi_hat));
    var_weight_sum = 0.0;
    amp_weight_sum = 0.0;
    average_var_numerator = 0.0;
    avgerage_amplitude = 0.0;

    for (int k=0; k<n_harmonics; k++) {

      // compute d0
      if (k == 0) {
        d0[0] = y[n]*xi;
      } else {
        d0[k] = d0[k-1]*xi;
      }

      // compute u0
      u0[k] = (b[0]*d0[k]) + (b[1]*d1[k]) + (b[2]*d2[k]) + (b[3]*d3[k])
        - ((a[1]*u1[k]) + (a[2]*u2[k]) + (a[3]*u3[k]));

      tracking_error[k] = cargf((u0[k]*conjf(u1[k])))*tracking_error_const[k];

      sigma_hat_sq[k] *= variance_gain;
      sigma_hat_sq[k] += (tracking_error[k]*tracking_error[k])*var_gain_minus_one;

      a_hat = cabsf(u0[k]);

      if (sigma_hat_sq[k] > 0.001) {
        var_weight = 1.0/sigma_hat_sq[k];
      } else {
        var_weight = 1000.0;
      }

      var_weight_sum += var_weight;
      amp_weight_sum += a_hat;
      average_var_numerator += (var_weight*tracking_error[k]);

      avgerage_amplitude = amp_weight_sum/(float) n_harmonics;

      if (print_sample) {
        fprintf(out, ",%f", a_hat);
      }

      d3[k] = d2[k];
      d2[k] = d1[k];
      d1[k] = d0[k];
      u3[k] = u2[k];
      u2[k] = u1[k];
      u1[k] = u0[k];
    }

    if (amp_weight_sum == 0) {
      amp_weight_sum = 1.0;
    }

    tracking_update = average_var_numerator/var_weight_sum;

    if (print_sample) {
      fprintf(out, "\n");
    }

    if(f_hat <= 20.0 || f_hat >= 4800.0 || (counter > min_contour_len && (avgerage_amplitude < amp_thresh ||
       fabsf(tracking_update) > update_thresh))) {
      break;
    }

    counter++;
    if (direction == 0) {
      n++;
    } else {
      n--;
    }

  }

  fclose(out);
  return 0;
}


int main(int argc, char *argv[]) {

  char *filepath = argv[1]; // path to audio file
  char *seed_path = argv[2]; // path to seed file
  char *output_path = argv[3]; // path to saved output

  // tracking parameters
  int n_harmonics = atoi(argv[4]); // number of harmonics
  float f_cutoff = atof(argv[5]); // cutoff frequency. Must be 50, 30, or 20
  float tracking_gain = atof(argv[6]); // tracking gain

  // contour killing parameters
  int min_contour_len = atof(argv[7]); // minimum contour length
  float amp_thresh = atof(argv[8]); // amplitude minimum threshold
  float update_thresh = atof(argv[9]); // maximum tracking update threshold

  int n_samples;
  float samplerate;
  float *y = readfile(&n_samples, &samplerate, filepath);

  FILE *seed_file_pointer;

  seed_file_pointer = fopen(seed_path, "r");
  if (seed_file_pointer == NULL){
    printf("[HLL] Error. Seed file is empty.\n");
    exit(1);
  }

  float freq;
  int seed;
  char *out;

  int contour_num = 0;

  while (!feof(seed_file_pointer)) {
    float c_float;
    fscanf(seed_file_pointer, "%f,%f\n", &c_float, &freq);
    seed = (int) c_float;

    // forward pass
    hll(y, seed, n_samples, samplerate, freq, contour_num, 0,
        tracking_gain, VARIANCE_GAIN, n_harmonics, output_path,
        f_cutoff, min_contour_len, amp_thresh, update_thresh);

    //backward pass
    hll(y, seed, n_samples, samplerate, freq, contour_num, 1,
        tracking_gain, VARIANCE_GAIN, n_harmonics, output_path,
        f_cutoff, min_contour_len, amp_thresh, update_thresh);
    contour_num++;
  }

  return 0;
}

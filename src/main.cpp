#include "SpectroWAVe.h"

int main(int argc, char* argv[]) {

  std::string filepath = argv[1]; 

  /**
   * The initial idea was to show the user each second how
   * the accumulated spectrum had change in the given sound file so far.
   * With an FFT size similar to the sample rate. We should get
   * the spectra respective to a second in the sampled audio. 
   *
   * This default behaviour may now be override.
   */
  int fftSize = 44100; // Default value.

  try {
    fftSize = std::stoi(argv[2]);
  } catch (...) {};

  int binning = 9;
  try {
    binning = std::stoi(argv[3]);
  } catch (...) {};

  std::cout << "Program inputs" << std::endl;
  std::cout << "---------------------" << std::endl;
  std::cout << "filepath: " << filepath << std::endl;
  std::cout << "fftSize: " << fftSize << std::endl;
  std::cout << "binning: " << binning << std::endl;
  std::cout << "---------------------" << std::endl;

  SpectroWAVe* program = new SpectroWAVe(filepath, fftSize, binning); 
  program->run();

  return 0;
};

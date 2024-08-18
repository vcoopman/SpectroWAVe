#ifndef SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSOR_H

#include <vector>

class SignalProcessor {
  /**
   * Interface for class in charge of the spectral processing of the signal of 1 channel of the input signal. 
   */

  public:
    SignalProcessor(std::vector<float> signal) : signal_(signal) {};
    virtual ~SignalProcessor() {};

    virtual std::vector<float> getSpectraMagnitudes(int fftSize, int fftOutputSize, int iterationCount) = 0;

  protected:
    std::vector<float> signal_;

};

#endif

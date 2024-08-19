#ifndef SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSOR_H

#include <vector>

class SignalProcessor {
  /**
   * Interface for classes in charge of the spectral processing of
   * the signal of a single channel of the input signal. 
   */

  public:
    SignalProcessor(std::vector<float> signal) : signal_(signal) {};
    virtual ~SignalProcessor() {};

    /**
     * Returns the spectral magnitudes of each frequency component. 
     * iterationCount argument is used to indicate which segment of the signal must be processed.
     */
    virtual std::vector<float> getSpectraMagnitudes(int fftSize, int fftOutputSize, int iterationCount) = 0;

  protected:
    std::vector<float> signal_;

};

#endif

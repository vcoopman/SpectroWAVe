#ifndef CUDA_SIGNAL_PROCESSOR_H
#define CUDA_SIGNAL_PROCESSOR_H

#include <iostream>

#include <cuda_runtime.h>
#include <cufft.h>
#include "cufft_utils.h"

#include "SignalProcessor.h"

class CUDASignalProcessor : public SignalProcessor {
  /**
   * CUDA Base implementation of a SignalProcessor.
   */

  public:
    CUDASignalProcessor(std::vector<float> signal);
    ~CUDASignalProcessor();

    std::vector<float> getSpectraMagnitudes(int fftInputSize, int fftOutputSize, int iterationCount) override;

  private:
    cufftReal* d_signal_;
    cufftComplex* d_signalSpectrum_;
    cufftHandle plan_;
};

#endif

#ifndef SIGNAL_PROCESSOR_ACCUMULATOR_H
#define SIGNAL_PROCESSOR_ACCUMULATOR_H

#include <vector>

#include "SignalCollector.h"
#include "SignalProcessor.h"

class SignalProcessorAccumulator {

  public:
    SignalProcessorAccumulator(SignalCollector* collector, int binning);
    ~SignalProcessorAccumulator();

    /**
     * TODO
     */
    std::vector<float> getTimeAverageSpectra();

    /**
     * The process of binning the spectral data consists
     * on averaging a "binning" number of spectral channels together.
     */
    std::vector<float> getBinnedTimeAverageSpectra();

  private:
    int binning_;
    int fftInputSize_;
    int fftOutputSize_;
    int iterationCount_;

    SignalCollector* collector_;

    std::vector<SignalProcessor*> processors_;
    std::vector<float> timeAverageSpectralMagnitudeCache_;

};

#endif

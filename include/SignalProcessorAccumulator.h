#ifndef SIGNAL_PROCESSOR_ACCUMULATOR_H
#define SIGNAL_PROCESSOR_ACCUMULATOR_H

#include <vector>

#include "SignalCollector.h"
#include "SignalProcessor.h"

class SignalProcessorAccumulator {

  public:
    SignalProcessorAccumulator(SignalCollector* collector, int fftInputSize, int binning);
    ~SignalProcessorAccumulator();

    /**
     * The process of binning the spectral data consists
     * on averaging a "binning" number of spectral channels together.
     */
    std::vector<float> getBinnedTimeAverageSpectraMagnitudes();

    int getRemainingIterations();

  private:
    int fftInputSize_;
    int fftOutputSize_;
    int expectedIterations_;
    int iterationCounter_;
    int binning_;

    SignalCollector* collector_;
    std::vector<SignalProcessor*> processors_;

    std::vector<float> timeAverageSpectralMagnitudeCache_;

    /**
     * Updates internal time average magnitudes cache. 
     */
    void updateTimeAverageSpectraMagnitudesCache();

};

#endif

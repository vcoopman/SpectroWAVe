#ifndef SIGNAL_PROCESSOR_ACCUMULATOR_H
#define SIGNAL_PROCESSOR_ACCUMULATOR_H

#include <vector>

#include "SignalCollector.h"
#include "SignalProcessor.h"

class SignalProcessorAccumulator {
  /**
   * SignalProcessorAccumulator accumulates results from SignalProcessor(s), which iterative processes the input signal(s).
   * SignalProcessorAccumulator is in charge of performing time-average and binning over the accumulated results.
   */

  public:
    SignalProcessorAccumulator(SignalCollector* collector, int fftInputSize, int binning);
    ~SignalProcessorAccumulator();

    /**
     * The process of binning the spectral data consists
     * on averaging a "binning" number of spectral channels together.
     *
     * When there aren't enough channels to match the "binning" number,
     * reminder channels magnitudes are discarded.
     */
    std::vector<float> getBinnedTimeAverageSpectraMagnitudes();

    int getRemainingIterations();

  private:
    /**
     * The Signal is processed by segments. This accumulator keeps track
     * of the executed iterations to indicate SignalProcessors which
     * segment of the signal to process next.
     */
    int iterationCounter_;

    int fftInputSize_;
    int fftOutputSize_; // real-to-complex FFT follows Hermitian simetry.

    int expectedIterations_;
    int binning_;

    SignalCollector* collector_;
    std::vector<SignalProcessor*> processors_;

    /**
     * Internal cache to keep results from previous iterations time-average spectra.
     */
    std::vector<float> timeAverageSpectralMagnitudeCache_;

    /**
     * Updates internal time-average spectra magnitudes cache. 
     */
    void updateTimeAverageSpectraMagnitudesCache();
};

#endif

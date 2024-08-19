#include <vector>
#include <chrono>
#include <thread>
#include <iostream>

#include "FileSignalCollector.h"
#include "SignalProcessorAccumulator.h"
#include "DisplayWindow.h"

class SpectroWAVe {
  /**
   * TODO
   */

  public:
    SpectroWAVe(std::string filepath, int fftInputSize, int binning);
    ~SpectroWAVe();

    void run();

  private:
    FileSignalCollector* collector_;
    SignalProcessorAccumulator* accumulator_;

    int iterationExpectedDuration_ms_;
    std::chrono::high_resolution_clock::time_point iterationStart_;
    std::chrono::high_resolution_clock::time_point iterationEnd_;

    DisplayWindow* display_;

    /**
     * Used to sleep the remaining time (after spectral processing) of the iteration.
     */
    void sleepRemainingIterationTime(int iterationCounter);
};

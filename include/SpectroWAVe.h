#include <vector>
#include <chrono>
#include <thread>
#include <iostream>

#include "FileSignalCollector.h"
#include "SignalProcessorAccumulator.h"
#include "DisplayWindow.h"

class SpectroWAVe {
  /**
   * SpectroWAVe is visualizer of the accumulated spectral magnitudes
   * across the different frequency components of a signal.
   *
   * It can read signal from files in the WAV format.
   * WAV format was intially choosed because the this file format stores data uncompressed
   * and because it gave the tool a cool name.
   *
   * SpectroWAVe is powered by CUDA (cuFFT) and SDL2. Enjoy! :)
   *
   * Mon 19, August 2024 (vcoopman) : Created.
   */

  public:
    SpectroWAVe(std::string filepath, int fftInputSize, int binning);
    ~SpectroWAVe();

    void run();

  private:
    FileSignalCollector* collector_;
    SignalProcessorAccumulator* accumulator_;

    /**
     * Iterations expected time depend in the signal sample rate and FFT size.
     *
     * For example:
     *  Sample Rate: 44100        Sample Rate: 44100
     *  FFT size: 44100           FFT size: 22050
     *  --                        --
     *  Iteration Time: 1 sec     Iteration Time: 0.5 sec
     */
    int iterationExpectedDuration_ms_;
    std::chrono::high_resolution_clock::time_point iterationStart_;
    std::chrono::high_resolution_clock::time_point iterationEnd_;

    DisplayWindow* display_;

    /**
     * Used to sleep the remaining time (after spectral processing) of the iteration.
     */
    void sleepRemainingIterationTime(int iterationCounter);
};

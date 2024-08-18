#include <vector>
#include <chrono>
#include <thread>
#include <iostream>

#include "AudioSignalCollector.h"
#include "SignalProcessorAccumulator.h"
#include "DisplayWindow.h"

class SpectroWAVe {
  public:
    SpectroWAVe();
    ~SpectroWAVe();

    void setUp(std::string filepath, int binning);
    void run();
    void cleanUp();

  private:
    bool isSetup_;

    SignalProcessorAccumulator* accumulator_;

    std::chrono::high_resolution_clock::time_point iterationStart_;
    std::chrono::high_resolution_clock::time_point iterationEnd_;

    DisplayWindow* display_;

};

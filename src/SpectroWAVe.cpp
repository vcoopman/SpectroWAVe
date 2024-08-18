#include "SpectroWAVe.h"

SpectroWAVe::SpectroWAVe() {
  isSetup_ = false;
};

SpectroWAVe::~SpectroWAVe() {};

void SpectroWAVe::setUp(std::string filepath, int binning) {
  if (isSetup_) cleanUp();

  accumulator_ = new SignalProcessorAccumulator(
    new AudioSignalCollector(filepath),
    binning
  );

  display_ = new DisplayWindow();

  isSetup_ = true;
};


void SpectroWAVe::run() {
  if (!isSetup_) { throw std::runtime_error("SpectroWAVe is not setUp!"); }

  std::cout << "Entering main loop" << std::endl;

  // TODO: Improve this to match full file length.
  int counter = 1;
  while (counter <= 120) {
    iterationStart_ = std::chrono::high_resolution_clock::now();
  
    std::vector<float> binnedTimeAverageSpectralMagnitudes = accumulator_->getBinnedTimeAverageSpectra();
    display_->display(binnedTimeAverageSpectralMagnitudes);
    display_->checkEvent();

    iterationEnd_ = std::chrono::high_resolution_clock::now();
    auto iterationElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(iterationEnd_ - iterationStart_).count();
    std::cout << "Iteration[" << counter << "] time: " << iterationElapsed  << " ms." << std::endl;

    // Calculate the remaining time to sleep until a second is complete.
    int remainingTime = 1000 - iterationElapsed;
    if (remainingTime > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(remainingTime));
    } else {
      std::cout << "Small anxiety - Iteration[" << counter << "] finished "
      << remainingTime << " milliseconds late." << std::endl;
    } 

    ++counter;
  }
};

void SpectroWAVe::cleanUp() {
  if (!isSetup_) return;
  // TODO: Should pointer also point to nullptr?
  // TODO: Clean up accumulator.
  // TODO: Clean display window.
  free(accumulator_);
  isSetup_ = false;
};

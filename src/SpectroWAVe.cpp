#include "SpectroWAVe.h"

#include "WAVFileSignalCollector.h"

SpectroWAVe::SpectroWAVe(std::string filepath, int fftInputSize, int binning) {
  collector_ = new WAVFileSignalCollector(filepath);
  collector_->load();

  iterationExpectedDuration_ms_ = (fftInputSize / static_cast<float>(collector_->getSignalSampleRate())) * 1000;
  std::cout << "Iterations expected duration: " << iterationExpectedDuration_ms_ << " ms." << std::endl;

  accumulator_ = new SignalProcessorAccumulator(collector_, fftInputSize, binning);
  display_ = new DisplayWindow(collector_);
};

SpectroWAVe::~SpectroWAVe() {
  free(collector_);
  free(accumulator_);
  free(display_);
};

void SpectroWAVe::run() {
  std::cout << "Entering main loop" << std::endl;

  int iterationCounter = 1;
  while (accumulator_->getRemainingIterations() > 0) {

    iterationStart_ = std::chrono::high_resolution_clock::now();
    {
      if (iterationCounter == 1) display_->startMusic();

      display_->display(
        accumulator_->getBinnedTimeAverageSpectraMagnitudes()
      );

      display_->checkEvent();
    }
    iterationEnd_ = std::chrono::high_resolution_clock::now();

    sleepRemainingIterationTime(iterationCounter);
    ++iterationCounter;
  }
};

void SpectroWAVe::sleepRemainingIterationTime(int iterationCounter) {
  // Calculate the remaining time to sleep
  auto iterationElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(iterationEnd_ - iterationStart_).count();
  int remainingTime = iterationExpectedDuration_ms_ - iterationElapsed;
  if (remainingTime > 0) {
    std::cout << "Iteration[" << iterationCounter << "] time: " 
      << iterationElapsed  << " ms. Sleeping for " << remainingTime << "ms." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(remainingTime));
  } else {
    std::cout << "Iteration[" << iterationCounter << "] finished: " << remainingTime << " ms late (Small anxiety)." << std::endl;
  } 
}

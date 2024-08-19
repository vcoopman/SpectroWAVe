#include "SignalProcessorAccumulator.h"
#include "CUDASignalProcessor.h"

SignalProcessorAccumulator::SignalProcessorAccumulator(SignalCollector* collector, int fftInputSize, int binning) :
  collector_(collector),
  binning_(binning),
  fftInputSize_(fftInputSize) 
{
  // Create processors, one per channel.
  std::vector<std::vector<float>> signals = collector_->getAllAvailableChannelsSignal();
  for (auto signal: signals) {
    SignalProcessor* processor = new CUDASignalProcessor(signal);
    processors_.push_back(processor);
  };

  fftOutputSize_ = (fftInputSize_ / 2) + 1; // real-to-complex FFT follows Hermitian simetry.
  std::cout << "DEBUG fftOutputSize_: " << fftOutputSize_ << std::endl; 

  timeAverageSpectralMagnitudeCache_ = std::vector<float>(fftOutputSize_);
  expectedIterations_ = collector_->getSignalFrames() / fftInputSize;
  iterationCounter_ = 0;
}

SignalProcessorAccumulator::~SignalProcessorAccumulator() {
  for (auto processor : processors_)
    free(processor);
};


void SignalProcessorAccumulator::updateTimeAverageSpectraMagnitudesCache() {
  if (getRemainingIterations() == 0)
    throw std::runtime_error("There are no more iterations to execute. All signal frames were processed.");

  std::vector<float> iterationAccumulatedSpectraMagnitudes(fftOutputSize_, 0);
  for (auto processor : processors_) {
    std::vector<float> spectraMagnitudes = processor->getSpectraMagnitudes(fftInputSize_, fftOutputSize_, iterationCounter_);
    // This could be a CUDA kernel. 
    // But for the moment iteration times are so low that CUDA optimization is unneeded. 
    for (int i = 0; i < this->fftOutputSize_ ; ++i)
      iterationAccumulatedSpectraMagnitudes[i] += spectraMagnitudes[i];
  }

  /**
   * What is correct here?
   * --------------------
   *  - Option A): 
   *      Sum all channels magnitudes, average by the number of available channels,
   *      and then sum and average this result together with the cache timed-average spectra.
   *
   *  - Option B):
   *      Sum all channels magnitudes, sum it with the cache time-average spectra,
   *      then average by time (iteration count).
   *
   *  Note: Both options introduce an error everytime they are time-averaged. 
   *        This is because we are storing just the previousr resulting time-averages,
   *        rather than the previous values.
   *
   * Conclusion:
   *  Probably B) is less wrong. Less averages, less error.
   *
   */

  // This could be a CUDA kernel too.
  // But for the moment iteration times are so low that CUDA optimization is unneeded. 
  for (int i = 0; i < fftOutputSize_ ; ++i) {
    timeAverageSpectralMagnitudeCache_[i] = \
      (iterationAccumulatedSpectraMagnitudes[i] + 
       // Multipliying the cache value for the iterationCounter_ is a naive way
       // to deal with constant averaging error. 
        (timeAverageSpectralMagnitudeCache_[i] * iterationCounter_))
          / (iterationCounter_ + 1);
  } 

  iterationCounter_ += 1;
}

std::vector<float> SignalProcessorAccumulator::getBinnedTimeAverageSpectraMagnitudes() {
  updateTimeAverageSpectraMagnitudesCache();

  std::vector<float> output(0);
  float accumulatedSum = 0.0;
  int counter = binning_;
  for (auto magnitude : timeAverageSpectralMagnitudeCache_) { // TODO: Reminder channels are ignored.
    accumulatedSum += magnitude;

    --counter;

    if (counter == 0) {
      output.push_back(accumulatedSum / binning_);
      accumulatedSum = 0.0;
      counter = binning_;
      continue;
    }
  }

  return output;
}

int SignalProcessorAccumulator::getRemainingIterations() {
  return expectedIterations_ - iterationCounter_;
}

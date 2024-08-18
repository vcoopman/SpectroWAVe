#include "SignalProcessorAccumulator.h"
#include "CUDASignalProcessor.h"

SignalProcessorAccumulator::SignalProcessorAccumulator(SignalCollector* collector, int binning) : collector_(collector), binning_(binning) {

  collector_->load();

  // Create processors, one per channel.
  std::vector<std::vector<float>> signals = collector_->getAllAvailableChannelsSignal();
  for (auto signal: signals) {
    SignalProcessor* processor = new CUDASignalProcessor(signal);
    processors_.push_back(processor);
  };

  /**
   * The idea is to show the user each second how
   * the accumulated spectrum has change in the given sound file so far.
   *
   * This is a hardcoded value focused on 44.1 kHz sampled audio.
   * With an FFT size similar to the sample rate. We should get
   * the spectra respective to a second in the sampled audio. 
   */
  fftInputSize_ = collector_->getSignalSampleRate(); // We want the last second spectra.
  fftOutputSize_ = (fftInputSize_ / 2) + 1; // real-to-complex follows Hermitian simetry.

  timeAverageSpectralMagnitudeCache_ = std::vector<float>(fftOutputSize_);
  iterationCount_ = 0;
}

SignalProcessorAccumulator::~SignalProcessorAccumulator() {};


std::vector<float> SignalProcessorAccumulator::getTimeAverageSpectra() {
  std::vector<float> accumulatedSpectraMagnitudes(fftOutputSize_, 0);
  for (auto processor : processors_) {

    std::vector<float> spectraMagnitudes = processor->getSpectraMagnitudes(fftInputSize_, fftOutputSize_, iterationCount_);

    // TODO: There is room for improvement here.
    for (int i = 0; i < this->fftOutputSize_ ; ++i) {
      accumulatedSpectraMagnitudes[i] += spectraMagnitudes[i];
    }
  }

  /**
   * What is correct?
   *  - Option A: Sum all channels magnitudes, average it across channels, and then sum and average this result with the cache timed average spectra.
   *  - Option B: Sum all channels magnitudes, sum it with the cache time-average spectra, then average by iteration count (time).
   *
   *  Note: Both options introduce an error everytime they are time-averaged. 
   *        This is because we are storing just the previousr resulting time-averages, rather than the previous values.
   *
   * Conclusion:
   *  Is option A what is called channel-average? I will implement B. Less averages, less error.
   */

  for (int i = 0; i < fftOutputSize_ ; ++i) {
    timeAverageSpectralMagnitudeCache_[i] = \
      (accumulatedSpectraMagnitudes[i] + 
        (timeAverageSpectralMagnitudeCache_[i] * iterationCount_)) // Note: Multipliying the cache value for the iterationCount_ is a naive way to deal with the previous mentioned error.
          / (iterationCount_ + 1);
  } 

  iterationCount_ += 1;
  return timeAverageSpectralMagnitudeCache_;
}

std::vector<float> SignalProcessorAccumulator::getBinnedTimeAverageSpectra() {

  std::vector<float> timeAverageSpectralMagnitudes = getTimeAverageSpectra();

  // TODO: Review this.
  //  int outputSize = timeAverageSpectralMagnitudes.size() / binning_; // Reminder channels are ignored.

  std::vector<float> output(0);

  int counter = binning_;
  float accumulatedSum = 0.0;
  for (auto magnitude : timeAverageSpectralMagnitudes) {
    accumulatedSum += magnitude;

    --counter;

    if (counter == 0) {
      output.push_back(accumulatedSum / binning_);
      counter = binning_;
      continue;
    }
  }

  return output;
}

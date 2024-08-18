#ifndef AUDIO_SIGNAL_COLLECTOR_H
#define AUDIO_SIGNAL_COLLECTOR_H


#include <string>

// https://libsndfile.github.io/libsndfile/api.html
#include <sndfile.h>

#include "SignalCollector.h"

class AudioSignalCollector : public SignalCollector {
  /**
   * Used to collect input signal from of an audio file in the WAVE format.
   */

  public:
    AudioSignalCollector(std::string filepath) : filepath_(filepath) {};
    ~AudioSignalCollector() {};

    void load() override;
    void unload() override;

    int getSignalChannels() override;
    int getSignalSampleRate() override;

  private:
    std::string filepath_;
    SF_INFO sfinfo_;
    SNDFILE* infile_;

};

#endif

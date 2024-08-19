#ifndef AUDIO_SIGNAL_COLLECTOR_H
#define AUDIO_SIGNAL_COLLECTOR_H

// https://libsndfile.github.io/libsndfile/api.html
#include <sndfile.h>

#include "FileSignalCollector.h"

class WAVFileSignalCollector : public FileSignalCollector {
  /**
   * Used to collect input signal from of an audio file in the WAVE format.
   */

  public:
    WAVFileSignalCollector(std::string filepath);
    ~WAVFileSignalCollector();

    void load() override;
    void unload() override;

    int getSignalChannels() override;
    int getSignalSampleRate() override;
    int getSignalFrames() override;

  private:
    SF_INFO sfinfo_;
    SNDFILE* infile_;
};

#endif

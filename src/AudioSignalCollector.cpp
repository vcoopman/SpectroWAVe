#include <iostream> // TODO: Remove

#include "AudioSignalCollector.h"

void AudioSignalCollector::load() {
    if (loaded) return;

    infile_ = sf_open(filepath_.c_str(), SFM_READ, &sfinfo_);

    //std::cout << "Sample rate: " << sfinfo_.samplerate << " Hz" << std::endl;
    //std::cout << "Format: " << sfinfo_.format << std::endl; // 65538 --> SF_FORMAT_WAV, SF_FORMAT_PCM_16 (signed 16 bit data) Half-precision floating point.
    //std::cout << "Channels: " << sfinfo_.channels << std::endl;
    //std::cout << "Frames: " << sfinfo_.frames << std::endl;

    std::vector<float> signal(sfinfo_.frames * sfinfo_.channels);
    sf_readf_float(infile_, signal.data(), sfinfo_.frames);

    auto it = signal.begin();
    for (size_t i = 0; i < sfinfo_.channels ; ++i) {
        signals_.push_back(std::vector<float>(it, it + (i == 0 ? sfinfo_.frames : sfinfo_.frames * i )));
        it += sfinfo_.frames;
    }

    // TODO: Debug
    //int counter = 1;
    //int counter_2 = 0;
    //for (auto signal : signals_) {
        //std::cout << "\n" << std::endl;
        //std::cout << "Signal " << counter << ": " <<  std::endl;
        //for (auto amplitude : signal) {
            //counter_2++;
            //std::cout << amplitude << " ";
            //if (counter_2 > 20000) break;
        //}
        //counter_2 = 0;
        //++counter;
    //}

    loaded = true;
}


void AudioSignalCollector::unload() {
    if (!loaded) return;

    sf_close(infile_);
    for (auto signal : signals_) signal.clear();

    loaded = false;
}

int AudioSignalCollector::getSignalChannels() {
    return sfinfo_.channels;
}

int AudioSignalCollector::getSignalSampleRate() {
    return sfinfo_.samplerate;
}

#include "WAVFileSignalCollector.h"

WAVFileSignalCollector::WAVFileSignalCollector(std::string filepath) :
    FileSignalCollector(filepath) {};

WAVFileSignalCollector::~WAVFileSignalCollector() {
    unload();
};

void WAVFileSignalCollector::load() {
    if (loaded_) return;

    infile_ = sf_open(filepath_.c_str(), SFM_READ, &sfinfo_);

    std::cout << "Loading file at: " << filepath_.c_str() << std::endl;
    std::cout << "Sample rate: " << sfinfo_.samplerate << " Hz" << std::endl;
    std::cout << "Format: " << sfinfo_.format << std::endl; // 65538 --> SF_FORMAT_WAV, SF_FORMAT_PCM_16 (signed 16 bit data) Half-precision floating point.
    std::cout << "Channels: " << sfinfo_.channels << std::endl;
    std::cout << "Frames: " << sfinfo_.frames << std::endl;

    std::vector<float> signal(sfinfo_.frames * sfinfo_.channels);
    sf_readf_float(infile_, signal.data(), sfinfo_.frames);

    // Split the signal by channels
    auto it = signal.begin();
    for (size_t i = 0; i < sfinfo_.channels ; ++i) {
        auto channelLowerLimit = it;
        auto channelUpperLimit = it + (i == 0 ? sfinfo_.frames : sfinfo_.frames * i );
        signals_.push_back(std::vector<float>(channelLowerLimit, channelUpperLimit));
        it += sfinfo_.frames;
    }

    std::cout << "Collector loaded." << std::endl;
    loaded_ = true;
}


void WAVFileSignalCollector::unload() {
    if (!loaded_) return;
    sf_close(infile_);
    for (auto signal : signals_) signal.clear();
    std::cout << "Collector unloaded." << std::endl;
    loaded_ = false;
}

int WAVFileSignalCollector::getSignalChannels() {
    return sfinfo_.channels;
}

int WAVFileSignalCollector::getSignalSampleRate() {
    return sfinfo_.samplerate;
}

int WAVFileSignalCollector::getSignalFrames() {
    return sfinfo_.frames;
}

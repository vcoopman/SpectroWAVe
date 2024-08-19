#ifndef SIGNAL_COLLECTOR_H
#define SIGNAL_COLLECTOR_H

#include <vector>
#include <string>
#include <iostream>

class SignalCollector {
  /**
   * Interface for objects in charge of collecting the signal.
   */

  public:
    SignalCollector() {};
    virtual ~SignalCollector() {};

    virtual void load() = 0;
    virtual void unload() = 0;
    bool isLoaded() { return loaded_; }

    virtual int getSignalChannels() = 0;
    virtual int getSignalSampleRate() = 0;
    virtual int getSignalFrames() = 0;

    std::vector<std::vector<float>> getAllAvailableChannelsSignal() { return signals_; }

  protected:
    bool loaded_;
    std::vector<std::vector<float>> signals_;

};

#endif

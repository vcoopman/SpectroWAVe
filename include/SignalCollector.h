#ifndef SIGNAL_COLLECTOR_H
#define SIGNAL_COLLECTOR_H

#include <vector>

class SignalCollector {
  /**
   * Interface for objects in charge of retriving the signal.
   */

  public:
    SignalCollector() = default; 
    virtual ~SignalCollector() {};

    virtual void load() = 0;
    virtual void unload() = 0;
    bool isLoaded() { return loaded; }

    virtual int getSignalChannels() = 0;
    virtual int getSignalSampleRate() = 0;

    std::vector<std::vector<float>> getAllAvailableChannelsSignal() { return signals_; }

  protected:
    bool loaded;
    std::vector<std::vector<float>> signals_;

};

#endif

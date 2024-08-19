#ifndef SIGNAL_COLLECTOR_H
#define SIGNAL_COLLECTOR_H

#include <vector>

class SignalCollector {
  /**
   * Interface for objects in charge of collecting the signal.
   */

  public:
    SignalCollector() {};
    virtual ~SignalCollector() {};

    /**
     * Load file info and signal into this kind object.
     */
    virtual void load() = 0;
    
    /**
     * Unloads file info and signal from this kind object.
     */
    virtual void unload() = 0;

    bool isLoaded() { return loaded_; }

    virtual int getSignalChannels() = 0;
    virtual int getSignalSampleRate() = 0;
    virtual int getSignalFrames() = 0;

    /**
     * Returns all loaded signals.
     */
    std::vector<std::vector<float>> getAllAvailableChannelsSignal() { return signals_; }

  protected:
    bool loaded_;
    std::vector<std::vector<float>> signals_;
};

#endif

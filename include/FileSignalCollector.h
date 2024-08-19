#ifndef FILE_SIGNAL_COLLECTOR_H
#define FILE_SIGNAL_COLLECTOR_H

#include "SignalCollector.h"

class FileSignalCollector : public SignalCollector {
  /**
   * Interface for collectors that collect input signal from a file. 
   */

  public:
    FileSignalCollector(std::string filepath) : filepath_(filepath) {};
    virtual ~FileSignalCollector() {};

    std::string getFilepath() { return filepath_; }

  protected:
    std::string filepath_;

};

#endif

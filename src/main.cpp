#include "SpectroWAVe.h"

int main(int argc, char* argv[]) {

  std::string filepath = argv[1]; 
  int binning = std::stoi(argv[2]);

  SpectroWAVe* program = new SpectroWAVe(); 
  program->setUp(filepath, binning);
  program->run();
  program->cleanUp();

  return 0;
};

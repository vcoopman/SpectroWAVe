CXX       := nvcc
OPTIONS   := -g -G -x cu
INC       := -I./include -I./lib
LIBS      := -L/usr/lib/x86_64-linux-gnu -lcufft -lsndfile -lSDL2 -lSDL2_ttf -lSDL2_mixer
LDFLAGS   := -std=c++11
TARGET    := build/executable
SRC_DIR   := ./src
SRC       := $(SRC_DIR)/*

# Default values
FFTSIZE = 44100
BINNING = 10

all:
	mkdir -p build  && \
		$(CXX) $(OPTIONS) $(LDFLAGS) $(INC) $(LIBS) $(SRC) -o $(TARGET)

exec:
	./build/executable $(FILE) $(FFTSIZE) $(BINNING) | tee ./build/out.log

clean:
	rm -f ./build/*

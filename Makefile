CXX       := nvcc
OPTIONS   := -g -G -x cu
INC       := -I./include -I./lib
LIBS      := -L/usr/lib/x86_64-linux-gnu -lcufft -lsndfile -lSDL2 -lSDL2_ttf
LDFLAGS   := -std=c++11
TARGET    := build/executable
SRC_DIR   := ./src
SRC       := $(SRC_DIR)/*

all:  
	$(CXX) $(OPTIONS) $(LDFLAGS) $(INC) $(LIBS) $(SRC) -o $(TARGET)

exec:
	./build/executable $(FILE) $(BINNING) | tee ./build/out.txt

clean:
	rm -f ./build/*

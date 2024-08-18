#include <iostream>
#include <vector>
#include <sndfile.h>
#include <cuda_runtime.h>
#include <cufft.h>

// Error checking macros
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define cufftCheckError(status) { \
    if (status != CUFFT_SUCCESS) { \
        std::cerr << "CUFFT error: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    // File path to the audio file
    const char* filename = "song1.wav";

    // Open the audio file
    SF_INFO sfinfo;
    SNDFILE* infile = sf_open(filename, SFM_READ, &sfinfo);
    if (!infile) {
        std::cerr << "Error opening audio file: " << sf_strerror(NULL) << std::endl;
        return EXIT_FAILURE;
    }

    // Output the sample rate and number of channels
    std::cout << "Sample rate: " << sfinfo.samplerate << " Hz" << std::endl;
    std::cout << "Channels: " << sfinfo.channels << std::endl;

    // Allocate buffer for audio data
    std::vector<float> h_signal(sfinfo.frames * sfinfo.channels);
    sf_readf_float(infile, h_signal.data(), sfinfo.frames);
    sf_close(infile);

    // Allocate device memory for both channels
    cufftComplex* d_signal[2];
    for (int ch = 0; ch < sfinfo.channels; ++ch) {
        cudaMalloc(&d_signal[ch], sizeof(cufftComplex) * sfinfo.frames);
        cudaCheckError();
        cudaMemcpy(d_signal[ch], &h_signal[ch * sfinfo.frames], sizeof(float) * sfinfo.frames, cudaMemcpyHostToDevice);
        cudaCheckError();
    }

    // Create cuFFT plan for both channels
    cufftHandle plan[2];
    for (int ch = 0; ch < sfinfo.channels; ++ch) {
        cufftResult result = cufftPlan1d(&plan[ch], sfinfo.frames, CUFFT_C2C, 1);
        cufftCheckError(result);
    }

    // Execute FFT for both channels
    for (int ch = 0; ch < sfinfo.channels; ++ch) {
        cufftResult result = cufftExecC2C(plan[ch], d_signal[ch], d_signal[ch], CUFFT_FORWARD);
        cufftCheckError(result);
    }

    // Copy results back to host for both channels
    std::vector<cufftComplex> h_result[sfinfo.channels];
    for (int ch = 0; ch < sfinfo.channels; ++ch) {
        h_result[ch].resize(sfinfo.frames);
        cudaMemcpy(h_result[ch].data(), d_signal[ch], sizeof(cufftComplex) * sfinfo.frames, cudaMemcpyDeviceToHost);
        cudaCheckError();
    }

    // Print results for both channels
    for (int ch = 0; ch < sfinfo.channels; ++ch) {
        std::cout << "Channel " << ch << " FFT Results:" << std::endl;
        for (size_t i = 0; i < sfinfo.frames; ++i) {
            float magnitude = sqrt(h_result[ch][i].x * h_result[ch][i].x + h_result[ch][i].y * h_result[ch][i].y);
            float phase = atan2(h_result[ch][i].y, h_result[ch][i].x);
            std::cout << "Index " << i << ": Magnitude = " << magnitude << ", Phase = " << phase << std::endl;
        }
    }

    // Clean up
    for (int ch = 0; ch < sfinfo.channels; ++ch) {
        cufftDestroy(plan[ch]);
        cudaFree(d_signal[ch]);
    }
    cudaCheckError();

    return 0;
}


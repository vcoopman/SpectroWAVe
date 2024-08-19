#include "CUDASignalProcessor.h"

CUDASignalProcessor::CUDASignalProcessor(std::vector<float> signal) :
    SignalProcessor(signal) {};

CUDASignalProcessor::~CUDASignalProcessor() {};

std::vector<float> CUDASignalProcessor::getSpectraMagnitudes(int fftInputSize, int fftOutputSize, int iterationCount) {

    // TODO: Remove Debug
    //std::cout << "getSpectraMagnitudes" << std::endl;
    //std::cout << "fftInputSize: " << fftInputSize << std::endl;
    //std::cout << "fftOutputSize: " << fftOutputSize << std::endl;
    //std::cout << "iterationCount: " << iterationCount << std::endl;

    // Calculate the appropiated segment of the signal to process this iteration.
    std::vector<float> input(fftInputSize);
    auto iterationLowerBound = signal_.begin() + (iterationCount * fftInputSize);
    auto iterationUpperBound = iterationLowerBound + fftInputSize;
    if (iterationUpperBound > signal_.end()) iterationUpperBound = signal_.end(); // Catch upper limit
    input.assign(iterationLowerBound, iterationUpperBound);

    std::vector<cufftComplex> output(fftOutputSize);

    // TODO: Remove Debug
    //std::cout << "Input vector size:" << input.size() << std::endl;
    //std::cout << "Output vector size:" << output.size() << std::endl;

    CUFFT_CALL(cufftPlan1d(&plan_, fftInputSize, CUFFT_R2C, 1));

    // Create device arrays
    CUDA_RT_CALL(cudaMalloc(&d_signal_, sizeof(cufftReal) * input.size()));
    CUDA_RT_CALL(cudaMalloc(&d_signalSpectrum_, sizeof(cufftComplex) * output.size()));

    CUDA_RT_CALL(cudaMemcpy(d_signal_, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice));

    // Execute out-of-place forward FFT
    CUFFT_CALL(cufftExecR2C(plan_, d_signal_, d_signalSpectrum_));

    // Copy results back to host
    CUDA_RT_CALL(cudaMemcpy(output.data(), d_signalSpectrum_, sizeof(cufftComplex) * output.size(), cudaMemcpyDeviceToHost));

    // Calculate magnitude and phase
    std::vector<float> result(output.size(), 0);
    for (int i = 0; i < output.size(); ++i) {
        float magnitude = sqrt(output[i].x * output[i].x + output[i].y * output[i].y);
        // float phase = atan2(output[i].y, output[i].x); // Phase is not used for this project.
        result[i] = magnitude;
    }

    // TODO: Remove Debug
    //std::cout << "Input vector size:" << input.size() << std::endl;
    //std::cout << "Output vector size:" << output.size() << std::endl;
    //std::cout << "Result vector size:" << result.size() << std::endl;

    // TODO: Do we need to actually destroy everything each time? Can we reuse them and just do CudaMemcpy.
    // Clean up
    CUDA_RT_CALL(cudaFree(d_signal_));
    CUDA_RT_CALL(cudaFree(d_signalSpectrum_));
    CUDA_RT_CALL(cufftDestroy(plan_));

    return result;
}

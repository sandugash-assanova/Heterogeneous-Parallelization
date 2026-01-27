#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>

// ядро cuda для обработки части массива на gpu
__global__ void processArrayGPU(float* input, float* output, int size, int offset) {
    // вычисляем глобальный индекс с учётом смещения
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем границы массива
    if (idx < size) {
        // простая обработка - возведение в квадрат
        output[offset + idx] = input[offset + idx] * input[offset + idx];
    }
}

// функция обработки части массива на cpu
void processArrayCPU(float* input, float* output, int start, int end) {
    // обрабатываем элементы от start до end
    for (int i = start; i < end; i++) {
        // простая обработка - возведение в квадрат
        output[i] = input[i] * input[i];
    }
}

// полная обработка массива только на cpu
void fullCPUProcessing(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * input[i];
    }
}

int main() {
    // размер массива для тестирования
    const int arraySize = 10000000;
    const int bytes = arraySize * sizeof(float);
    
    // выделяем память на хосте
    float* h_input = new float[arraySize];
    float* h_outputCPU = new float[arraySize];
    float* h_outputGPU = new float[arraySize];
    float* h_outputHybrid = new float[arraySize];
    
    // инициализируем входной массив
    for (int i = 0; i < arraySize; i++) {
        h_input[i] = (float)i / 100.0f;
    }
    
    // --- вариант 1: только cpu ---
    auto cpuStart = std::chrono::high_resolution_clock::now();
    fullCPUProcessing(h_input, h_outputCPU, arraySize);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    
    // --- вариант 2: только gpu ---
    float *d_input, *d_output;
    
    // выделяем память на gpu
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // копируем данные на gpu
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // конфигурация запуска
    int threadsPerBlock = 256;
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;
    
    auto gpuStart = std::chrono::high_resolution_clock::now();
    
    // запускаем ядро для всего массива
    processArrayGPU<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, arraySize, 0);
    
    // ждём завершения
    cudaDeviceSynchronize();
    
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    auto gpuTime = std::chrono::duration_cast<std::chrono::microseconds>(gpuEnd - gpuStart).count();
    
    // копируем результат с gpu
    cudaMemcpy(h_outputGPU, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // --- вариант 3: гибридная обработка ---
    // делим массив пополам: первую половину на cpu, вторую на gpu
    int cpuPart = arraySize / 2;
    int gpuPart = arraySize - cpuPart;
    
    auto hybridStart = std::chrono::high_resolution_clock::now();
    
    // создаём поток для обработки на cpu
    std::thread cpuThread([&]() {
        processArrayCPU(h_input, h_outputHybrid, 0, cpuPart);
    });
    
    // пока cpu работает, копируем вторую часть на gpu
    cudaMemcpy(d_input + cpuPart, h_input + cpuPart, gpuPart * sizeof(float), cudaMemcpyHostToDevice);
    
    // конфигурация для gpu части
    int gpuBlocks = (gpuPart + threadsPerBlock - 1) / threadsPerBlock;
    
    // запускаем обработку на gpu
    processArrayGPU<<<gpuBlocks, threadsPerBlock>>>(d_input, d_output, gpuPart, cpuPart);
    
    // ждём завершения cpu потока
    cpuThread.join();
    
    // ждём завершения gpu
    cudaDeviceSynchronize();
    
    // копируем результат gpu части
    cudaMemcpy(h_outputHybrid + cpuPart, d_output + cpuPart, gpuPart * sizeof(float), cudaMemcpyDeviceToHost);
    
    auto hybridEnd = std::chrono::high_resolution_clock::now();
    auto hybridTime = std::chrono::duration_cast<std::chrono::microseconds>(hybridEnd - hybridStart).count();
    
    // --- проверка корректности ---
    bool correct = true;
    for (int i = 0; i < 100; i++) {
        if (abs(h_outputCPU[i] - h_outputHybrid[i]) > 0.001f) {
            correct = false;
            break;
        }
    }
    
    // --- выводим результаты ---
    std::cout << "Размер массива: " << arraySize << " элементов\n\n";
    
    std::cout << "=== Результаты ===\n";
    std::cout << "Только CPU:\n";
    std::cout << "  Время: " << cpuTime << " мкс\n\n";
    
    std::cout << "Только GPU:\n";
    std::cout << "  Время: " << gpuTime << " мкс\n";
    std::cout << "  Ускорение относительно CPU: " << (float)cpuTime / gpuTime << "x\n\n";
    
    std::cout << "Гибридная обработка (CPU + GPU):\n";
    std::cout << "  Время: " << hybridTime << " мкс\n";
    std::cout << "  Ускорение относительно CPU: " << (float)cpuTime / hybridTime << "x\n";
    std::cout << "  Отношение к GPU: " << (float)hybridTime / gpuTime << "x\n\n";
    
    std::cout << "Результат проверки: " << (correct ? "корректен" : "некорректен") << "\n";
    
    // освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_outputCPU;
    delete[] h_outputGPU;
    delete[] h_outputHybrid;
    
    return 0;
}

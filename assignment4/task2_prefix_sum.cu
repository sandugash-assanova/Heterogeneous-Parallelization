#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// ядро cuda для вычисления префиксной суммы с использованием разделяемой памяти
__global__ void prefixSum(float* input, float* output, int size) {
    // выделяем разделяемую память для блока потоков
    extern __shared__ float sharedData[];
    
    // вычисляем глобальный индекс потока
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // локальный индекс внутри блока
    int localIdx = threadIdx.x;
    
    // загружаем данные из глобальной памяти в разделяемую
    if (globalIdx < size) {
        sharedData[localIdx] = input[globalIdx];
    } else {
        sharedData[localIdx] = 0.0f;
    }
    
    // ждём пока все потоки загрузят данные
    __syncthreads();
    
    // алгоритм блочного сканирования (up-sweep фаза)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (localIdx + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            sharedData[index] += sharedData[index - stride];
        }
        // синхронизация после каждого шага
        __syncthreads();
    }
    
    // down-sweep фаза для получения эксклюзивного сканирования
    for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
        int index = (localIdx + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            sharedData[index + stride] += sharedData[index];
        }
        // синхронизация после каждого шага
        __syncthreads();
    }
    
    // записываем результат обратно в глобальную память
    if (globalIdx < size) {
        output[globalIdx] = sharedData[localIdx];
    }
}

// последовательная функция префиксной суммы на cpu
void cpuPrefixSum(float* input, float* output, int size) {
    // первый элемент остаётся без изменений
    output[0] = input[0];
    // каждый следующий элемент это сумма предыдущего результата и текущего входа
    for (int i = 1; i < size; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

int main() {
    // размер массива согласно заданию
    const int arraySize = 1000000;
    // размер в байтах
    const int bytes = arraySize * sizeof(float);
    
    // выделяем память на хосте
    float* h_input = new float[arraySize];
    float* h_outputCPU = new float[arraySize];
    float* h_outputGPU = new float[arraySize];
    
    // инициализируем входной массив
    for (int i = 0; i < arraySize; i++) {
        h_input[i] = 1.0f; // используем 1.0 для простоты проверки
    }
    
    // --- замер времени на cpu ---
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpuPrefixSum(h_input, h_outputCPU, arraySize);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    
    // --- работа с gpu ---
    float *d_input, *d_output;
    
    // выделяем память на устройстве
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // копируем входные данные на gpu
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // конфигурация запуска
    int threadsPerBlock = 256;
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;
    // размер разделяемой памяти на блок
    int sharedMemSize = threadsPerBlock * sizeof(float);
    
    // --- замер времени на gpu ---
    auto gpuStart = std::chrono::high_resolution_clock::now();
    
    // запускаем ядро с выделением разделяемой памяти
    prefixSum<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, arraySize);
    
    // ждём завершения
    cudaDeviceSynchronize();
    
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    auto gpuTime = std::chrono::duration_cast<std::chrono::microseconds>(gpuEnd - gpuStart).count();
    
    // копируем результат на хост
    cudaMemcpy(h_outputGPU, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // --- проверка корректности (сравниваем первые 10 элементов) ---
    std::cout << "Проверка корректности (первые 10 элементов):\n";
    bool correct = true;
    for (int i = 0; i < 10 && i < arraySize; i++) {
        std::cout << "CPU[" << i << "] = " << h_outputCPU[i] 
                  << ", GPU[" << i << "] = " << h_outputGPU[i] << "\n";
        if (abs(h_outputCPU[i] - h_outputGPU[i]) > 0.001f) {
            correct = false;
        }
    }
    
    // --- выводим результаты ---
    std::cout << "\nРазмер массива: " << arraySize << " элементов\n";
    std::cout << "Время CPU: " << cpuTime << " мкс\n";
    std::cout << "Время GPU: " << gpuTime << " мкс\n";
    std::cout << "Ускорение: " << (float)cpuTime / gpuTime << "x\n";
    std::cout << "Результат " << (correct ? "корректен" : "некорректен") << "\n";
    
    // освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_outputCPU;
    delete[] h_outputGPU;
    
    return 0;
}

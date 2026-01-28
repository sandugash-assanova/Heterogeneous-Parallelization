#include <iostream>          // ввод и вывод в консоль
#include <cuda_runtime.h>   // cuda runtime api
#include <chrono>           // измерение времени
#include <thread>           // std::thread

// gpu ядро для обработки части массива
__global__ void processOnGPU(float* input, float* output, int size, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
    
    if (idx < size) { // проверка выхода за границы
        float value = input[offset + idx]; // чтение элемента с учётом смещения
        for (int i = 0; i < 100; i++) { // искусственная нагрузка
            value = value * 1.01f - 0.001f; // вычисления
        }
        output[offset + idx] = value; // запись результата
    }
}

// cpu версия обработки
void processOnCPU(float* input, float* output, int start, int end) {
    for (int i = start; i < end; i++) { // перебор диапазона
        float value = input[i]; // чтение элемента
        for (int j = 0; j < 100; j++) { // та же нагрузка что на gpu
            value = value * 1.01f - 0.001f; // вычисления
        }
        output[i] = value; // запись результата
    }
}

int main() {
    const int arraySize = 10000000; // размер массива
    const int bytes = arraySize * sizeof(float); // объём данных
    
    std::cout << "=== ГИБРИДНОЕ ПРИЛОЖЕНИЕ CPU + GPU ===\n"; // заголовок
    std::cout << "Размер массива: " << arraySize << " элементов\n\n"; // вывод
    
    float* h_input = new float[arraySize]; // входной массив на cpu
    float* h_output = new float[arraySize]; // выходной массив на cpu
    
    for (int i = 0; i < arraySize; i++) { // инициализация данных
        h_input[i] = static_cast<float>(i % 1000) / 100.0f; // заполнение
    }
    
    float *h_pinnedInput, *h_pinnedOutput; // pinned memory
    cudaMallocHost(&h_pinnedInput, bytes); // закреплённая память cpu
    cudaMallocHost(&h_pinnedOutput, bytes); // закреплённая память cpu
    
    memcpy(h_pinnedInput, h_input, bytes); // копирование в pinned memory
    
    float *d_input, *d_output; // указатели gpu
    cudaMalloc(&d_input, bytes); // выделение памяти gpu
    cudaMalloc(&d_output, bytes); // выделение памяти gpu
    
    cudaStream_t stream1, stream2; // cuda streams
    cudaStreamCreate(&stream1); // создание stream1
    cudaStreamCreate(&stream2); // создание stream2
    
    cudaEvent_t start, stop; // события общего времени
    cudaEvent_t copyStart, copyEnd; // события копирования
    cudaEvent_t computeStart, computeEnd; // события вычислений
    cudaEventCreate(&start); // создание события
    cudaEventCreate(&stop); // создание события
    cudaEventCreate(&copyStart); // создание события
    cudaEventCreate(&copyEnd); // создание события
    cudaEventCreate(&computeStart); // создание события
    cudaEventCreate(&computeEnd); // создание события
    
    // ===== тест 1: только cpu =====
    std::cout << "=== ТЕСТ 1: ТОЛЬКО CPU ===\n"; // заголовок
    
    auto cpuStart = std::chrono::high_resolution_clock::now(); // старт таймера
    
    processOnCPU(h_input, h_output, 0, arraySize); // вычисление на cpu
    
    auto cpuEnd = std::chrono::high_resolution_clock::now(); // конец таймера
    auto cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        cpuEnd - cpuStart).count(); // время выполнения
    
    std::cout << "Время выполнения: " << cpuTime << " мс\n\n"; // вывод
    
    // ===== тест 2: только gpu (синхронно) =====
    std::cout << "=== ТЕСТ 2: ТОЛЬКО GPU (синхронная передача) ===\n"; // заголовок
    
    cudaEventRecord(start); // начало общего измерения
    
    cudaEventRecord(copyStart); // начало копирования
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice); // копирование на gpu
    cudaEventRecord(copyEnd); // конец копирования
    
    int threadsPerBlock = 256; // потоки в блоке
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock; // число блоков
    
    cudaEventRecord(computeStart); // начало вычислений
    processOnGPU<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, arraySize, 0); // запуск ядра
    cudaEventRecord(computeEnd); // конец вычислений
    
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost); // копирование обратно
    
    cudaEventRecord(stop); // конец общего времени
    cudaEventSynchronize(stop); // ожидание завершения
    
    float gpuTotalTime = 0; // общее время
    float gpuCopyTime = 0; // время копирования
    float gpuComputeTime = 0; // время вычислений
    
    cudaEventElapsedTime(&gpuTotalTime, start, stop); // расчёт общего времени
    cudaEventElapsedTime(&gpuCopyTime, copyStart, copyEnd); // расчёт копирования
    cudaEventElapsedTime(&gpuComputeTime, computeStart, computeEnd); // расчёт вычислений
    
    std::cout << "Общее время: " << gpuTotalTime << " мс\n"; // вывод
    std::cout << "  - Время копирования на GPU: " << gpuCopyTime << " мс\n"; // вывод
    std::cout << "  - Время вычислений: " << gpuComputeTime << " мс\n"; // вывод
    std::cout << "  - Время копирования обратно: "
              << (gpuTotalTime - gpuCopyTime - gpuComputeTime) << " мс\n"; // вывод
    std::cout << "Доля накладных расходов: "
              << ((gpuTotalTime - gpuComputeTime) / gpuTotalTime * 100) << "%\n\n"; // вывод
    
    // ===== тест 3: базовый гибрид =====
    std::cout << "=== ТЕСТ 3: ГИБРИДНЫЙ (базовый, без оптимизаций) ===\n"; // заголовок
    
    int cpuPart = arraySize * 0.3; // доля cpu
    int gpuPart = arraySize - cpuPart; // доля gpu
    
    auto hybridStart = std::chrono::high_resolution_clock::now(); // старт таймера
    
    std::thread cpuThread([&]() { // cpu в отдельном потоке
        processOnCPU(h_input, h_output, 0, cpuPart); // cpu обработка
    });
    
    cudaMemcpy(d_input + cpuPart, h_input + cpuPart,
               gpuPart * sizeof(float), cudaMemcpyHostToDevice); // копирование gpu части
    
    int gpuBlocks = (gpuPart + threadsPerBlock - 1) / threadsPerBlock; // блоки gpu
    processOnGPU<<<gpuBlocks, threadsPerBlock>>>(d_input, d_output, gpuPart, cpuPart); // запуск ядра
    
    cudaDeviceSynchronize(); // ожидание gpu
    
    cudaMemcpy(h_output + cpuPart, d_output + cpuPart,
               gpuPart * sizeof(float), cudaMemcpyDeviceToHost); // копирование результата
    
    cpuThread.join(); // ожидание cpu
    
    auto hybridEnd = std::chrono::high_resolution_clock::now(); // конец таймера
    auto hybridTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        hybridEnd - hybridStart).count(); // время выполнения
    
    std::cout << "Время выполнения: " << hybridTime << " мс\n"; // вывод
    std::cout << "Ускорение относительно CPU: "
              << (float)cpuTime / hybridTime << "x\n\n"; // ускорение
    
    // ===== тест 4: оптимизированный гибрид =====
    std::cout << "=== ТЕСТ 4: ОПТИМИЗИРОВАННЫЙ ГИБРИДНЫЙ (async + streams) ===\n"; // заголовок
    
    auto optimizedStart = std::chrono::high_resolution_clock::now(); // старт таймера
    
    std::thread cpuThreadOpt([&]() { // cpu поток
        processOnCPU(h_pinnedInput, h_pinnedOutput, 0, cpuPart); // cpu часть
    });
    
    int chunkSize = gpuPart / 2; // размер чанка
    
    int offset1 = cpuPart; // смещение первого чанка
    cudaMemcpyAsync(d_input + offset1, h_pinnedInput + offset1,
                    chunkSize * sizeof(float), cudaMemcpyHostToDevice, stream1); // async копирование
    
    int chunk1Blocks = (chunkSize + threadsPerBlock - 1) / threadsPerBlock; // блоки
    processOnGPU<<<chunk1Blocks, threadsPerBlock, 0, stream1>>>(
        d_input, d_output, chunkSize, offset1); // запуск ядра
    
    cudaMemcpyAsync(h_pinnedOutput + offset1, d_output + offset1,
                    chunkSize * sizeof(float), cudaMemcpyDeviceToHost, stream1); // async копирование
    
    int offset2 = cpuPart + chunkSize; // смещение второго чанка
    int chunk2Size = gpuPart - chunkSize; // размер второго чанка
    
    cudaMemcpyAsync(d_input + offset2, h_pinnedInput + offset2,
                    chunk2Size * sizeof(float), cudaMemcpyHostToDevice, stream2); // async копирование
    
    int chunk2Blocks = (chunk2Size + threadsPerBlock - 1) / threadsPerBlock; // блоки
    processOnGPU<<<chunk2Blocks, threadsPerBlock, 0, stream2>>>(
        d_input, d_output, chunk2Size, offset2); // запуск ядра
    
    cudaMemcpyAsync(h_pinnedOutput + offset2, d_output + offset2,
                    chunk2Size * sizeof(float), cudaMemcpyDeviceToHost, stream2); // async копирование
    
    cudaStreamSynchronize(stream1); // ожидание stream1
    cudaStreamSynchronize(stream2); // ожидание stream2
    
    cpuThreadOpt.join(); // ожидание cpu
    
    auto optimizedEnd = std::chrono::high_resolution_clock::now(); // конец таймера
    auto optimizedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        optimizedEnd - optimizedStart).count(); // время выполнения
    
    std::cout << "Время выполнения: " << optimizedTime << " мс\n"; // вывод
    std::cout << "Ускорение относительно CPU: "
              << (float)cpuTime / optimizedTime << "x\n"; // ускорение
    std::cout << "Ускорение относительно базового гибрида: "
              << (float)hybridTime / optimizedTime << "x\n\n"; // ускорение
    
    cudaStreamDestroy(stream1); // удаление stream1
    cudaStreamDestroy(stream2); // удаление stream2
    cudaEventDestroy(start); // удаление события
    cudaEventDestroy(stop); // удаление события
    cudaEventDestroy(copyStart); // удаление события
    cudaEventDestroy(copyEnd); // удаление события
    cudaEventDestroy(computeStart); // удаление события
    cudaEventDestroy(computeEnd); // удаление события
    
    cudaFree(d_input); // освобождение gpu памяти
    cudaFree(d_output); // освобождение gpu памяти
    cudaFreeHost(h_pinnedInput); // освобождение pinned memory
    cudaFreeHost(h_pinnedOutput); // освобождение pinned memory
    delete[] h_input; // освобождение cpu памяти
    delete[] h_output; // освобождение cpu памяти
    
    return 0; // завершение программы
}

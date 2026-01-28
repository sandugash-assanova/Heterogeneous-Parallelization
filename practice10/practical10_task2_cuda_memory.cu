#include <iostream>          // ввод и вывод в консоль
#include <cuda_runtime.h>   // runtime api cuda

// ядро с эффективным (коалесцированным) доступом к памяти
// потоки одного warp читают соседние адреса
__global__ void coalescedAccess(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
    
    if (idx < size) { // проверка выхода за границы
        output[idx] = input[idx] * 2.0f + 1.0f; // простое вычисление
    }
}

// ядро с неэффективным доступом к памяти
// доступ с большим шагом (stride)
__global__ void stridedAccess(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
    
    if (idx < size) { // проверка границ
        int stridedIdx = (idx * 32) % size; // индекс с шагом
        output[idx] = input[stridedIdx] * 2.0f + 1.0f; // некoалесцированный доступ
    }
}

// ядро с использованием разделяемой памяти
__global__ void sharedMemoryKernel(float* input, float* output, int size) {
    __shared__ float sharedData[256]; // разделяемая память блока
    
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс
    int localIdx = threadIdx.x; // локальный индекс в блоке
    
    if (globalIdx < size) { // проверка границ
        sharedData[localIdx] = input[globalIdx]; // загрузка в shared memory
    }
    
    __syncthreads(); // синхронизация потоков блока
    
    if (globalIdx < size) { // повторная проверка
        float value = sharedData[localIdx]; // чтение из shared memory
        value = value * 2.0f + 1.0f; // вычисление
        output[globalIdx] = value; // запись результата
    }
}

// ядро с увеличенной работой на поток
__global__ void optimizedThreadOrganization(float* input, float* output, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // базовый индекс потока
    
    if (idx < size) { // первый элемент
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
    if (idx + 1 < size) { // второй элемент
        output[idx + 1] = input[idx + 1] * 2.0f + 1.0f;
    }
    if (idx + 2 < size) { // третий элемент
        output[idx + 2] = input[idx + 2] * 2.0f + 1.0f;
    }
    if (idx + 3 < size) { // четвёртый элемент
        output[idx + 3] = input[idx + 3] * 2.0f + 1.0f;
    }
}

// функция измерения времени выполнения ядра
float measureKernelTime(void (*kernel)(float*, float*, int),
                        float* d_input, float* d_output, int size,
                        int threadsPerBlock, int numBlocks,
                        const char* kernelName) {
    cudaEvent_t start, stop; // события cuda
    cudaEventCreate(&start); // создание start
    cudaEventCreate(&stop);  // создание stop
    
    cudaEventRecord(start); // фиксация начала
    
    kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, size); // запуск ядра
    
    cudaEventRecord(stop); // фиксация конца
    cudaEventSynchronize(stop); // ожидание завершения
    
    float milliseconds = 0; // переменная времени
    cudaEventElapsedTime(&milliseconds, start, stop); // расчёт времени
    
    std::cout << kernelName << ": " << milliseconds << " мс\n"; // вывод
    
    cudaEventDestroy(start); // удаление события
    cudaEventDestroy(stop);  // удаление события
    
    return milliseconds; // возврат времени
}

int main() {
    const int arraySize = 10000000; // размер массива
    const int bytes = arraySize * sizeof(float); // объём памяти
    
    std::cout << "Размер массива: " << arraySize << " элементов\n"; // вывод
    std::cout << "Размер данных: " << bytes / (1024 * 1024) << " МБ\n\n"; // вывод
    
    float* h_input = new float[arraySize]; // массив на хосте
    float* h_output = new float[arraySize]; // массив результата
    
    for (int i = 0; i < arraySize; i++) { // инициализация входных данных
        h_input[i] = static_cast<float>(i % 1000) / 10.0f;
    }
    
    float *d_input, *d_output; // указатели на устройстве
    cudaMalloc(&d_input, bytes); // выделение памяти gpu
    cudaMalloc(&d_output, bytes); // выделение памяти gpu
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice); // копирование на gpu
    
    int threadsPerBlock = 256; // потоки в блоке
    int numBlocks = (arraySize + threadsPerBlock - 1) / threadsPerBlock; // число блоков
    
    std::cout << "=== ТЕСТИРОВАНИЕ ПАТТЕРНОВ ДОСТУПА К ПАМЯТИ ===\n"; // заголовок
    std::cout << "Конфигурация: " << numBlocks << " блоков x "
              << threadsPerBlock << " потоков\n\n"; // конфигурация
    
    float timeCoalesced = measureKernelTime(
        coalescedAccess, d_input, d_output,
        arraySize, threadsPerBlock, numBlocks,
        "1. Коалесцированный доступ"); // тест 1
    
    float timeStrided = measureKernelTime(
        stridedAccess, d_input, d_output,
        arraySize, threadsPerBlock, numBlocks,
        "2. Доступ с шагом (неэффективный)"); // тест 2
    
    float timeShared = measureKernelTime(
        sharedMemoryKernel, d_input, d_output,
        arraySize, threadsPerBlock, numBlocks,
        "3. С разделяемой памятью"); // тест 3
    
    int optimizedBlocks =
        (arraySize + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4); // блоки для оптимизации
    
    float timeOptimized = measureKernelTime(
        optimizedThreadOrganization, d_input, d_output,
        arraySize, threadsPerBlock, optimizedBlocks,
        "4. Оптимизированная организация потоков"); // тест 4
    
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost); // копирование результата
    
    for (int i = 0; i < 5; i++) { // проверка первых элементов
        float expected = h_input[i] * 2.0f + 1.0f; // ожидаемое значение
        std::cout << "Input[" << i << "] = " << h_input[i]
                  << ", Output[" << i << "] = " << h_output[i]
                  << ", Expected = " << expected << "\n"; // вывод
    }
    
    cudaFree(d_input); // освобождение памяти gpu
    cudaFree(d_output); // освобождение памяти gpu
    delete[] h_input; // освобождение памяти cpu
    delete[] h_output; // освобождение памяти cpu
    
    return 0; // завершение программы
}

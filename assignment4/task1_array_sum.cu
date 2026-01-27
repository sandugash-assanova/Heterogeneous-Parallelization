#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// функция ядра cuda для суммирования массива
// каждый поток обрабатывает один элемент массива
__global__ void arraySum(float* input, float* output, int size) {
    // вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем что индекс не выходит за границы массива
    if (idx < size) {
        // используем атомарную операцию для безопасного добавления к общей сумме
        atomicAdd(output, input[idx]);
    }
}

// последовательная функция суммирования на cpu
float cpuSum(float* array, int size) {
    float sum = 0.0f;
    // проходим по всем элементам массива
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum;
}

int main() {
    // размер массива согласно заданию
    const int arraySize = 100000;
    // размер в байтах для выделения памяти
    const int bytes = arraySize * sizeof(float);
    
    // выделяем память на хосте (cpu)
    float* h_input = new float[arraySize];
    float h_output = 0.0f;
    
    // инициализируем массив случайными значениями
    for (int i = 0; i < arraySize; i++) {
        h_input[i] = 1.0f; // используем 1.0 для простоты проверки
    }
    
    // --- начинаем замер времени на cpu ---
    auto cpuStart = std::chrono::high_resolution_clock::now();
    float cpuResult = cpuSum(h_input, arraySize);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    // вычисляем затраченное время в микросекундах
    auto cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    
    // --- начинаем работу с gpu ---
    float *d_input, *d_output;
    
    // выделяем память на устройстве (gpu)
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));
    
    // копируем данные с хоста на устройство
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    // инициализируем выходную переменную нулём
    cudaMemcpy(d_output, &h_output, sizeof(float), cudaMemcpyHostToDevice);
    
    // определяем конфигурацию запуска ядра
    int threadsPerBlock = 256; // количество потоков в блоке
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock; // количество блоков
    
    // --- начинаем замер времени на gpu ---
    auto gpuStart = std::chrono::high_resolution_clock::now();
    
    // запускаем ядро cuda
    arraySum<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, arraySize);
    
    // ждём завершения всех потоков на gpu
    cudaDeviceSynchronize();
    
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    // вычисляем затраченное время в микросекундах
    auto gpuTime = std::chrono::duration_cast<std::chrono::microseconds>(gpuEnd - gpuStart).count();
    
    // копируем результат обратно на хост
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    // --- выводим результаты ---
    std::cout << "Размер массива: " << arraySize << " элементов\n";
    std::cout << "Результат CPU: " << cpuResult << "\n";
    std::cout << "Результат GPU: " << h_output << "\n";
    std::cout << "Время CPU: " << cpuTime << " мкс\n";
    std::cout << "Время GPU: " << gpuTime << " мкс\n";
    std::cout << "Ускорение: " << (float)cpuTime / gpuTime << "x\n";
    
    // освобождаем память на gpu
    cudaFree(d_input);
    cudaFree(d_output);
    
    // освобождаем память на cpu
    delete[] h_input;
    
    return 0;
}

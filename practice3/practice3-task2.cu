#include <cuda_runtime.h> // библиотека cuda
#include <iostream> // ввод и вывод
#include <vector> // контейнер vector

__device__ void heapify(int *data, int n, int i) { // функция восстановления кучи
    int largest = i; // считаем что текущий элемент самый большой
    int left = 2 * i + 1; // индекс левого потомка
    int right = 2 * i + 2; // индекс правого потомка

    if (left < n && data[left] > data[largest]) // если левый потомок больше
        largest = left; // обновляем индекс

    if (right < n && data[right] > data[largest]) // если правый потомок больше
        largest = right; // обновляем индекс

    if (largest != i) { // если найден больший элемент
        int temp = data[i]; // сохраняем текущий элемент
        data[i] = data[largest]; // меняем элементы
        data[largest] = temp; // завершаем обмен
        heapify(data, n, largest); // рекурсивно восстанавливаем кучу
    }
}

__global__ void heapSortKernel(int *data, int n) { // kernel пирамидальной сортировки
    for (int i = n / 2 - 1; i >= 0; i--) // строим кучу
        heapify(data, n, i); // восстанавливаем кучу

    for (int i = n - 1; i >= 0; i--) { // извлекаем элементы
        int temp = data[0]; // сохраняем корень
        data[0] = data[i]; // переносим последний элемент
        data[i] = temp; // ставим корень в конец
        heapify(data, i, 0); // восстанавливаем кучу
    }
}

void gpuHeapSort(std::vector<int> &arr) { // функция запуска сортировки
    int *deviceData; // память gpu
    int size = arr.size(); // размер массива

    cudaMalloc(&deviceData, size * sizeof(int)); // выделяем память
    cudaMemcpy(deviceData, arr.data(), size * sizeof(int), cudaMemcpyHostToDevice); // копируем данные

    heapSortKernel<<<1, 1>>>(deviceData, size); // запускаем kernel
    cudaDeviceSynchronize(); // ждем завершения

    cudaMemcpy(arr.data(), deviceData, size * sizeof(int), cudaMemcpyDeviceToHost); // копируем результат
    cudaFree(deviceData); // освобождаем память
}

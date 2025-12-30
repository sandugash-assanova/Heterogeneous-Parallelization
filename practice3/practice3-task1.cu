#include <cuda_runtime.h> // подключаем библиотеку для работы с cuda
#include <iostream> // подключаем ввод и вывод
#include <vector> // подключаем контейнер vector
#include <algorithm> // подключаем стандартные алгоритмы

__device__ void deviceSwap(int &a, int &b) { // функция обмена элементов на gpu
    int temp = a; // сохраняем значение a во временную переменную
    a = b; // присваиваем a значение b
    b = temp; // присваиваем b сохраненное значение
}

__global__ void quickSortKernel(int *data, int left, int right) { // kernel для быстрой сортировки
    int i = left; // левая граница
    int j = right; // правая граница
    int pivot = data[(left + right) / 2]; // выбираем опорный элемент

    while (i <= j) { // пока индексы не пересеклись
        while (data[i] < pivot) i++; // ищем элемент больше опорного
        while (data[j] > pivot) j--; // ищем элемент меньше опорного

        if (i <= j) { // если индексы корректны
            deviceSwap(data[i], data[j]); // меняем элементы местами
            i++; // сдвигаем левый индекс
            j--; // сдвигаем правый индекс
        }
    }
}

void gpuQuickSort(std::vector<int> &arr) { // функция запуска сортировки
    int *deviceData; // указатель на память gpu
    int size = arr.size(); // размер массива

    cudaMalloc(&deviceData, size * sizeof(int)); // выделяем память на gpu
    cudaMemcpy(deviceData, arr.data(), size * sizeof(int), cudaMemcpyHostToDevice); // копируем данные

    quickSortKernel<<<1, 1>>>(deviceData, 0, size - 1); // запускаем kernel
    cudaDeviceSynchronize(); // ждем завершения

    cudaMemcpy(arr.data(), deviceData, size * sizeof(int), cudaMemcpyDeviceToHost); // копируем результат
    cudaFree(deviceData); // освобождаем память
}

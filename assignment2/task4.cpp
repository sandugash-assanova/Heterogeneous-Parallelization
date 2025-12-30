#include <iostream>        // ввод и вывод в консоль
#include <vector>          // контейнер vector
#include <chrono>          // измерение времени
#include <random>          // генерация случайных чисел
#include <cuda_runtime.h>  // работа с cuda api

using namespace std;       // используем пространство имен std

// функция слияния двух отсортированных частей массива на gpu
// выполняется на устройстве
__device__ void mergeDevice(int* arr, int left, int mid, int right, int* temp) {
    int i = left;          // индекс левой части
    int j = mid + 1;       // индекс правой части
    int k = left;          // индекс для временного массива
    
    // сливаем пока обе части не закончились
    while (i <= mid && j <= right) {
        // если элемент слева меньше или равен
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++]; // берем элемент из левой части
        } else {
            temp[k++] = arr[j++]; // берем элемент из правой части
        }
    }
    
    // копируем оставшиеся элементы левой части
    while (i <= mid) {
        temp[k++] = arr[i++];
    }
    
    // копируем оставшиеся элементы правой части
    while (j <= right) {
        temp[k++] = arr[j++];
    }
    
    // копируем результат обратно в основной массив
    for (int i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}

// kernel для сортировки маленьких подмассивов
// каждый поток работает со своим куском
__global__ void mergeSortKernel(int* arr, int* temp, int n, int chunkSize) {
    // глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // левая граница куска
    int left = tid * chunkSize;
    
    // проверяем выход за границы массива
    if (left >= n) return;
    
    // правая граница куска
    int right = min(left + chunkSize - 1, n - 1);
    
    // сортировка вставками для маленького массива
    for (int i = left + 1; i <= right; i++) {
        int key = arr[i];  // текущий элемент
        int j = i - 1;     // индекс слева
        
        // сдвигаем элементы больше key вправо
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        // вставляем key на нужное место
        arr[j + 1] = key;
    }
}

// kernel для слияния отсортированных кусков
__global__ void mergeKernel(int* arr, int* temp, int n, int chunkSize) {
    // глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // левая граница пары кусков
    int left = tid * 2 * chunkSize;
    
    // проверяем границы
    if (left >= n) return;
    
    // середина массива
    int mid = min(left + chunkSize - 1, n - 1);
    // правая граница
    int right = min(left + 2 * chunkSize - 1, n - 1);
    
    // если есть что сливать
    if (mid < right) {
        mergeDevice(arr, left, mid, right, temp); // вызываем device функцию
    }
}

// проверка что массив отсортирован
bool isSorted(const vector<int>& arr) {
    // проходим по массиву
    for (size_t i = 1; i < arr.size(); i++) {
        // если порядок нарушен
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true; // массив отсортирован
}

// основная функция сортировки слиянием на gpu
void mergeSortCUDA(vector<int>& arr) {
    int n = arr.size(); // размер массива
    
    // указатели на память gpu
    int *d_arr, *d_temp;
    
    // выделяем память под массив
    cudaMalloc(&d_arr, n * sizeof(int));
    // выделяем память под временный массив
    cudaMalloc(&d_temp, n * sizeof(int));
    
    // копируем данные с cpu на gpu
    cudaMemcpy(d_arr, arr.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    
    // размер блока потоков
    int blockSize = 256;
    
    // начальный размер куска
    int chunkSize = 32;
    
    // считаем количество кусков
    int numChunks = (n + chunkSize - 1) / chunkSize;
    // считаем количество блоков
    int gridSize = (numChunks + blockSize - 1) / blockSize;
    
    // первичная сортировка маленьких кусков
    mergeSortKernel<<<gridSize, blockSize>>>(d_arr, d_temp, n, chunkSize);
    
    // ждем завершения kernel
    cudaDeviceSynchronize();
    
    // поэтапное слияние кусков
    while (chunkSize < n) {
        // количество операций слияния
        int numMerges = (n + 2 * chunkSize - 1) / (2 * chunkSize);
        // пересчитываем сетку
        gridSize = (numMerges + blockSize - 1) / blockSize;
        
        // запуск kernel слияния
        mergeKernel<<<gridSize, blockSize>>>(d_arr, d_temp, n, chunkSize);
        
        // ждем завершения
        cudaDeviceSynchronize();
        
        // увеличиваем размер куска
        chunkSize *= 2;
    }
    
    // копируем результат обратно на cpu
    cudaMemcpy(arr.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // освобождаем память gpu
    cudaFree(d_arr);
    cudaFree(d_temp);
}

// реализация merge на cpu
void mergeCPU(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1); // временный массив
    int i = left, j = mid + 1, k = 0;   // индексы
    
    // слияние двух частей
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    // копируем остаток левой части
    while (i <= mid) {
        temp[k++] = arr[i++];
    }
    
    // копируем остаток правой части
    while (j <= right) {
        temp[k++] = arr[j++];
    }
    
    // переносим результат в основной массив
    for (int i = 0; i < k; i++) {
        arr[left + i] = temp[i];
    }
}

// рекурсивная сортировка слиянием на cpu
void mergeSortCPU(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2; // находим середину
        mergeSortCPU(arr, left, mid);        // сортируем левую часть
        mergeSortCPU(arr, mid + 1, right);   // сортируем правую часть
        mergeCPU(arr, left, mid, right);     // сливаем части
    }
}

// функция тестирования
void testMergeSort(int size) {
    cout << "===== тест на массиве из " << size << " элементов =====" << endl;
    
    vector<int> arr1(size); // массив для cpu
    vector<int> arr2(size); // массив для gpu
    
    random_device rd;       // источник энтропии
    mt19937 gen(rd());      // генератор
    uniform_int_distribution<> dis(1, 100000); // диапазон чисел
    
    // заполняем массивы
    for (int i = 0; i < size; i++) {
        int value = dis(gen);
        arr1[i] = value;
        arr2[i] = value;
    }
    
    // вывод первых элементов
    cout << "первые 10 элементов: ";
    for (int i = 0; i < 10; i++) {
        cout << arr1[i] << " ";
    }
    cout << endl;
    
    // замер cpu сортировки
    auto startCPU = chrono::high_resolution_clock::now();
    mergeSortCPU(arr1, 0, size - 1);
    auto endCPU = chrono::high_resolution_clock::now();
    auto durationCPU = chrono::duration_cast<chrono::milliseconds>(endCPU - startCPU);
    
    cout << "cpu сортировка: " << durationCPU.count() << " мс" << endl;
    cout << "отсортирован: " << (isSorted(arr1) ? "да" : "нет") << endl;
    
    // замер gpu сортировки
    auto startGPU = chrono::high_resolution_clock::now();
    mergeSortCUDA(arr2);
    auto endGPU = chrono::high_resolution_clock::now();
    auto durationGPU = chrono::duration_cast<chrono::milliseconds>(endGPU - startGPU);
    
    cout << "gpu сортировка: " << durationGPU.count() << " мс" << endl;
    cout << "отсортирован: " << (isSorted(arr2) ? "да" : "нет") << endl;
    
    // проверяем совпадение результатов
    bool equal = (arr1 == arr2);
    cout << "результаты совпадают: " << (equal ? "да" : "нет") << endl;
    
    // считаем ускорение
    double speedup = (double)durationCPU.count() / durationGPU.count();
    cout << "ускорение: " << speedup << "x" << endl;
    
    // вывод первых элементов после сортировки
    cout << "первые 10 элементов после сортировки: ";
    for (int i = 0; i < 10; i++) {
        cout << arr2[i] << " ";
    }
    cout << endl << endl;
}

int main() {
    int deviceCount; // количество cuda устройств
    cudaGetDeviceCount(&deviceCount); // получаем число устройств
    
    // если gpu не найден
    if (deviceCount == 0) {
        cout << "cuda устройства не найдены" << endl;
        return 1;
    }
    
    // получаем свойства gpu
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "используется gpu: " << prop.name << endl;
    cout << "compute capability: " << prop.major << "." << prop.minor << endl;
    cout << "количество sm: " << prop.multiProcessorCount << endl << endl;
    
    // тесты
    testMergeSort(10000);
    testMergeSort(100000);
    
    // выводы
    cout << "===== выводы =====" << endl;
    cout << "сортировка слиянием хорошо параллелится на gpu" << endl;
    cout << "gpu эффективен на больших массивах" << endl;
    cout << "на маленьких массивах накладные расходы велики" << endl;
    
    return 0; // завершение программы
}

// компиляция: nvcc task4.cu -o task4
// запуск: ./task4

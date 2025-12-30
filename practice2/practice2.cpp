#include <iostream>    // ввод и вывод в консоль
#include <vector>      // контейнер vector
#include <chrono>      // измерение времени
#include <random>      // генерация случайных чисел
#include <omp.h>       // библиотека openmp
#include <algorithm>   // swap и min
#include <iomanip>     // форматированный вывод

using namespace std;   // используем пространство имен std

// ============= СОРТИРОВКА ПУЗЫРЬКОМ =============

// последовательная версия
void bubbleSortSequential(vector<int>& arr) {
    int n = arr.size(); // размер массива
    
    // внешний цикл определяет количество проходов
    for (int i = 0; i < n - 1; i++) {
        // внутренний цикл сравнивает соседние элементы
        for (int j = 0; j < n - i - 1; j++) {
            // если элементы стоят в неправильном порядке
            if (arr[j] > arr[j + 1]) {
                // меняем их местами
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// параллельная версия пузырька (четно-нечетная)
void bubbleSortParallel(vector<int>& arr) {
    int n = arr.size();      // размер массива
    bool swapped = true;    // флаг наличия обменов
    
    // выполняем пока были обмены
    while (swapped) {
        swapped = false;    // сбрасываем флаг
        
        // четный проход
        #pragma omp parallel for
        for (int i = 0; i < n - 1; i += 2) {
            // сравниваем пары (0,1), (2,3) и т.д.
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]); // выполняем обмен
                swapped = true;           // отмечаем что был обмен
            }
        }
        
        // нечетный проход
        #pragma omp parallel for
        for (int i = 1; i < n - 1; i += 2) {
            // сравниваем пары (1,2), (3,4) и т.д.
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]); // выполняем обмен
                swapped = true;           // отмечаем обмен
            }
        }
    }
}

// ============= СОРТИРОВКА ВЫБОРОМ =============

// последовательная версия
void selectionSortSequential(vector<int>& arr) {
    int n = arr.size(); // размер массива
    
    // внешний цикл по позициям
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i; // считаем текущий элемент минимальным
        
        // ищем минимум справа
        for (int j = i + 1; j < n; j++) {
            // если найден элемент меньше
            if (arr[j] < arr[minIndex]) {
                minIndex = j; // обновляем индекс минимума
            }
        }
        
        // если минимум не на текущей позиции
        if (minIndex != i) {
            swap(arr[i], arr[minIndex]); // меняем элементы
        }
    }
}

// параллельная версия сортировки выбором
void selectionSortParallel(vector<int>& arr) {
    int n = arr.size(); // размер массива
    
    // внешний цикл последовательный
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;     // индекс минимума
        int minValue = arr[i]; // значение минимума
        
        // параллельная область
        #pragma omp parallel
        {
            int localMinIndex = i;     // локальный индекс минимума
            int localMinValue = arr[i]; // локальное значение минимума
            
            // параллельный поиск минимума
            #pragma omp for
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < localMinValue) {
                    localMinValue = arr[j]; // обновляем локальный минимум
                    localMinIndex = j;
                }
            }
            
            // критическая секция
            #pragma omp critical
            {
                if (localMinValue < minValue) {
                    minValue = localMinValue; // обновляем глобальный минимум
                    minIndex = localMinIndex;
                }
            }
        }
        
        // выполняем обмен
        if (minIndex != i) {
            swap(arr[i], arr[minIndex]);
        }
    }
}

// ============= СОРТИРОВКА ВСТАВКОЙ =============

// последовательная версия
void insertionSortSequential(vector<int>& arr) {
    int n = arr.size(); // размер массива
    
    // начинаем со второго элемента
    for (int i = 1; i < n; i++) {
        int key = arr[i]; // элемент для вставки
        int j = i - 1;    // индекс слева
        
        // сдвигаем элементы вправо
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j]; // сдвиг элемента
            j--;
        }
        
        // вставляем элемент
        arr[j + 1] = key;
    }
}

// параллельная версия сортировки вставкой
void insertionSortParallel(vector<int>& arr) {
    int n = arr.size(); // размер массива
    int numThreads = omp_get_max_threads(); // количество потоков
    int chunkSize = n / numThreads; // размер куска
    
    // параллельная сортировка кусков
    #pragma omp parallel
    {
        int threadId = omp_get_thread_num(); // номер потока
        int start = threadId * chunkSize;   // начало куска
        int end = (threadId == numThreads - 1) ? n : start + chunkSize;
        
        // сортировка вставкой внутри куска
        for (int i = start + 1; i < end; i++) {
            int key = arr[i];
            int j = i - 1;
            
            while (j >= start && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }
    
    // последовательное слияние кусков
    for (int size = chunkSize; size < n; size *= 2) {
        for (int start = 0; start < n; start += 2 * size) {
            int mid = min(start + size - 1, n - 1);
            int end = min(start + 2 * size - 1, n - 1);
            
            if (mid < end) {
                vector<int> temp(end - start + 1); // временный массив
                int i = start, j = mid + 1, k = 0;
                
                // слияние
                while (i <= mid && j <= end) {
                    if (arr[i] <= arr[j]) {
                        temp[k++] = arr[i++];
                    } else {
                        temp[k++] = arr[j++];
                    }
                }
                
                while (i <= mid) temp[k++] = arr[i++];
                while (j <= end) temp[k++] = arr[j++];
                
                // копируем результат
                for (int i = 0; i < k; i++) {
                    arr[start + i] = temp[i];
                }
            }
        }
    }
}

// ============= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =============

// проверка сортировки
bool isSorted(const vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) {
            return false; // порядок нарушен
        }
    }
    return true; // массив отсортирован
}

int main() {
    omp_set_num_threads(4); // используем 4 потока
    cout << "Используется потоков: " << omp_get_max_threads() << endl;

    // размеры массивов для теста
    vector<int> sizes = {10, 100, 1000}; 

    for (int size : sizes) {
        cout << "\n=== Тест на массиве из " << size << " элементов ===\n";

        // создаем два одинаковых массива
        vector<int> arr(size);
        vector<int> arrCopy(size);

        // генератор случайных чисел
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(1, 1000);

        for (int i = 0; i < size; i++) {
            arr[i] = dis(gen);
            arrCopy[i] = arr[i];
        }

        // 1. Пузырёк
        cout << "\n--- Сортировка пузырьком ---\n";
        bubbleSortSequential(arr);
        cout << "последовательная отсортирован: " << (isSorted(arr) ? "да" : "нет") << endl;

        arr = arrCopy;
        bubbleSortParallel(arr);
        cout << "параллельная отсортирован: " << (isSorted(arr) ? "да" : "нет") << endl;

        // 2. Выбором
        cout << "\n--- Сортировка выбором ---\n";
        arr = arrCopy;
        selectionSortSequential(arr);
        cout << "последовательная отсортирован: " << (isSorted(arr) ? "да" : "нет") << endl;

        arr = arrCopy;
        selectionSortParallel(arr);
        cout << "параллельная отсортирован: " << (isSorted(arr) ? "да" : "нет") << endl;

        // 3. Вставкой
        cout << "\n--- Сортировка вставкой ---\n";
        arr = arrCopy;
        insertionSortSequential(arr);
        cout << "последовательная отсортирован: " << (isSorted(arr) ? "да" : "нет") << endl;

        arr = arrCopy;
        insertionSortParallel(arr);
        cout << "параллельная отсортирован: " << (isSorted(arr) ? "да" : "нет") << endl;
    }

    cout << "\n=== Тест завершён ===" << endl;
    return 0;
}

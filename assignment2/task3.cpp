#include <iostream>   // ввод и вывод в консоль
#include <vector>     // контейнер vector
#include <chrono>     // измерение времени
#include <random>     // генерация случайных чисел
#include <omp.h>      // библиотека openmp
#include <algorithm> // swap и min

using namespace std;  // используем пространство имен std

// последовательная сортировка выбором
void selectionSortSequential(vector<int>& arr) {
    int n = arr.size(); // размер массива
    
    // внешний цикл по всем позициям массива
    for (int i = 0; i < n - 1; i++) {
        // считаем текущий элемент минимальным
        int minIndex = i;
        
        // ищем минимальный элемент справа от текущего
        for (int j = i + 1; j < n; j++) {
            // если найден элемент меньше текущего минимума
            if (arr[j] < arr[minIndex]) {
                minIndex = j; // запоминаем индекс минимума
            }
        }
        
        // если минимум не на текущей позиции
        if (minIndex != i) {
            swap(arr[i], arr[minIndex]); // меняем элементы местами
        }
    }
}

// параллельная сортировка выбором с openmp
void selectionSortParallel(vector<int>& arr) {
    int n = arr.size(); // размер массива
    
    // внешний цикл нельзя параллелить из-за зависимости шагов
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;      // индекс текущего минимума
        int minValue = arr[i]; // значение текущего минимума
        
        // создаем параллельную область
        #pragma omp parallel
        {
            // локальный индекс минимума для потока
            int localMinIndex = i;
            // локальное значение минимума для потока
            int localMinValue = arr[i];
            
            // параллельный цикл поиска минимума
            #pragma omp for schedule(static)
            for (int j = i + 1; j < n; j++) {
                // ищем минимум в своей части массива
                if (arr[j] < localMinValue) {
                    localMinValue = arr[j]; // обновляем локальный минимум
                    localMinIndex = j;      // сохраняем индекс
                }
            }
            
            // критическая секция для объединения результатов
            #pragma omp critical
            {
                // сравниваем локальный минимум с глобальным
                if (localMinValue < minValue) {
                    minValue = localMinValue; // обновляем глобальный минимум
                    minIndex = localMinIndex; // обновляем индекс минимума
                }
            }
        }
        
        // меняем найденный минимум с текущей позицией
        if (minIndex != i) {
            swap(arr[i], arr[minIndex]); // выполняем обмен
        }
    }
}

// функция проверки отсортированности массива
bool isSorted(const vector<int>& arr) {
    // проходим по всем элементам массива
    for (size_t i = 1; i < arr.size(); i++) {
        // если нарушен порядок
        if (arr[i] < arr[i - 1]) {
            return false; // массив не отсортирован
        }
    }
    return true; // массив отсортирован
}

// функция тестирования сортировки
void testSorting(int size) {
    cout << "===== тестирование на массиве из " << size << " элементов =====" << endl;
    
    // создаем первый массив
    vector<int> arr1(size);
    // создаем второй массив
    vector<int> arr2(size);
    
    // источник случайных чисел
    random_device rd;
    // генератор mt19937
    mt19937 gen(rd());
    // распределение значений
    uniform_int_distribution<> dis(1, 10000);
    
    // заполняем массивы одинаковыми значениями
    for (int i = 0; i < size; i++) {
        int value = dis(gen); // генерируем число
        arr1[i] = value;      // записываем в первый массив
        arr2[i] = value;      // записываем во второй массив
    }
    
    // вывод первых элементов до сортировки
    cout << "первые 10 элементов до сортировки: ";
    for (int i = 0; i < min(10, size); i++) {
        cout << arr1[i] << " ";
    }
    cout << endl;
    
    // замер времени последовательной сортировки
    auto startSeq = chrono::high_resolution_clock::now();
    selectionSortSequential(arr1); // выполняем сортировку
    auto endSeq = chrono::high_resolution_clock::now();
    auto durationSeq = chrono::duration_cast<chrono::milliseconds>(endSeq - startSeq);
    
    // вывод результатов последовательной версии
    cout << "последовательная сортировка: " << durationSeq.count() << " миллисекунд" << endl;
    cout << "массив отсортирован: " << (isSorted(arr1) ? "да" : "нет") << endl;
    
    // вывод первых элементов после сортировки
    cout << "первые 10 элементов после сортировки: ";
    for (int i = 0; i < min(10, size); i++) {
        cout << arr1[i] << " ";
    }
    cout << endl << endl;
    
    // замер времени параллельной сортировки
    auto startPar = chrono::high_resolution_clock::now();
    selectionSortParallel(arr2); // выполняем параллельную сортировку
    auto endPar = chrono::high_resolution_clock::now();
    auto durationPar = chrono::duration_cast<chrono::milliseconds>(endPar - startPar);
    
    // вывод результатов параллельной версии
    cout << "параллельная сортировка: " << durationPar.count() << " миллисекунд" << endl;
    cout << "массив отсортирован: " << (isSorted(arr2) ? "да" : "нет") << endl;
    
    // проверка совпадения массивов
    bool arraysEqual = (arr1 == arr2);
    cout << "результаты совпадают: " << (arraysEqual ? "да" : "нет") << endl;
    
    // расчет ускорения
    double speedup = (double)durationSeq.count() / durationPar.count();
    cout << "ускорение: " << speedup << "x" << endl;
    
    // анализ полученных результатов
    cout << "анализ:" << endl;
    if (speedup > 1.0) {
        cout << "параллельная версия работает быстрее" << endl;
    } else if (speedup < 1.0) {
        cout << "параллельная версия медленнее из-за накладных расходов" << endl;
        cout << "сортировка выбором плохо параллелится" << endl;
    } else {
        cout << "обе версии работают одинаково" << endl;
    }
    
    cout << endl;
}

int main() {
    // задаем количество потоков openmp
    omp_set_num_threads(4);
    // вывод количества используемых потоков
    cout << "используется потоков: " << omp_get_max_threads() << endl << endl;
    
    // тест на массиве из 1000 элементов
    testSorting(1000);
    // тест на массиве из 10000 элементов
    testSorting(10000);
    
    // общие выводы программы
    cout << "===== общие выводы =====" << endl;
    cout << "сортировка выбором имеет сложность o(n^2)" << endl;
    cout << "она плохо подходит для параллелизации" << endl;
    cout << "так как каждый шаг зависит от предыдущего" << endl;
    cout << "распараллелить можно только поиск минимума" << endl;
    cout << "накладные расходы часто превышают выигрыш" << endl;
    cout << "для больших данных лучше использовать другие алгоритмы" << endl;
    
    return 0; // завершение программы
}

#include <iostream>   // ввод и вывод в консоль
#include <vector>     // контейнер vector
#include <chrono>     // измерение времени
#include <random>     // генерация случайных чисел
#include <omp.h>      // поддержка openmp
#include <limits>     // числовые пределы типов

using namespace std;  // чтобы не писать std:: каждый раз

// функция для последовательного поиска минимума и максимума
void findMinMaxSequential(const vector<int>& arr, int& minVal, int& maxVal) {
    // устанавливаем первый элемент как начальный минимум
    minVal = arr[0];
    // устанавливаем первый элемент как начальный максимум
    maxVal = arr[0];
    
    // цикл по массиву начиная со второго элемента
    for (size_t i = 1; i < arr.size(); i++) {
        // проверяем меньше ли текущий элемент минимума
        if (arr[i] < minVal) {
            // обновляем минимум
            minVal = arr[i];
        }
        // проверяем больше ли текущий элемент максимума
        if (arr[i] > maxVal) {
            // обновляем максимум
            maxVal = arr[i];
        }
    }
}

// функция для параллельного поиска минимума и максимума
void findMinMaxParallel(const vector<int>& arr, int& minVal, int& maxVal) {
    // задаем максимально возможное значение int
    minVal = numeric_limits<int>::max();
    // задаем минимально возможное значение int
    maxVal = numeric_limits<int>::min();
    
    // параллельный цикл openmp
    // reduction(min:minVal) объединяет минимумы потоков
    // reduction(max:maxVal) объединяет максимумы потоков
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < arr.size(); i++) {
        // сравниваем элемент с текущим минимумом потока
        if (arr[i] < minVal) {
            // обновляем минимум потока
            minVal = arr[i];
        }
        // сравниваем элемент с текущим максимумом потока
        if (arr[i] > maxVal) {
            // обновляем максимум потока
            maxVal = arr[i];
        }
    }
}

int main() {
    const int SIZE = 10000;          // размер массива
    vector<int> arr(SIZE);           // создаем массив заданного размера
    
    // источник случайности от системы
    random_device rd;
    // генератор mt19937
    mt19937 gen(rd());
    // равномерное распределение от 1 до 100000
    uniform_int_distribution<> dis(1, 100000);
    
    // заполняем массив случайными числами
    for (int i = 0; i < SIZE; i++) {
        // присваиваем случайное число
        arr[i] = dis(gen);
    }
    
    // вывод информации о массиве
    cout << "массив из " << SIZE << " элементов создан" << endl;
    // вывод первых элементов для проверки
    cout << "первые 10 элементов: ";
    // цикл для вывода первых 10 чисел
    for (int i = 0; i < 10; i++) {
        // печатаем элемент
        cout << arr[i] << " ";
    }
    // перевод строки
    cout << endl << endl;
    
    // переменные для последовательного результата
    int minSeq, maxSeq;
    // переменные для параллельного результата
    int minPar, maxPar;
    
    // фиксируем время начала последовательного алгоритма
    auto startSeq = chrono::high_resolution_clock::now();
    // вызываем последовательную функцию
    findMinMaxSequential(arr, minSeq, maxSeq);
    // фиксируем время окончания
    auto endSeq = chrono::high_resolution_clock::now();
    
    // считаем длительность выполнения
    auto durationSeq = chrono::duration_cast<chrono::microseconds>(endSeq - startSeq);
    
    // вывод результатов последовательного алгоритма
    cout << "последовательное выполнение:" << endl;
    cout << "минимум: " << minSeq << endl;
    cout << "максимум: " << maxSeq << endl;
    cout << "время: " << durationSeq.count() << " микросекунд" << endl << endl;
    
    // фиксируем время начала параллельного алгоритма
    auto startPar = chrono::high_resolution_clock::now();
    // вызываем параллельную функцию
    findMinMaxParallel(arr, minPar, maxPar);
    // фиксируем время окончания
    auto endPar = chrono::high_resolution_clock::now();
    
    // считаем длительность параллельного выполнения
    auto durationPar = chrono::duration_cast<chrono::microseconds>(endPar - startPar);
    
    // вывод результатов параллельного алгоритма
    cout << "параллельное выполнение (openmp):" << endl;
    cout << "минимум: " << minPar << endl;
    cout << "максимум: " << maxPar << endl;
    cout << "время: " << durationPar.count() << " микросекунд" << endl << endl;
    
    // проверяем совпадают ли результаты
    if (minSeq == minPar && maxSeq == maxPar) {
        // если совпадают
        cout << "результаты совпадают - все правильно" << endl;
    } else {
        // если есть ошибка
        cout << "результаты не совпадают - что-то пошло не так" << endl;
    }
    
    // считаем коэффициент ускорения
    double speedup = (double)durationSeq.count() / durationPar.count();
    // выводим ускорение
    cout << "ускорение: " << speedup << "x" << endl;
    
    // выводим текстовые выводы
    cout << endl << "выводы:" << endl;
    // если параллельная версия быстрее
    if (speedup > 1.0) {
        cout << "параллельная версия быстрее в " << speedup << " раз" << endl;
    }
    // если параллельная версия медленнее
    else if (speedup < 1.0) {
        cout << "параллельная версия медленнее, возможно массив слишком маленький" << endl;
        cout << "накладные расходы на потоки выше выигрыша" << endl;
    }
    // если скорости примерно равны
    else {
        cout << "обе версии работают примерно одинаково" << endl;
    }
    
    return 0; // успешное завершение программы
}

// команда компиляции с поддержкой openmp
// g++ -fopenmp task2.cpp -o task2
// команда запуска программы
// ./task2

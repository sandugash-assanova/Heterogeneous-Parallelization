#include <iostream> // подключение библиотеки ввода-вывода
#include <vector> // подключение библиотеки для работы с векторами
#include <random> // подключение библиотеки для генерации случайных чисел
#include <chrono> // подключение библиотеки для измерения времени
#include <omp.h> // подключение библиотеки openmp для параллелизации
#include <iomanip> // подключение библиотеки для форматирования вывода
#include <limits> // подключение библиотеки для работы с предельными значениями типов

// задание 1: динамическое выделение памяти и вычисление среднего
void task1() {
    std::cout << "\n=== Задание 1 ===" << std::endl; // вывод заголовка задания
    
    const int SIZE = 50000; // константа размера массива
    int* array = new int[SIZE]; // динамическое выделение памяти для массива
    
    std::random_device rd; // инициализация генератора случайных чисел
    std::mt19937 gen(rd()); // создание генератора мерсенна на основе random_device
    std::uniform_int_distribution<> dis(1, 100); // создание равномерного распределения от 1 до 100
    
    for (int i = 0; i < SIZE; i++) { // цикл по всем элементам массива
        array[i] = dis(gen); // заполнение элемента случайным числом
    }
    
    long long sum = 0; // переменная для накопления суммы элементов
    for (int i = 0; i < SIZE; i++) { // цикл для суммирования элементов
        sum += array[i]; // добавление текущего элемента к сумме
    }
    
    double average = static_cast<double>(sum) / SIZE; // вычисление среднего значения
    std::cout << "Средее значение: " << std::fixed << std::setprecision(2) << average << std::endl; // вывод среднего с двумя знаками после запятой
    
    delete[] array; // освобождение динамически выделенной памяти
}

// задание 2: последовательный поиск минимума и максимума
void task2() {
    std::cout << "\n=== Задание 2 ===" << std::endl; // вывод заголовка задания
    
    const int SIZE = 1000000; // константа размера массива
    std::vector<int> array(SIZE); // создание вектора заданного размера
    
    std::random_device rd; // инициализация генератора случайных чисел
    std::mt19937 gen(rd()); // создание генератора мерсенна
    std::uniform_int_distribution<> dis(1, 1000); // создание распределения от 1 до 1000
    
    for (int i = 0; i < SIZE; i++) { // цикл заполнения массива
        array[i] = dis(gen); // заполнение элемента случайным числом
    }
    
    auto start = std::chrono::high_resolution_clock::now(); // запуск таймера
    
    int min_val = array[0]; // инициализация минимума первым элементом
    int max_val = array[0]; // инициализация максимума первым элементом
    
    for (int i = 1; i < SIZE; i++) { // цикл по остальным элементам массива
        if (array[i] < min_val) { // проверка на новый минимум
            min_val = array[i]; // обновление минимума
        }
        if (array[i] > max_val) { // проверка на новый максимум
            max_val = array[i]; // обновление максимума
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now(); // остановка таймера
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); // вычисление времени выполнения в микросекундах
    
    std::cout << "Минимум: " << min_val << std::endl; // вывод минимального значения
    std::cout << "Максимум: " << max_val << std::endl; // вывод максимального значения
    std::cout << "Время выполнения (последовательно): " << duration.count() << " мкс" << std::endl; // вывод времени выполнения
}

// задание 3: параллельный поиск минимума и максимума с openmp
void task3() {
    std::cout << "\n=== Задание 3 ===" << std::endl; // вывод заголовка задания
    
    const int SIZE = 1000000; // константа размера массива
    std::vector<int> array(SIZE); // создание вектора заданного размера
    
    std::random_device rd; // инициализация генератора случайных чисел
    std::mt19937 gen(rd()); // создание генератора мерсенна
    std::uniform_int_distribution<> dis(1, 1000); // создание распределения от 1 до 1000
    
    for (int i = 0; i < SIZE; i++) { // цикл заполнения массива
        array[i] = dis(gen); // заполнение элемента случайным числом
    }
    
    // последовательная версия для сравнения
    auto start_seq = std::chrono::high_resolution_clock::now(); // запуск таймера для последовательной версии
    
    int min_val_seq = array[0]; // инициализация минимума
    int max_val_seq = array[0]; // инициализация максимума
    
    for (int i = 1; i < SIZE; i++) { // цикл поиска минимума и максимума
        if (array[i] < min_val_seq) min_val_seq = array[i]; // обновление минимума
        if (array[i] > max_val_seq) max_val_seq = array[i]; // обновление максимума
    }
    
    auto end_seq = std::chrono::high_resolution_clock::now(); // остановка таймера
    auto duration_seq = std::chrono::duration_cast<std::chrono::microseconds>(end_seq - start_seq); // вычисление времени выполнения
    
    // параллельная версия с openmp
    auto start_par = std::chrono::high_resolution_clock::now(); // запуск таймера для параллельной версии
    
    int min_val_par = std::numeric_limits<int>::max(); // инициализация минимума максимальным значением int
    int max_val_par = std::numeric_limits<int>::min(); // инициализация максимума минимальным значением int
    
    #pragma omp parallel for reduction(min:min_val_par) reduction(max:max_val_par) // директива openmp для параллельного цикла с редукцией
    for (int i = 0; i < SIZE; i++) { // параллельный цикл по элементам массива
        if (array[i] < min_val_par) min_val_par = array[i]; // обновление локального минимума
        if (array[i] > max_val_par) max_val_par = array[i]; // обновление локального максимума
    }
    
    auto end_par = std::chrono::high_resolution_clock::now(); // остановка таймера
    auto duration_par = std::chrono::duration_cast<std::chrono::microseconds>(end_par - start_par); // вычисление времени выполнения
    
    std::cout << "Последовательно:" << std::endl; // заголовок для последовательной версии
    std::cout << "  Минимум: " << min_val_seq << std::endl; // вывод минимума
    std::cout << "  Максимум: " << max_val_seq << std::endl; // вывод максимума
    std::cout << "  Время: " << duration_seq.count() << " мкс" << std::endl; // вывод времени
    
    std::cout << "Параллельно (OpenMP):" << std::endl; // заголовок для параллельной версии
    std::cout << "  Минимум: " << min_val_par << std::endl; // вывод минимума
    std::cout << "  Максимум: " << max_val_par << std::endl; // вывод максимума
    std::cout << "  Время: " << duration_par.count() << " мкс" << std::endl; // вывод времени
    
    double speedup = static_cast<double>(duration_seq.count()) / duration_par.count(); // вычисление ускорения
    std::cout << "Ускорение: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl; // вывод ускорения
}

// задание 4: вычисление среднего значения последовательно и параллельно
void task4() {
    std::cout << "\n=== Задание 4 ===" << std::endl; // вывод заголовка задания
    
    const int SIZE = 5000000; // константа размера массива
    std::vector<int> array(SIZE); // создание вектора заданного размера
    
    std::random_device rd; // инициализация генератора случайных чисел
    std::mt19937 gen(rd()); // создание генератора мерсенна
    std::uniform_int_distribution<> dis(1, 100); // создание распределения от 1 до 100
    
    for (int i = 0; i < SIZE; i++) { // цикл заполнения массива
        array[i] = dis(gen); // заполнение элемента случайным числом
    }
    
    // последовательное вычисление среднего
    auto start_seq = std::chrono::high_resolution_clock::now(); // запуск таймера
    
    long long sum_seq = 0; // переменная для накопления суммы
    for (int i = 0; i < SIZE; i++) { // цикл суммирования элементов
        sum_seq += array[i]; // добавление элемента к сумме
    }
    double average_seq = static_cast<double>(sum_seq) / SIZE; // вычисление среднего
    
    auto end_seq = std::chrono::high_resolution_clock::now(); // остановка таймера
    auto duration_seq = std::chrono::duration_cast<std::chrono::microseconds>(end_seq - start_seq); // вычисление времени
    
    // параллельное вычисление среднего с openmp
    auto start_par = std::chrono::high_resolution_clock::now(); // запуск таймера
    
    long long sum_par = 0; // переменная для накопления суммы
    
    #pragma omp parallel for reduction(+:sum_par) // директива openmp для параллельного цикла с редукцией суммы
    for (int i = 0; i < SIZE; i++) { // параллельный цикл по элементам
        sum_par += array[i]; // добавление элемента к локальной сумме
    }
    double average_par = static_cast<double>(sum_par) / SIZE; // вычисление среднего
    
    auto end_par = std::chrono::high_resolution_clock::now(); // остановка таймера
    auto duration_par = std::chrono::duration_cast<std::chrono::microseconds>(end_par - start_par); // вычисление времени
    
    std::cout << "Последовательно:" << std::endl; // заголовок последовательной версии
    std::cout << "  Среднее: " << std::fixed << std::setprecision(2) << average_seq << std::endl; // вывод среднего
    std::cout << "  Время: " << duration_seq.count() << " мкс" << std::endl; // вывод времени
    
    std::cout << "Параллельно (OpenMP с редукцией):" << std::endl; // заголовок параллельной версии
    std::cout << "  Среднее: " << std::fixed << std::setprecision(2) << average_par << std::endl; // вывод среднего
    std::cout << "  Время: " << duration_par.count() << " мкс" << std::endl; // вывод времени
    
    double speedup = static_cast<double>(duration_seq.count()) / duration_par.count(); // вычисление ускорения
    std::cout << "Ускорение: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl; // вывод ускорения
}

// главная функция программы
int main() {
    std::cout << "Программа по гетерогенной параллелизации" << std::endl; // вывод заголовка программы
    std::cout << "Количество доступных потоков: " << omp_get_max_threads() << std::endl; // вывод количества доступных потоков openmp
    
    task1(); // вызов функции первого задания
    task2(); // вызов функции второго задания
    task3(); // вызов функции третьего задания
    task4(); // вызов функции четвертого задания
    
    return 0; // возврат успешного завершения программы
}
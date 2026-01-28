#include <iostream>   // ввод и вывод в консоль
#include <vector>     // контейнер vector
#include <cmath>      // математические функции
#include <omp.h>      // библиотека openmp

// функция вычисления суммы массива параллельно
double parallelSum(const std::vector<double>& data, int numThreads) {
    double sum = 0.0; // переменная для накопления суммы
    
    // задаём количество потоков openmp
    omp_set_num_threads(numThreads);
    
    // параллельный цикл с редукцией суммы
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < data.size(); i++) { // перебор элементов массива
        sum += data[i]; // добавляем элемент к общей сумме
    }
    
    return sum; // возвращаем итоговую сумму
}

// функция вычисления среднего значения
double parallelMean(const std::vector<double>& data, int numThreads) {
    double sum = parallelSum(data, numThreads); // считаем сумму параллельно
    return sum / data.size(); // делим на размер массива
}

// функция вычисления дисперсии параллельно
double parallelVariance(const std::vector<double>& data, double mean, int numThreads) {
    double variance = 0.0; // переменная для накопления дисперсии
    
    // задаём количество потоков
    omp_set_num_threads(numThreads);
    
    // параллельный цикл для суммы квадратов отклонений
    #pragma omp parallel for reduction(+:variance)
    for (size_t i = 0; i < data.size(); i++) { // перебор элементов массива
        double diff = data[i] - mean; // отклонение от среднего
        variance += diff * diff; // добавляем квадрат отклонения
    }
    
    return variance / data.size(); // нормализация дисперсии
}

// последовательная версия для сравнения
void sequentialCompute(const std::vector<double>& data, double& sum, double& mean, double& variance) {
    sum = 0.0; // инициализация суммы
    for (size_t i = 0; i < data.size(); i++) { // последовательный перебор
        sum += data[i]; // добавление элемента
    }
    
    mean = sum / data.size(); // вычисление среднего
    
    variance = 0.0; // инициализация дисперсии
    for (size_t i = 0; i < data.size(); i++) { // повторный проход по массиву
        double diff = data[i] - mean; // отклонение от среднего
        variance += diff * diff; // сумма квадратов отклонений
    }
    variance /= data.size(); // нормализация
}

int main() {
    const size_t arraySize = 100000000; // размер массива
    
    std::cout << "Создание массива из " << arraySize << " элементов...\n"; // сообщение в консоль
    
    std::vector<double> data(arraySize); // выделение памяти под массив
    for (size_t i = 0; i < arraySize; i++) { // инициализация массива
        data[i] = static_cast<double>(i % 1000) / 10.0; // заполнение значениями
    }
    
    std::cout << "Массив создан\n\n"; // подтверждение создания массива
    
    std::cout << "=== ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ ===\n"; // заголовок
    double seqSum, seqMean, seqVariance; // переменные результатов
    
    double startTime = omp_get_wtime(); // старт таймера
    
    sequentialCompute(data, seqSum, seqMean, seqVariance); // последовательные вычисления
    
    double endTime = omp_get_wtime(); // конец таймера
    double seqTime = endTime - startTime; // время выполнения
    
    std::cout << "Время выполнения: " << seqTime << " сек\n"; // вывод времени
    std::cout << "Сумма: " << seqSum << "\n"; // вывод суммы
    std::cout << "Среднее: " << seqMean << "\n"; // вывод среднего
    std::cout << "Дисперсия: " << seqVariance << "\n\n"; // вывод дисперсии
    
    std::cout << "=== ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ (OpenMP) ===\n"; // заголовок
    std::cout << "Потоки | Время (сек) | Ускорение | Эффективность | Доля парал.\n"; // таблица
    std::cout << "-------|-------------|-----------|---------------|-------------\n"; // разделитель
    
    int threadCounts[] = {1, 2, 4, 8, 16}; // набор потоков для теста
    
    for (int numThreads : threadCounts) { // перебор числа потоков
        startTime = omp_get_wtime(); // старт таймера
        
        double parSum = parallelSum(data, numThreads); // параллельная сумма
        double parMean = parallelMean(data, numThreads); // параллельное среднее
        double parVariance = parallelVariance(data, parMean, numThreads); // параллельная дисперсия
        
        endTime = omp_get_wtime(); // конец таймера
        double parTime = endTime - startTime; // время выполнения
        
        double speedup = seqTime / parTime; // ускорение
        double efficiency = speedup / numThreads; // эффективность
        
        double parallelFraction = 0.0; // доля параллельной части
        if (numThreads > 1 && speedup > 1.0) { // проверка корректности
            parallelFraction = (numThreads * (speedup - 1.0)) /
                               (speedup * (numThreads - 1.0)); // формула амдала
        }
        
        std::cout << numThreads << "      | "
                  << parTime << " | "
                  << speedup << "x   | "
                  << (efficiency * 100) << "%      | "
                  << (parallelFraction * 100) << "%\n"; // вывод строки таблицы
    }
    
    std::cout << "\n=== АНАЛИЗ ПО ЗАКОНУ АМДАЛА ===\n"; // заголовок
    std::cout << "Закон Амдала: S = 1 / ((1-p) + p/n)\n"; // формула
    std::cout << "где S - ускорение, p - доля параллельной части, n - число потоков\n\n"; // пояснение
    
    std::cout << "Доля парал. части | Макс. ускорение (теор.)\n"; // заголовок таблицы
    std::cout << "------------------|------------------------\n"; // разделитель
    for (double p : {0.5, 0.75, 0.9, 0.95, 0.99}) { // перебор долей
        double maxSpeedup = 1.0 / (1.0 - p); // теоретический предел
        std::cout << (p * 100) << "%           | " << maxSpeedup << "x\n"; // вывод строки
    }
    
    std::cout << "\n=== ПРОФИЛИРОВАНИЕ ===\n"; // заголовок
    std::cout << "Последовательная часть программы:\n"; // описание
    std::cout << "  - Инициализация массива\n"; // пункт
    std::cout << "  - Деление на размер массива (в mean и variance)\n"; // пункт
    std::cout << "  - Вывод результатов\n\n"; // пункт
    
    std::cout << "Параллельная часть программы:\n"; // описание
    std::cout << "  - Циклы вычисления суммы\n"; // пункт
    std::cout << "  - Циклы вычисления дисперсии\n\n"; // пункт
    
    std::cout << "ВЫВОДЫ:\n"; // заголовок
    std::cout << "1. При увеличении потоков ускорение растёт, но не линейно\n"; // вывод
    std::cout << "2. Эффективность использования потоков падает с их ростом\n"; // вывод
    std::cout << "3. Последовательные части ограничивают максимальное ускорение\n"; // вывод
    std::cout << "4. Накладные расходы снижают производительность\n"; // вывод
    
    return 0; // завершение программы
}

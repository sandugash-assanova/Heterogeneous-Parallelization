#include <iostream>     // ввод и вывод в консоль
#include <mpi.h>        // библиотека mpi
#include <vector>       // контейнер vector
#include <cmath>        // математические функции
#include <iomanip>      // форматированный вывод

// функция вычисления суммы локальной части массива
double computeLocalSum(const std::vector<double>& localData) {
    double sum = 0.0;                      // инициализация суммы
    for (size_t i = 0; i < localData.size(); i++) { // цикл по локальному массиву
        sum += localData[i];               // добавляем элемент к сумме
    }
    return sum;                            // возвращаем локальную сумму
}

// функция поиска минимума в локальной части
double computeLocalMin(const std::vector<double>& localData) {
    if (localData.empty()) return 0.0;     // проверка на пустой массив
    
    double minVal = localData[0];          // начальное минимальное значение
    for (size_t i = 1; i < localData.size(); i++) { // проход по массиву
        if (localData[i] < minVal) {       // сравнение текущего элемента
            minVal = localData[i];         // обновление минимума
        }
    }
    return minVal;                         // возвращаем минимум
}

// функция поиска максимума в локальной части
double computeLocalMax(const std::vector<double>& localData) {
    if (localData.empty()) return 0.0;     // проверка на пустой массив
    
    double maxVal = localData[0];          // начальное максимальное значение
    for (size_t i = 1; i < localData.size(); i++) { // проход по массиву
        if (localData[i] > maxVal) {       // сравнение текущего элемента
            maxVal = localData[i];         // обновление максимума
        }
    }
    return maxVal;                         // возвращаем максимум
}

int main(int argc, char** argv) {
    // инициализация mpi
    MPI_Init(&argc, &argv);
    
    // переменная для общего числа процессов
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize); // получаем количество процессов
    
    // переменная для ранга процесса
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // получаем ранг текущего процесса
    
    // фиксированный размер задачи для strong scaling
    const long long strongScalingSize = 100000000; // 100 млн элементов
    
    // размер данных на процесс для weak scaling
    const long long sizePerProcess = 10000000; // 10 млн элементов
    long long weakScalingSize = sizePerProcess * worldSize; // общий размер задачи
    
    // вывод информации только на root процессе
    if (rank == 0) {
        std::cout << "\n=== АНАЛИЗ МАСШТАБИРУЕМОСТИ MPI ===\n";
        std::cout << "Количество процессов: " << worldSize << "\n\n";
    }
    
    // -------- strong scaling --------
    if (rank == 0) {
        std::cout << "=== STRONG SCALING TEST ===\n";
        std::cout << "Фиксированный размер задачи: " << strongScalingSize << " элементов\n";
        std::cout << "Размер на процесс: " << strongScalingSize / worldSize << " элементов\n\n";
    }
    
    // размер локального массива
    long long strongLocalSize = strongScalingSize / worldSize;
    
    // последний процесс обрабатывает остаток
    if (rank == worldSize - 1) {
        strongLocalSize += strongScalingSize % worldSize;
    }
    
    // локальный массив данных
    std::vector<double> strongLocalData(strongLocalSize);
    
    // глобальное смещение для корректной инициализации
    long long globalOffset = rank * (strongScalingSize / worldSize);
    
    // заполнение локального массива
    for (long long i = 0; i < strongLocalSize; i++) {
        strongLocalData[i] = static_cast<double>((globalOffset + i) % 10000) / 100.0;
    }
    
    // синхронизация всех процессов
    MPI_Barrier(MPI_COMM_WORLD);
    
    // старт таймера
    double strongStartTime = MPI_Wtime();
    
    // локальные вычисления
    double localSum = computeLocalSum(strongLocalData); // сумма
    double localMin = computeLocalMin(strongLocalData); // минимум
    double localMax = computeLocalMax(strongLocalData); // максимум
    
    // время локальных вычислений
    double localComputeTime = MPI_Wtime() - strongStartTime;
    
    // старт измерения коммуникаций
    double commStartTime = MPI_Wtime();
    
    // глобальная сумма через reduce
    double globalSum = 0.0;
    MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // глобальный минимум
    double globalMin = 0.0;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    // глобальный максимум
    double globalMax = 0.0;
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // время reduce
    double reduceCommTime = MPI_Wtime() - commStartTime;
    
    // allreduce тест
    commStartTime = MPI_Wtime();
    double allreduceSum = 0.0;
    MPI_Allreduce(&localSum, &allreduceSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double allreduceCommTime = MPI_Wtime() - commStartTime;
    
    // синхронизация перед финальным замером
    MPI_Barrier(MPI_COMM_WORLD);
    
    // общее время выполнения
    double strongTotalTime = MPI_Wtime() - strongStartTime;
    
    // вывод результатов на root
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Результаты вычислений:\n";
        std::cout << "  Сумма: " << globalSum << "\n";
        std::cout << "  Минимум: " << globalMin << "\n";
        std::cout << "  Максимум: " << globalMax << "\n\n";
    }
    
    // завершение mpi
    MPI_Finalize();
    
    return 0; // корректное завершение программы
}

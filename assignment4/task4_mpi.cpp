#include <iostream>
#include <mpi.h>
#include <chrono>
#include <cmath>

// функция обработки части массива
void processArray(float* input, float* output, int size) {
    // простая обработка - возведение в квадрат
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * input[i];
    }
}

int main(int argc, char** argv) {
    // инициализируем mpi окружение
    MPI_Init(&argc, &argv);
    
    // получаем общее количество процессов
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    
    // получаем ранг (номер) текущего процесса
    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    
    // общий размер массива
    const int totalSize = 10000000;
    
    // вычисляем размер части для каждого процесса
    int localSize = totalSize / worldSize;
    // последний процесс обрабатывает остаток если деление неровное
    if (worldRank == worldSize - 1) {
        localSize = totalSize - localSize * (worldSize - 1);
    }
    
    // выделяем память для локальной части
    float* localInput = new float[localSize];
    float* localOutput = new float[localSize];
    
    // массивы для главного процесса
    float* globalInput = nullptr;
    float* globalOutput = nullptr;
    
    // главный процесс (ранг 0) создаёт и инициализирует полный массив
    if (worldRank == 0) {
        globalInput = new float[totalSize];
        globalOutput = new float[totalSize];
        
        // инициализируем массив
        for (int i = 0; i < totalSize; i++) {
            globalInput[i] = (float)i / 100.0f;
        }
        
        std::cout << "Размер массива: " << totalSize << " элементов\n";
        std::cout << "Количество процессов: " << worldSize << "\n\n";
    }
    
    // массивы для описания размеров и смещений каждого процесса
    int* sendCounts = nullptr;
    int* displacements = nullptr;
    
    if (worldRank == 0) {
        sendCounts = new int[worldSize];
        displacements = new int[worldSize];
        
        // вычисляем размеры и смещения для каждого процесса
        for (int i = 0; i < worldSize; i++) {
            sendCounts[i] = totalSize / worldSize;
            displacements[i] = i * sendCounts[i];
        }
        // корректируем размер для последнего процесса
        sendCounts[worldSize - 1] = totalSize - displacements[worldSize - 1];
    }
    
    // барьер синхронизации перед началом замера
    MPI_Barrier(MPI_COMM_WORLD);
    
    // начинаем замер времени
    double startTime = MPI_Wtime();
    
    // рассылаем части массива всем процессам
    // используем MPI_Scatterv так как размеры могут различаться
    MPI_Scatterv(
        globalInput,      // буфер отправки (только у root)
        sendCounts,       // массив размеров для каждого процесса
        displacements,    // массив смещений
        MPI_FLOAT,        // тип данных
        localInput,       // буфер приёма
        localSize,        // размер локальной части
        MPI_FLOAT,        // тип данных
        0,                // ранг root процесса
        MPI_COMM_WORLD    // коммуникатор
    );
    
    // каждый процесс обрабатывает свою часть массива
    processArray(localInput, localOutput, localSize);
    
    // собираем результаты обратно на главном процессе
    MPI_Gatherv(
        localOutput,      // буфер отправки
        localSize,        // размер локальных данных
        MPI_FLOAT,        // тип данных
        globalOutput,     // буфер приёма (только у root)
        sendCounts,       // массив размеров
        displacements,    // массив смещений
        MPI_FLOAT,        // тип данных
        0,                // ранг root процесса
        MPI_COMM_WORLD    // коммуникатор
    );
    
    // барьер перед окончанием замера
    MPI_Barrier(MPI_COMM_WORLD);
    
    // завершаем замер времени
    double endTime = MPI_Wtime();
    double elapsedTime = endTime - startTime;
    
    // главный процесс выводит результаты
    if (worldRank == 0) {
        std::cout << "Время выполнения: " << elapsedTime * 1000000 << " мкс\n";
        std::cout << "Время выполнения: " << elapsedTime * 1000 << " мс\n\n";
        
        // проверка корректности (первые несколько элементов)
        std::cout << "Проверка (первые 5 элементов):\n";
        for (int i = 0; i < 5; i++) {
            float expected = globalInput[i] * globalInput[i];
            std::cout << "Input[" << i << "] = " << globalInput[i] 
                      << ", Output[" << i << "] = " << globalOutput[i]
                      << ", Expected = " << expected << "\n";
        }
        
        // освобождаем память глобальных массивов
        delete[] globalInput;
        delete[] globalOutput;
        delete[] sendCounts;
        delete[] displacements;
    }
    
    // освобождаем локальную память
    delete[] localInput;
    delete[] localOutput;
    
    // завершаем mpi окружение
    MPI_Finalize();
    
    return 0;
}

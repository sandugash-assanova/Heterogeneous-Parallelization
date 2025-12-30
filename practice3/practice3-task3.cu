#include <iostream> // ввод и вывод
#include <vector> // контейнер vector
#include <algorithm> // стандартная сортировка
#include <chrono> // измерение времени

void cpuSort(std::vector<int> &arr) { // сортировка на cpu
    std::sort(arr.begin(), arr.end()); // используем стандартную сортировку
}

int main() { // точка входа
    const int size = 100000; // размер массива
    std::vector<int> data(size); // создаем массив

    for (int i = 0; i < size; i++) // заполняем массив
        data[i] = rand() % 100000; // случайные числа

    auto cpuData = data; // копия для cpu

    auto start = std::chrono::high_resolution_clock::now(); // старт времени
    cpuSort(cpuData); // сортируем на cpu
    auto end = std::chrono::high_resolution_clock::now(); // конец времени

    std::chrono::duration<double> cpuTime = end - start; // считаем время
    std::cout << "cpu time: " << cpuTime.count() << " seconds" << std::endl; // выводим результат

    return 0; // завершение программы
}

#ifndef POPULATION_H
#define POPULATION_H

#include <iostream>
#include <algorithm>
#include <curand_kernel.h>
#include "cities.h"
#include "city.h"

class population
{
private:
    double total_fit;

    //static bool chromosome_compare(const chromosome* a, const chromosome* b) { a->fit_probability > b->fit_probability; }
public:
    chromosome* parents;
    int population_size;

    __host__ __device__ population() {}
    __host__ __device__ population(chromosome* parents, int population_size) : total_fit(0.0), parents(parents), 
        population_size(population_size) {}

    __host__ __device__ double get_tlfit() { return total_fit; }

    /**
     * @brief       Функция population::popgener(...) генерирует начальную популяцию для хромососомы по 
     *              index.
     * 
     * @param[in]   genes список генов для задания хромососы;
     * @param[in]   first_gen из условия задачи "tsp" для хромосомы должен быть задан первый ген;
     * @param[in]   rand_state указатель на псевдослучайную последовательность на __device__;
     * @param[in]   gens_count размер хромосомы (т.е. количество генов в ней).
     * @param[in]   index индекс текущего потока (текущей хромосомы). 
     */
    __device__ void popgener(const gen* genes, const gen first_gen, curandState* local_rand_state, int index)
    {
        parents[index].gengener(genes, first_gen, local_rand_state);
    }

    /**
     * @brief       Функция chromosome_sort(...) сортирует хрмосомы по их приспособленности.
     * 
     * @param[in]   start -- начало сортируемого массива;
     * @param[in]   end -- конец сортируемого массива.
     */
    //void chromosome_sort(size_t start, size_t end)
    //{
    //    bool(*comparator)(const chromosome* a, const chromosome* b) = chromosome_compare;
    //    std::sort(parents + start, parents + end, comparator);
    //}

    #define NEWPOPSIZE int(new_pop_size(probability))
    /**
     * @brief       Функция population::new_pop_size рассчитывает размер новой популяции.
     * 
     * @param[in]   probability -- вероятность попадания решения в новую популяцию.
     * @return      int -- число размера новой популяции.
     */
    int new_pop_size(double probability)
    {
        int new_size = 0;
        for (int chrom = 0; chrom < population_size || parents[chrom].fit_probability < probability; ++chrom) { new_size += 1;}
        return new_size;
    }

    /**
     * @brief       Функция popilation::roul_selection(...) обновлят текущую популяцию методом "селекция
     *              на основе рулетки": каждая хромосома имеет некоторую вероятность отбора в новую популяцию.
     *              Чем больше значение fitness, тем большую вероятность отбора решения в новую популяцию. Данные 
     *              метод позволяет включить в новую популяцию не самые лучшие решения, но в перспективе, благодаря
     *              мутации, может быть достигнуто наилучшее решение.
     *               
     * @param[in]   local_rand_state -- указатель на псевдослучайную последовательность на __device__.
     * 
     * @details     Функция population::roul_selection(...) основна на формуле \f$\frac{f_i}{\sum_{j=1}^{N}fj}\f$,
     *              где N -- размер популяции, fi -- значение приспособленности конкретной популяции. Таким образом,
     *              формула представляет собой отношений конкретного значения fitness к общему значению fitness всей
     *              популяции.
     * 
     *              Сначала population::roul_selection(...) вычисляет значение total_fit как общее значение fitness 
     *              всй популяции. Далее для каждой хромосомы вычисляется вероятность попадания в новую популяцию по 
     *              формуле выше. Наконец определяется случайное число probability, которое определяет состав обновленной 
     *              популяции на основе отсортированной старой популяции по значению fit_probability. Если все прошло успешно, 
     *              то текущая популяция и ее размер будут обновлены.
     */
    void roul_selection()
    {
        for (int chrom = 0; chrom < population_size; ++chrom) { total_fit += parents[chrom].get_fit(); }
        for (int chrom = 0; chrom < population_size; ++chrom) { parents[chrom].fit_probability = parents[chrom].get_fit()/total_fit; }

        //chromosome_sort(0, population_size);

        double probability = std::rand() / RAND_MAX + 1.0;
        int new_population_size = NEWPOPSIZE;
        chromosome* new_population;
        cudaMalloc((void**)&new_population, new_population_size*sizeof(chromosome));   

        if (probability < 0.1) {
            for (int chrom = 0; chrom < population_size || parents[chrom].fit_probability < 0.1; ++chrom) { 
                new_population[chrom] = parents[chrom];
            }
        }
        else if (probability < 0.3) {
            for (int chrom = 0; chrom < population_size || parents[chrom].fit_probability < 0.3; ++chrom) { 
                new_population[chrom] = parents[chrom];
            }
        }
        else if (probability < 0.6) {
            for (int chrom = 0; chrom < population_size || parents[chrom].fit_probability < 0.6; ++chrom) { 
                new_population[chrom] = parents[chrom];
            }
        }
        else if (probability < 1.0) {
            for (int chrom = 0; chrom < population_size || parents[chrom].fit_probability < 1.0; ++chrom) { 
                new_population[chrom] = parents[chrom];
            }            
        }

        parents = new_population;
        population_size = new_population_size;
        cudaFree(new_population);
    }
};

#endif
#ifndef POPULATION_H
#define POPULATION_H

#include <iostream>
#include <algorithm>
#include <curand_kernel.h>
#include "cities.h"
#include "city.h"

class population
{
public:
    chromosome* parents;
    chromosome* new_parents_state;
    int population_size;
    int new_population_size;

    __host__ __device__ population() {}
    __host__ __device__ population(chromosome* parents, int population_size) : parents(parents), new_parents_state(nullptr), 
        population_size(population_size), new_population_size(0) { }

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
     * @brief       Функция set_newpop_size(...) определяет размер нового этапа для популяции, а также устанавливает 
     *              значение вероятность отбора для каждой хромосоы популяции на основе метода "рулетки": для каждой 
     *              хромосомы орпеделяется некоторая вероятность отбора (sel_probability) на основе значения fitness.  
     *              Чем больше значение fitness, тем большую вероятность отбора решения  в новую популяцию. Данный 
     *              метод позволяет включить в новую популяцию не самые лучшие решения, но в перспективе, благодаря 
     *              мутации, может быть достигнуто наилучшее решение.
     * 
     * @param[in]   probability -- вероятность попадания хромосомы в новую популяцию.
     *
     * @details     Функция population::set_newpop_size(...) основна на формуле \f$\frac{f_i}{\sum_{j=1}^{N}fj}\f$,
     *              где N -- размер популяции, fi -- значение приспособленности конкретной популяции. Таким образом,
     *              формула представляет собой отношений конкретного значения fitness к общему значению fitness всей
     *              популяции.
     * 
     *              Сначала population::set_newpop_size(...) вычисляет значение total_fit как общее значение fitness 
     *              всй популяции. Далее для каждой хромосомы вычисляется вероятность попадания в новую популяцию по 
     *              формуле выше. Наконец определяется случайное число probability, которое определяет состав обновленной 
     *              популяции на основе отсортированной старой популяции по значению sel_probability. Если все прошло успешно, 
     *              то будет определн размер новой популяции, а для каждой хромосомы будет рассчитана вероятность отбора в 
     *              новую популяцию.
     */
    __host__ int set_newpop_size(double probability)
    {
        int total_fit = 0, new_size = 0;
        for (int chrom = 0; chrom < population_size; ++chrom) { total_fit += parents[chrom].get_fit(); }
        for (int chrom = 0; chrom < population_size; ++chrom) { parents[chrom].sel_probability = parents[chrom].get_fit() / total_fit; }
        for (int chrom = 0; chrom < population_size; ++chrom) {
            if (parents[chrom].sel_probability <= probability) { new_size += 1; }
        }
        new_population_size = new_size;
        cudaDeviceSynchronize();
        return new_population_size;
    };

    /**
     * @brief       Функция popilation::roul_selection(...) обновлят текущую путем отбора в новую популяцию только тех
     *              хромосом, вероятность отбора в новую популяцию которых (sel_probability) меньше или равны заданной 
     *              вероятности.
     *               
     * @param[in]   probability -- вероятность попадания хромосомы в новую популяцию.
     */
    __host__ void roul_selection(double probability)
    {
        if (new_population_size == 0 || new_parents_state == nullptr) { return; }
        for (int chrom = 0, i = 0; chrom < population_size && i < new_population_size; ++chrom) {
            if (parents[chrom].sel_probability <= probability) { new_parents_state[i++] = parents[chrom]; }
        }

        cudaFree(parents);
        parents = new_parents_state;
        population_size = new_population_size;
        new_population_size = 0;
    }
};

#endif
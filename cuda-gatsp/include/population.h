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
     * @param[in]   genes -- список генов для задания хромососы;
     * @param[in]   first_gen -- из условия задачи "tsp" для хромосомы должен быть задан первый ген;
     * @param[in]   rand_state -- указатель на псевдослучайную последовательность на __device__;
     * @param[in]   gens_count -- размер хромосомы (т.е. количество генов в ней).
     * @param[in]   curr_index -- индекс текущего потока (текущей хромосомы). 
     */
    __device__ void popgener(const gen* genes, const gen first_gen, curandState* local_rand_state, int curr_index)
    {
        parents[curr_index].gengener(genes, first_gen, local_rand_state);
    }

    /**
     * @brief       Функция popcross(...) обновляет текущее состояние популяции путем генерации потомков от текущих хромосом.
     *              Решение-потомок строится на основе хромосомы родителя и хромосомы потомка, которые определяются случайно, после чего
     *              вызывается функция cudaCrossover, которая разрезает хромосомы в двух точках, переворачивает и меняет местами (двухточечный
     *              оператор кроссинговера). В результате свойства population::parents будет обновлено.
     * 
     * @param[in]   local_rand_state -- указатель на псевдослучайную последовательность;
     * @param[in]   curr_index -- индекс текущего потока.
     */
    __device__ void popcross(curandState* local_rand_state, int curr_index)
    {
        chromosome parent1 = parents[(int)(curand_uniform(local_rand_state) * (population_size - 1))];
        chromosome parent2 = parents[(int)(curand_uniform(local_rand_state) * (population_size - 1))];
        new_parents_state[curr_index] = crossover(parent1, parent2, local_rand_state);
    }

    /**
     * @brief       Функция popmutation(...) вызывает оператор мутации у i-ой хромосомы i-го потока.
     * 
     * @param[in]   local_rand_state -- указатель на псевдослучайную последовательность;
     * @param[in]   chrom_index -- индекс i-ой хромосомы для i-го потока. 
     */
    __device__ void popmutation(curandState* local_rand_state, int chrom_index)
    {
        parents[chrom_index].mutation(local_rand_state);
    }

    /**
     * @brief       Функция set_roul_popsize(...) определяет размер нового этапа для популяции, а также устанавливает 
     *              значение вероятность отбора для каждой хромосоы популяции на основе метода "рулетки": для каждой 
     *              хромосомы орпеделяется некоторая вероятность отбора (sel_probability) на основе значения fitness.  
     *              Чем больше значение fitness, тем большую вероятность отбора решения  в новую популяцию. Данный 
     *              метод позволяет включить в новую популяцию не самые лучшие решения, но в перспективе, благодаря 
     *              мутации, может быть достигнуто наилучшее решение.
     * 
     * @param[in]   probability -- вероятность попадания хромосомы в новую популяцию.
     *
     * @details     Функция population::set_roul_popsize(...) основна на формуле \f$\frac{f_i}{\sum_{j=1}^{N}fj}\f$,
     *              где N -- размер популяции, fi -- значение приспособленности конкретной популяции. Таким образом,
     *              формула представляет собой отношений конкретного значения fitness к общему значению fitness всей
     *              популяции.
     * 
     *              Сначала population::set_roul_popsize(...) вычисляет значение total_fit как общее значение fitness 
     *              всй популяции. Далее для каждой хромосомы вычисляется вероятность попадания в новую популяцию по 
     *              формуле выше. Наконец определяется случайное число probability, которое определяет состав обновленной 
     *              популяции на основе отсортированной старой популяции по значению sel_probability. Если все прошло успешно, 
     *              то будет определн размер новой популяции, а для каждой хромосомы будет рассчитана вероятность отбора в 
     *              новую популяцию.
     */
    __host__ int set_roul_popsize(double probability)
    {
        double total_fit = 0, new_size = 0;
        for (int chrom = 0; chrom < population_size; ++chrom) { total_fit += parents[chrom].get_fit(); }
        for (int chrom = 0; chrom < population_size; ++chrom) { parents[chrom].sel_probability = parents[chrom].get_fit() / total_fit; }
        for (int chrom = 0; chrom < population_size; ++chrom) { if (probability < parents[chrom].sel_probability) { new_size += 1; } }
        new_population_size = new_size;
        cudaDeviceSynchronize();
        return new_population_size;
    };

    /**
     * @brief       Функция popilation::roul_selection(...) обновлят текущую популяцию путем отбора в новую популяцию 
     *              только тех хромосом, вероятность отбора в новую популяцию которых (sel_probability) меньше или равны 
     *              заданной вероятности.
     *               
     * @param[in]   probability -- вероятность попадания хромосомы в новую популяцию.
     */
    __host__ void roul_selection(double probability)
    {
        // Проверка на создание нового состояния требуемого размера.
        if (new_population_size == 0 || new_parents_state == nullptr) { return; }
        for (int chrom = 0, i = 0; chrom < population_size && i < new_population_size; ++chrom) {
            if (probability < parents[chrom].sel_probability) { new_parents_state[i++] = parents[chrom]; }
        }

        // Очизаем прошлое состояние и устанаваливаем новое. 
        cudaFree(parents);
        parents = new_parents_state;
        population_size = new_population_size;
        new_population_size = 0;
        cudaFree(new_parents_state);
    }
};

#endif
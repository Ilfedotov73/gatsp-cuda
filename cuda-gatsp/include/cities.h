#ifndef CITIES_H
#define CITIES_H

#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <cmath>

#include "city.h"

class cities
{
private:
    double fit;
public:
    city* citieslist;
    int cities_count;
    double sel_probability;

    __host__ __device__ cities() {}
    __host__ __device__ cities(city* citieslist, int cities_size) : fit(-1.0), citieslist(citieslist), cities_count(cities_size), 
        sel_probability(0.0) {}

    __host__ __device__ double get_fit() const { return fit; }
    
    /**
     * @brief       Функция cities::gengener(...) задает случайные гены для хромосомы размера --
     *              cities_size методом "дробавик". Метод генерации начальной популяции "дробавик" 
     *              использует случайный выбор альтернатив из всей области решения задачи (всех 
     *              возможных генов).
     * 
     * @param[in]   genes список генов для задания хромососы;
     * @param[in]   first_gen из условия задачи "tsp" для хромосомы должен быть задан первый ген;
     * @param[in]   local_rand_state указатель на псевдослучайную последовательность на __device__.
     * 
     * @details     Функция cities::cudaGenerated(...) записывает гены в хромососу в случайном параядке:
     *              во-первых, из условия задачи tsp, необходимо указать точку старта, т.е. записать в 
     *              citieslist[0] указатель на ген -- начальный город через параметр first_gen, во-вторых,
     *              перед записьсю случайного (не первого) гена необходимо убедиться, что такой ген не  
     *              отсутствовал ранее: для этого реализован перебор всех ранее записанных в хромосому генов. 
     */
    __device__ gen* gengener(const gen* genes, const gen first_gen, curandState* local_rand_state) 
    {
        citieslist[0] = first_gen;
        for (int i = 1; i < cities_count;) {
            int rand_idx = (int)(curand_uniform(local_rand_state) * cities_count);
            for (int j = 0; j < i; ++j) {
                if (citieslist[j].id == genes[rand_idx].id) { goto continue2;}
            }
            citieslist[i++] = genes[rand_idx];
            continue2:;
        }
    }

    /**
     * @brief       Функция cities::mutatiоn(...) реализует двухточечную мутацию путем перестановки двух
     *              случайных генов местами.
     *
     * @param[in]   local_rand_state -- указатель на псевдослучайную последовательность на __device__.
     */
    __device__ void mutation(curandState* local_rand_state)
    {
        int rand_idx1 = (int)(curand_uniform(local_rand_state) * cities_count);
        int rand_idx2 = (int)(curand_uniform(local_rand_state) * cities_count);

        for (; rand_idx1 == rand_idx2;) { rand_idx2 = (int)(curand_uniform(local_rand_state) * (cities_count - 1)); }

        gen tempptr = citieslist[rand_idx1];
        citieslist[rand_idx1] = citieslist[rand_idx2];
        citieslist[rand_idx2] = tempptr;
    }

    /**
     * @brief       Функция fitness() устанавалиает показатель приспособленности хромосы в свойство cities::fit.
     */
    __device__ void fitness()
    {
        double value = 0;
        for (int i = 1; i < cities_count; ++i) {
            value += distance(citieslist[i-1].get_point(), citieslist[i].get_point());
        }
        fit = value;
    }
};
using chromosome = cities;

/**
 * @brief       Функция crossover(...) реализует оператор скрещивания, путем унаследования хромосом потомком 
 *              от одного конретного родителя. Выбор родиделя для наследования хромосом основан на его значении
 *              приспособленности, чем больше доля этого значения к общей приспособленности родителей, тем больше 
 *              вероятность того, что хромосома-потомок унаследует гены от наиболее приспособленного к окружающей 
 *              среде родителя.
 *              
 * 
 * @param[in]   parent1 -- хромосома первого родителя;
 * @param[in]   parent2 -- хромосома второго родителя;
 * @param[in]   local_rand_state -- указатель на псевдослучайную последовательность на __device__;
 * @param[in]   chromosome_size -- размер хромосомы.
 * 
 * @return      chromosome -- хромосома потомка.
 */
__device__ inline chromosome crossover(const chromosome& parent1, const chromosome& parent2, curandState* local_rand_state) 
{
    double total_fit = parent1.get_fit() + parent2.get_fit();
    if (curand_uniform_double(local_rand_state) < parent1.get_fit() / total_fit) { return parent1; }
    else { return parent2; }
} 

#endif
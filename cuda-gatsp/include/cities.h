#ifndef CITIES_H
#define CITIES_H

#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <device_functions.h>

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
    __host__ __device__ cities(city* citieslist, int cities_count) : fit(-1.0), citieslist(citieslist), cities_count(cities_count), 
        sel_probability(0.0) {}

    __host__ __device__ double get_fit() const { return fit; }
    
    /**
     * @brief       Функция cities::gengener(...) задает случайные гены для хромосомы размера 
     *              cities_count методом "дробовик". Метод генерации начальной популяции "дробовик" 
     *              использует случайный выбор альтернатив из всей области решения задачи (всех 
     *              возможных генов).
     * 
     * @param[in]   genes список генов для задания хромососы;
     * @param[in]   first_gen из условия задачи "tsp" для хромосомы должен быть задан первый ген;
     * @param[in]   local_rand_state указатель на псевдослучайную последовательность на __device__.
     * 
     * @details     Функция cities::gengener(...) записывает гены в хромососу в случайном параядке:
     *              во-первых, из условия задачи tsp, необходимо указать точку старта, т.е. записать в 
     *              citieslist[0] указатель на ген -- начальный город через параметр first_gen, во-вторых,
     *              перед записьсю случайного (не первого) гена необходимо убедиться, что такой ген не  
     *              отсутствовал ранее: для этого реализован перебор всех ранее записанных в хромосому генов. 
     */
    __device__ void gengener(const gen* genes, const gen first_gen, curandState* local_rand_state) 
    {
        citieslist[0] = first_gen;
        for (int i = 1; i < cities_count;) {
            int rand_idx = min((int)(curand_uniform(local_rand_state) * cities_count), cities_count - 1);
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
        int rand_idx1 = min((int)(curand_uniform(local_rand_state) * cities_count), cities_count - 1);
        int rand_idx2 = min((int)(curand_uniform(local_rand_state) * cities_count), cities_count - 1);

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
 * @param[in]   parents_src -- исходная-родительская популяция;
 * @param[in]   parent_idx1 -- индекс хромосомы первого родителя;
 * @param[in]   parent_idx2 -- индекс хромосомы второго родителя;
 * @param[in]   local_rand_state -- указатель на псевдослучайную последовательность на __device__;
 * 
 * @return      chromosome -- хромосома потомка.
 */
__device__ inline chromosome crossover(chromosome* parents_src, int parent_idx1, int parent_idx2, curandState* local_rand_state) 
{
    double total_fit = parents_src[parent_idx1].get_fit() + parents_src[parent_idx2].get_fit();
    if (curand_uniform_double(local_rand_state) < parents_src[parent_idx1].get_fit() / total_fit) { return parents_src[parent_idx1]; }
    else { return parents_src[parent_idx2]; }
} 

#endif
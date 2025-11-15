#include <curand_kernel.h>
#include <iostream>
#include <time.h>

#include "../../cuda-tspui/include/vec2.h"
#include "../include/city.h"
#include "../include/cities.h"
#include "../include/population.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef unsigned int uint32;

#define checkCudaErrors(val) checkCuda(val, #val, __FILE__, __LINE__)
void checkCuda(cudaError_t result, const char* const func, const char* const file, int const line)
{
    if (result) {
        std::cerr << "CUDA error: " << static_cast<uint32>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

/**
 * @brief       Функция result_output(...) выводит результирующую популяцию в выходной поток для отладки вывода.
 *              Структура вывода:
 *              chromosome_i_fit || chromosome_i_sel_probability: 
 *                  [chromosome_i_gen_0, chromosome_i_gen_1, ... , chromosome_i_gen_j]
 *
 * @param[in]   d_pop результирующая популяция для вывода.
 */
void result_output(population* d_pop)
{
    std::cerr << "Result:\n";
    for (int i = 0; i < d_pop->population_size; ++i) {
        std::cerr << "* "; std::cout << d_pop->parents[i].get_fit() << " || " << d_pop->parents[i].sel_probability << '\n';
        for (int j = 0; j < d_pop->parents[i].cities_count; ++j) {
            std::cerr << "** "; std::cout << d_pop->parents[i].citieslist[j].id << ": "
                << d_pop->parents[i].citieslist[j].get_point().x() << d_pop->parents[i].citieslist[j].get_point().y() << '\n';
        }
    }
    std::clog << "\rDone.\n";
}

#define RND (curand_uniform_double(rand_state))
/**
 * @brief       Функция genes_gener(...) генерирует случайную область решения.
 *
 * @param[in]   d_gens_val указатель на массив генов, которые необходимо инициализировать;
 * @param[in]   rand_state указатель на псевдослучайную последовательность;
 * @param[in]   gens_count количество генов в хромосоме.
 */
__global__ void rnd_genes_gener(gen* d_gens_val, curandState* rand_state, int gens_count)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
        for (int i = 0; i < gens_count; ++i) { d_gens_val[i] = gen(vec2(RND, RND), i); }
    }
}

/**
 * @brief       Функция cudaPopgener(...) инициализирует генерацию хромосом в начальной
 *              популяции. Каждая хромосома инициализируется в отдельном t.x + b.x потоке.
 *
 * @param[in]   d_pop указатель на начальную популяцию;
 * @param[in]   d_gens_val указатель на область решений;
 * @param[in]   rand_state указатель на псевдослучайную последовательность.
 */
__global__ void cudaPopgener(population* d_pop, gen* d_gens_val, curandState* rand_state)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= d_pop->population_size) { return; }
    curand_init(1984, index, 0, &rand_state[index]);
    curandState local_rand_state = rand_state[index];
    gen first_gen = d_gens_val[0];
    d_pop->popgener(d_gens_val, first_gen, &local_rand_state, index);
}

/**
 * @brief       Функция cudaFitness(...) запускается t*b потоках. В каждом потоке определятся индекс
 *              соответствующей хромосомы, у которой, вызывается fitness().
 *
 * @param[in]   d_pop популяция, для хромосом которой требуется вычислить значение приспособленности.
 */
__global__ void cudaFitness(population* d_pop)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= d_pop->population_size) { return; }
    d_pop->parents[index].fitness();
}

__global__ void cudaCrossover(population* d_pop, curandState* rand_state)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= d_pop->new_population_size) { return; }
    curand_init(1984, index, 0, &rand_state[index]);
    curandState local_rand_state = rand_state[index];
    d_pop->popcross(&local_rand_state, index);
}

int main()
{
    /**
     * @param   GENS_COUNT размер доступной области решения задачи (количество городов);
     * @param   POPULATION_SIZE размер популяции (кол-во решений или хромосом в популяции);
     * @param   MAX_ITERACTION_LIMIT лимит иттераций алгоритма gatsp.
     */
    int GENS_COUNT = 5,
        POPULATION_SIZE = 4,
        MAX_ITERACTION_LIMIT = 1;

    /**
     * @param MUTATION_PROBABILITY заданная вероятность мутации хромосомы;
     */
    double  MUTATION_PROBABILITY = 0.3;

    // Опеределение Cuda сетки.
    int threadsPerBlocks = 256;
    int blocksPerGrid = (POPULATION_SIZE + threadsPerBlocks - 1) / threadsPerBlocks;

    gen* d_gens_val;
    checkCudaErrors(cudaMallocManaged((void**)&d_gens_val, GENS_COUNT * sizeof(gen)));

    chromosome* d_chroms;
    checkCudaErrors(cudaMallocManaged((void**)&d_chroms, POPULATION_SIZE * sizeof(chromosome)));

    population* d_pop;
    checkCudaErrors(cudaMallocManaged((void**)&d_pop, sizeof(population)));

    // Инициализируем псевдослучайную последовательность для генерации генов.
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, sizeof(curandState)));

    // Инициализация псевдослучайной последовательность для генерации начальной популяции.
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, POPULATION_SIZE*sizeof(curandState)));

    //  Инициализация псевдослучайной последовательность для оператора кроссинговера;
    curandState* d_rand_state3;

    // Формируем начальную популяцию.
    d_pop->parents = d_chroms;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        // Выделяем унифицированную память для каждой хромосомы
        checkCudaErrors(cudaMallocManaged((void**)&d_pop->parents[i].citieslist, GENS_COUNT*sizeof(gen)));
        d_pop->parents[i].cities_count = GENS_COUNT;
    }
    d_pop->population_size = POPULATION_SIZE;

    // Формируем гены случайными значениямию.
    rnd_genes_gener<<<1,1>>>(d_gens_val, d_rand_state, GENS_COUNT);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Генерация начальной популяции.
    std::cerr << "Generation of the initial population.\n";
    cudaPopgener<<<blocksPerGrid, threadsPerBlocks>>>(d_pop, d_gens_val, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;

    // Запуск генетического алгоритма.
    //-----------------------------------------------------------------------------------
    std::cerr << "Start gatsp in " << threadsPerBlocks << " threads and " << blocksPerGrid << " blocks.\n";

    start = clock();
    for (int iter = 0; iter < MAX_ITERACTION_LIMIT; ++iter) {
        std::cerr << "Iteration: " << iter << ".\n";

        std::cerr << "Start fitnress.\n";
        cudaFitness<<<blocksPerGrid, threadsPerBlocks>>>(d_pop);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        std::cerr << "Start selection.\n";
        double probability = std::rand() / RAND_MAX + 1.0;
        int new_pop_size = d_pop->set_roul_popsize(probability);
        // Выделение унифицированной памяти для нового состояния популяции.
        checkCudaErrors(cudaMallocManaged((void**)&d_pop->new_parents_state, new_pop_size * sizeof(chromosome)));
        // Генерируем новое состояние популяции.
        d_pop->roul_selection(probability);
        checkCudaErrors(cudaDeviceSynchronize());

        std::cerr << "Start crossover.\n";
        d_pop->new_population_size = d_pop->population_size * d_pop->population_size;
        checkCudaErrors(cudaMallocManaged((void**)&d_rand_state3, d_pop->new_population_size*sizeof(curandState)));
        blocksPerGrid = (d_pop ->new_population_size + threadsPerBlocks - 1) / threadsPerBlocks;
        cudaCrossover<<<blocksPerGrid, threadsPerBlocks>>>(d_pop, d_rand_state3);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
    stop = clock();

    // Вывод результата.
    result_output(d_pop);
    //-----------------------------------------------------------------------------------
    // Заврешение работы генетического алгоритма

    double timer = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer << " seconds.\n";

    // Очистка GPU.
    checkCudaErrors(cudaFree(d_gens_val));
    checkCudaErrors(cudaFree(d_pop->parents));
    checkCudaErrors(cudaFree(d_pop));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(d_rand_state3));
    cudaDeviceReset();
}
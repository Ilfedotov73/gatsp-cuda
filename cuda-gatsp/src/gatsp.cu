#include <Windows.h>

#include <vector>

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../cuda-tspui/include/vec2.h"
#include "../include/city.h"
#include "../include/cities.h"
#include "../include/population.h"

typedef unsigned int uint32;
typedef unsigned long long ullong;

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

void print_debug(population* population)
{
    std::cerr << "Result:\n";
    for (int i = 0; i < population->population_size; ++i) {
        std::cerr << "* " << "chromosome " << i << ": fitness = " << population->parents[i].get_fit() 
            << " || " << population->parents[i].sel_probability << ".\n";
        for (int j = 0; j < population->parents[i].cities_count; ++j) {
            std::cerr << "** " << population->parents[i].citieslist[j].id << ".\n";
        }
    }
}

void print_result(std::vector<double> fitnesses)
{
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    std::cerr << "Result:\n";
    std::cout << fitnesses[0] << " ";
    for (int i = 1, size = fitnesses.size(); i < size; ++i) {
        if (fitnesses[i] < fitnesses[i - 1]) { SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN); }
        else if (fitnesses[i] > fitnesses[i - 1]) { SetConsoleTextAttribute(hConsole, FOREGROUND_RED); }
        else { SetConsoleTextAttribute(hConsole, FOREGROUND_INTENSITY); }
        std::cout << fitnesses[i] << " ";
    }
    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
}

// CudaKernels
//------------------------------------------------------------------------------------------------

/**
 * @brief       Функция cudaPopgener(...) инициализирует генерацию хромосом в начальной
 *              популяции. Каждая хромосома инициализируется в отдельном t.x + b.x потоке.
 *
 * @param[in]   population -- указатель на начальную популяцию;
 * @param[in]   genes -- указатель на область решений;
 * @param[in]   rand_state -- указатель на псевдослучайную последовательность.
 */
__global__ void cudaPopgener(population* population, gen* genes, curandState* rand_state)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= population->population_size) { return; }
    population->popgener(genes, genes[2], &rand_state[index], index);
}

/**
 * @brief       Генерирует псевдослучайную последовательность на population_size потоках GPU.
 * 
 * @param[in]   rand_state -- указатель на псевдослучайную последовательность;
 * @param[in]   population_size -- заданный (текущий размер) популяции;
 * @param[in]   seed -- сид для генерации псевдослучайной последовательности, например, время.
 */
__global__ void randPopInit(curandState* rand_state, int population_size, ullong seed)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= population_size) { return; }
    curand_init(1984+index, seed, 0, &rand_state[index]);
}
//------------------------------------------------------------------------------------------------

/**
 * @brief       Функция cudaFitness(...) запускается t*b потоках. В каждом потоке определятся индекс
 *              соответствующей хромосомы, у которой, вызывается fitness().
 *
 * @param[in]   population -- популяция, для хромосом которой требуется вычислить значение приспособленности.
 */
__global__ void cudaFitness(population* population)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= population->population_size) { return; }
    population->parents[index].fitness();
}

/**
 * @brief       Функция cudaCrossover(...) запускается t*b потоках, шде в каждом потоке происходит вызов
 *              оператора кроссинговера в популяции -- d_pop->popcross(...).
 *
 * @param[in]   population -- популяция, внутри которой необходимо выполнить скрещивание хромосом.
 * @param[in]   parents_src -- прошлое состояние хромосом популяции;
 * @param[in]   rand_state -- указателбя на псевдослучайную последовательность;
 * @param[in]   old_population_size -- размер старой популции (кол-во хромосом в ней).
 */
__global__ void cudaCrossover(population* population, chromosome* parents_src, curandState* rand_state, size_t old_population_size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= population->population_size) { return; }
    population->popcross(&rand_state[index], index, parents_src, old_population_size);
}

/**
 * @brief       Функция cudaMutation(...) запускается в t*b потоках, где в каждом потоке, с заданной
 *              вероятностью (mutation_probability), может пройзойти мутация хромосомы. В каждом потоке
 *              обрабатывается соответствующая потоку хромосома.
 *
 * @param[in]   population -- указатель на популяцию;
 * @param[in]   rand_state  -- указатель на псевдослучайную последовательность;
 * @param[in]   mutation_probability -- вероятность мутации хромосомы;
 */
__global__ void cudaMutation(population* population, curandState* rand_state, double mutation_probability)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= population->population_size) { return; }
    curandState local_rand_state = rand_state[index];
    if (curand_uniform_double(&local_rand_state) < mutation_probability) { return; }
    population->parents[index].mutation(&local_rand_state);
}

/**
 * @brief       Функция chromosome_malloc(...) выделяет память в размер population_size по адресу 
 *              current_chromosome_state.
 * 
 * @param[in]   current_chromosome_state -- текущее состояние хромосом популяции, для которого необходимо
 *              выделить  память;
 * @param[in]   population_size -- размер популяции;
 * @param[in]   genes_count -- кол-во генов в хромосоме.
 */
chromosome* chromosome_malloc(chromosome* current_chromosome_state, size_t population_size, int genes_count)
{
    // Выделяем память для текущего состояния хромосом популяции.
    checkCudaErrors(cudaMallocManaged((void**)&current_chromosome_state, population_size * sizeof(chromosome)));
    for (int i = 0; i < population_size; ++i) {
        // Выделяем память для каждой хромосомы в текущем состоянии популяции.
        current_chromosome_state[i].cities_count = genes_count;
        checkCudaErrors(cudaMallocManaged((void**)&current_chromosome_state[i].citieslist, genes_count * sizeof(gen)));
    }
    return current_chromosome_state;
}

int main()
{
    int GENES_COUNT = 4,
        POPULATION_SIZE = 4,
        MAX_ITERACTION_LIMIT = 15;

    int THREADS_PER_BLOCKS = 512,
        BLOCKS_PER_GRID = (POPULATION_SIZE + THREADS_PER_BLOCKS - 1) / THREADS_PER_BLOCKS;

    dim3 BLOCKS(BLOCKS_PER_GRID),
         THREADS(THREADS_PER_BLOCKS);

    std::vector<double> fitnesses;

    bool DEBUG = false;

    // Заданная вероятность для метода "рулетки".
    double SEL_PROBABILITY = 0.45,
           MUTATION_PROBABILITY = 0.3,
           POPULATIONGROWTH = 0.3;

    // Инициализпци псевдослучайной последовательности для работы с популяцией.
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, POPULATION_SIZE * sizeof(curandState)));
    randPopInit<<<BLOCKS, THREADS>>>(d_rand_state, POPULATION_SIZE, time(NULL));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    gen* d_genes;
    checkCudaErrors(cudaMallocManaged((void**)&d_genes, GENES_COUNT * sizeof(gen)));
    d_genes[0] = gen(vec2(55.7522, 37.6156), 0); // Москва
    d_genes[1] = gen(vec2(55.7887, 49.1221), 1); // Казань
    d_genes[2] = gen(vec2(54.3282, 48.3866), 2); // Ульяновск
    d_genes[3] = gen(vec2(54.9924, 73.3686), 3); // Омск
    checkCudaErrors(cudaDeviceSynchronize());

    // Инициализация состояний хромосом популяции.
    chromosome** d_chromosomes_states;
    size_t d_chromosomes_states_size = MAX_ITERACTION_LIMIT*3;
    checkCudaErrors(cudaMallocManaged((void**)&d_chromosomes_states, d_chromosomes_states_size*sizeof(chromosome*)));
    
    // CURRENT_CHROMOSOME_STATE хранит текущее состояние хромосом популяции. 
    // При разных состояниях CURRENT_CHROMOSOME_STATE, хромосмы попопуляции будут иметь разные значения и разный размер.
    chromosome* CURRENT_CHROMOSOME_STATE = d_chromosomes_states[0];
        
    CURRENT_CHROMOSOME_STATE = chromosome_malloc(CURRENT_CHROMOSOME_STATE, POPULATION_SIZE, GENES_COUNT);
    checkCudaErrors(cudaDeviceSynchronize());

    // OLD_CHROMOSOME_STATE хранит в себе старое состояние хромосом популяции.
    // По умолчанию старое состояние *CHROMOSOME_STATE = текущему состоянию *CHROMOSOME_STATE.
    chromosome* OLD_CHROMOSOME_STATE;
    size_t OLD_POPULATION_SIZE;

    // Инициализая начальной популяции.
    population* d_population;
    checkCudaErrors(cudaMallocManaged((void**)&d_population, sizeof(population)));
    d_population->population_size = POPULATION_SIZE;
    d_population->parents = CURRENT_CHROMOSOME_STATE;
    checkCudaErrors(cudaDeviceSynchronize());

    // Случайное заполнение хромосом начальной популяции генами.
    cudaPopgener<<<BLOCKS, THREADS>>>(d_population, d_genes, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cerr << "Begin population.\n";
    if (DEBUG) { print_debug(d_population); }
     
    // Запуск генетического алгоритма.
    //------------------------------------------------------------------------------------------------
    std::cerr << "Start gatsp in " << THREADS_PER_BLOCKS << " threads and " << BLOCKS_PER_GRID << " blocks.\n";

    for (int i = 0; i < MAX_ITERACTION_LIMIT; ++i) {
        std::cerr << "Start fitness.\n";
        cudaFitness<<<BLOCKS, THREADS>>>(d_population);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        if (DEBUG) { print_debug(d_population); }

        std::cerr << "Start selection.\n";
        // Сохранение старого состояния популяции.
        OLD_CHROMOSOME_STATE = CURRENT_CHROMOSOME_STATE;
        OLD_POPULATION_SIZE = d_population->population_size;

        // Установка нового размера и состояния популяции.
        POPULATION_SIZE = d_population->roul_newpopsize(SEL_PROBABILITY);
        if (POPULATION_SIZE == 0) { break; }
        ++CURRENT_CHROMOSOME_STATE;

        // Выделение памяти для нового состояния популяции.
        CURRENT_CHROMOSOME_STATE = chromosome_malloc(CURRENT_CHROMOSOME_STATE, POPULATION_SIZE, GENES_COUNT);
        d_population->population_size = POPULATION_SIZE;
        d_population->parents = CURRENT_CHROMOSOME_STATE;
        checkCudaErrors(cudaDeviceSynchronize());

        // Отбор хромосом.
        d_population->roul_selection(SEL_PROBABILITY, OLD_CHROMOSOME_STATE, OLD_POPULATION_SIZE);
        checkCudaErrors(cudaDeviceSynchronize());

        if (DEBUG) { print_debug(d_population); }

        std::cerr << "Start crossover.\n";

        // Сохранение старого состояния популяции.
        OLD_CHROMOSOME_STATE = CURRENT_CHROMOSOME_STATE;
        OLD_POPULATION_SIZE = d_population->population_size;

        POPULATION_SIZE = d_population->population_size + (d_population->population_size * POPULATIONGROWTH);

        ++CURRENT_CHROMOSOME_STATE;
        // Выделение памяти для нового состояния популяции.
        CURRENT_CHROMOSOME_STATE = chromosome_malloc(CURRENT_CHROMOSOME_STATE, POPULATION_SIZE, GENES_COUNT);
        d_population->population_size = POPULATION_SIZE;
        d_population->parents = CURRENT_CHROMOSOME_STATE;
        checkCudaErrors(cudaDeviceSynchronize());

        cudaCrossover<<<BLOCKS, THREADS>>>(d_population, OLD_CHROMOSOME_STATE, d_rand_state, OLD_POPULATION_SIZE);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        if (DEBUG) { print_debug(d_population); }

        std::cerr << "Start mutation.\n";
        // Выделяем псевдослучаную последовательность для популяции нового размера
        randPopInit<<<BLOCKS, THREADS>>>(d_rand_state, POPULATION_SIZE, time(NULL));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        cudaMutation<<<BLOCKS, THREADS>>>(d_population, d_rand_state, MUTATION_PROBABILITY);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        if (DEBUG) { print_debug(d_population); }

        double fitness_min = d_population->parents[0].get_fit();
        for (int i = 1; i < d_population->population_size; ++i) {
            double fitness_temp = d_population->parents[i].get_fit();
            if (fitness_min > fitness_temp) { fitness_min = fitness_temp; }
        }
        fitnesses.push_back(fitness_min);
    }
    std::cerr << "Start fitness.\n";
    cudaFitness<<<BLOCKS, THREADS>>>(d_population);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    double fitness_min = d_population->parents[0].get_fit();
    for (int i = 1; i < d_population->population_size; ++i) {
        double fitness_temp = d_population->parents[i].get_fit();
        if (fitness_min > fitness_temp) { fitness_min = fitness_temp; }
    }
    fitnesses.push_back(fitness_min);

    if (DEBUG) { print_debug(d_population); }

    print_result(fitnesses);
    //------------------------------------------------------------------------------------------------

    // Очистка памяти
    checkCudaErrors(cudaGetLastError());
    //------------------------------------------------------------------------------------------------
    cudaFree(d_genes);
    for (int i = 0; i < d_chromosomes_states_size; ++i) { cudaFree(d_chromosomes_states[i]); }
    cudaFree(d_chromosomes_states);
    cudaFree(d_population);
    cudaFree(d_rand_state);
    cudaDeviceReset();
}
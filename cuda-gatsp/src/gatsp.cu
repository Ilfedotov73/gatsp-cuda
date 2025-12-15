//#include <Windows.h>

#include <time.h>
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

const std::string RED = "\033[31m";
const std::string GREEN = "\033[32m";
const std::string BLUE = "\033[34m";
const std::string YELLOW = "\033[33m";
const std::string RESET = "\033[0m"; 

// linux.
void print_result(std::vector<double> fitnesses)
{
    std::cerr << "Result:\n";
    std::cout << fitnesses[0] << " ";
    for (int i = 1, size = fitnesses.size(); i < size; ++i) {
        if (fitnesses[i] < fitnesses[i - 1]) { std::cout << GREEN; }
        else if (fitnesses[i] > fitnesses[i - 1]) { std::cout << RED; }
        else { std::cout << YELLOW; }
        std::cout << fitnesses[i] << " ";
    }
    std::cout << RESET << '\n';
}

/* windows. 
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
*/

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
    int GENES_COUNT = 100,
        POPULATION_SIZE = 30,
        MAX_ITERACTION_LIMIT = 20;

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
    // genes list.
    {
        d_genes[0] = gen(vec2(40.7128, -74.0060), 0); // Нью-Йорк
        d_genes[1] = gen(vec2(34.0522, -118.2437), 1); // Лос-Анджелес
        d_genes[2] = gen(vec2(51.5074, -0.1278), 2); // Лондон
        d_genes[3] = gen(vec2(48.8566, 2.3522), 3); // Париж
        d_genes[4] = gen(vec2(35.6762, 139.6503), 4); // Токио
        d_genes[5] = gen(vec2(52.5200, 13.4050), 5); // Берлин
        d_genes[6] = gen(vec2(55.7558, 37.6173), 6); // Москва
        d_genes[7] = gen(vec2(19.4326, -99.1332), 7); // Мехико
        d_genes[8] = gen(vec2(28.6139, 77.2090), 8); // Нью-Дели
        d_genes[9] = gen(vec2(23.1291, 113.2644), 9); // Гуанчжоу
        d_genes[10] = gen(vec2(41.9028, 12.4964), 10); // Рим
        d_genes[11] = gen(vec2(37.7749, -122.4194), 11); // Сан-Франциско
        d_genes[12] = gen(vec2(31.2304, 121.4737), 12); // Шанхай
        d_genes[13] = gen(vec2(39.9042, 116.4074), 13); // Пекин
        d_genes[14] = gen(vec2(37.5665, 126.9780), 14); // Сеул
        d_genes[15] = gen(vec2(55.6761, 12.5683), 15); // Копенгаген
        d_genes[16] = gen(vec2(59.3293, 18.0686), 16); // Стокгольм
        d_genes[17] = gen(vec2(60.1699, 24.9384), 17); // Хельсинки
        d_genes[18] = gen(vec2(59.9139, 10.7522), 18); // Осло
        d_genes[19] = gen(vec2(52.3676, 4.9041), 19); // Амстердам
        d_genes[20] = gen(vec2(50.8503, 4.3517), 20); // Брюссель
        d_genes[21] = gen(vec2(38.7223, -9.1393), 21); // Лиссабон
        d_genes[22] = gen(vec2(41.3851, 2.1734), 22); // Барселона
        d_genes[23] = gen(vec2(40.4168, -3.7038), 23); // Мадрид
        d_genes[24] = gen(vec2(45.4642, 9.1900), 24); // Милан
        d_genes[25] = gen(vec2(43.7696, 11.2558), 25); // Флоренция
        d_genes[26] = gen(vec2(50.1109, 8.6821), 26); // Франкфурт
        d_genes[27] = gen(vec2(53.5511, 9.9937), 27); // Гамбург
        d_genes[28] = gen(vec2(48.1351, 11.5820), 28); // Мюнхен
        d_genes[29] = gen(vec2(47.3769, 8.5417), 29); // Цюрих
        d_genes[30] = gen(vec2(46.9480, 7.4474), 30); // Берн
        d_genes[31] = gen(vec2(45.5017, -73.5673), 31); // Монреаль
        d_genes[32] = gen(vec2(43.6532, -79.3832), 32); // Торонто
        d_genes[33] = gen(vec2(49.2827, -123.1207), 33); // Ванкувер
        d_genes[34] = gen(vec2(41.8781, -87.6298), 34); // Чикаго
        d_genes[35] = gen(vec2(25.7617, -80.1918), 35); // Майами
        d_genes[36] = gen(vec2(32.7767, -96.7970), 36); // Даллас
        d_genes[37] = gen(vec2(29.7604, -95.3698), 37); // Хьюстон
        d_genes[38] = gen(vec2(33.4484, -112.0740), 38); // Финикс
        d_genes[39] = gen(vec2(39.7392, -104.9903), 39); // Денвер
        d_genes[40] = gen(vec2(45.5152, -122.6784), 40); // Портленд
        d_genes[41] = gen(vec2(47.6062, -122.3321), 41); // Сиэтл
        d_genes[42] = gen(vec2(32.7157, -117.1611), 42); // Сан-Диего
        d_genes[43] = gen(vec2(36.1699, -115.1398), 43); // Лас-Вегас
        d_genes[44] = gen(vec2(30.2672, -97.7431), 44); // Остин
        d_genes[45] = gen(vec2(33.7490, -84.3880), 45); // Атланта
        d_genes[46] = gen(vec2(42.3601, -71.0589), 46); // Бостон
        d_genes[47] = gen(vec2(39.9526, -75.1652), 47); // Филадельфия
        d_genes[48] = gen(vec2(38.9072, -77.0369), 48); // Вашингтон
        d_genes[49] = gen(vec2(37.5407, -77.4360), 49); // Ричмонд
        d_genes[50] = gen(vec2(35.2271, -80.8431), 50); // Шарлотт
        d_genes[51] = gen(vec2(27.9506, -82.4572), 51); // Тампа
        d_genes[52] = gen(vec2(28.5383, -81.3792), 52); // Орландо
        d_genes[53] = gen(vec2(39.7684, -86.1581), 53); // Индианаполис
        d_genes[54] = gen(vec2(39.1031, -84.5120), 54); // Цинциннати
        d_genes[55] = gen(vec2(42.3314, -83.0458), 55); // Детройт
        d_genes[56] = gen(vec2(44.9778, -93.2650), 56); // Миннеаполис
        d_genes[57] = gen(vec2(38.6270, -90.1994), 57); // Сент-Луис
        d_genes[58] = gen(vec2(39.0997, -94.5786), 58); // Канзас-Сити
        d_genes[59] = gen(vec2(35.1495, -90.0490), 59); // Мемфис
        d_genes[60] = gen(vec2(36.1627, -86.7816), 60); // Нашвилл
        d_genes[61] = gen(vec2(29.9511, -90.0715), 61); // Новый Орлеан
        d_genes[62] = gen(vec2(32.2988, -90.1848), 62); // Джексон
        d_genes[63] = gen(vec2(39.9612, -82.9988), 63); // Колумбус
        d_genes[64] = gen(vec2(40.4406, -79.9959), 64); // Питтсбург
        d_genes[65] = gen(vec2(41.4993, -81.6944), 65); // Кливленд
        d_genes[66] = gen(vec2(42.8864, -78.8784), 66); // Буффало
        d_genes[67] = gen(vec2(43.1610, -77.6109), 67); // Рочестер
        d_genes[68] = gen(vec2(38.2527, -85.7585), 68); // Луисвилл
        d_genes[69] = gen(vec2(38.6270, -90.1994), 69); // Сент-Луис
        d_genes[70] = gen(vec2(39.7392, -104.9903), 70); // Денвер
        d_genes[71] = gen(vec2(33.4484, -112.0740), 71); // Финикс
        d_genes[72] = gen(vec2(36.1699, -115.1398), 72); // Лас-Вегас
        d_genes[73] = gen(vec2(32.7157, -117.1611), 73); // Сан-Диего
        d_genes[74] = gen(vec2(34.0522, -118.2437), 74); // Лос-Анджелес
        d_genes[75] = gen(vec2(37.7749, -122.4194), 75); // Сан-Франциско
        d_genes[76] = gen(vec2(45.5051, -122.6750), 76); // Портленд
        d_genes[77] = gen(vec2(47.6062, -122.3321), 77); // Сиэтл
        d_genes[78] = gen(vec2(61.2181, -149.9003), 78); // Анкоридж
        d_genes[79] = gen(vec2(21.3069, -157.8583), 79); // Гонолулу
        d_genes[80] = gen(vec2(44.0582, -121.3153), 80); // Бенд
        d_genes[81] = gen(vec2(40.5865, -122.3917), 81); // Реддинг
        d_genes[82] = gen(vec2(38.5816, -121.4944), 82); // Сакраменто
        d_genes[83] = gen(vec2(36.7372, -119.7871), 83); // Фресно
        d_genes[84] = gen(vec2(35.2828, -120.6596), 84); // Сан-Луис-Обиспо
        d_genes[85] = gen(vec2(34.4208, -119.6982), 85); // Санта-Барбара
        d_genes[86] = gen(vec2(34.0522, -118.2437), 86); // Лос-Анджелес
        d_genes[87] = gen(vec2(33.6846, -117.8265), 87); // Ирвайн
        d_genes[88] = gen(vec2(33.8358, -117.9113), 88); // Анахайм
        d_genes[89] = gen(vec2(32.7157, -117.1611), 89); // Сан-Диего
        d_genes[90] = gen(vec2(32.2226, -110.9747), 90); // Тусон
        d_genes[91] = gen(vec2(35.0844, -106.6504), 91); // Альбукерке
        d_genes[92] = gen(vec2(39.7392, -104.9903), 92); // Денвер
        d_genes[93] = gen(vec2(41.8781, -87.6298), 93); // Чикаго
        d_genes[94] = gen(vec2(42.3314, -83.0458), 94); // Детройт
        d_genes[95] = gen(vec2(43.6532, -79.3832), 95); // Торонто
        d_genes[96] = gen(vec2(45.5017, -73.5673), 96); // Монреаль
        d_genes[97] = gen(vec2(49.2827, -123.1207), 97); // Ванкувер
        d_genes[98] = gen(vec2(51.0447, -114.0719), 98); // Калгари
        d_genes[99] = gen(vec2(53.5461, -113.4938), 99); // Эдмонтон
        d_genes[100] = gen(vec2(45.4215, -75.6972), 100); // Оттава
        d_genes[101] = gen(vec2(43.6511, -79.3470), 101); // Торонто
        d_genes[102] = gen(vec2(49.8951, -97.1384), 102); // Виннипег
        d_genes[103] = gen(vec2(50.4547, -104.6067), 103); // Регина
        d_genes[104] = gen(vec2(52.1304, -106.6608), 104); // Саскатун
        d_genes[105] = gen(vec2(44.6488, -63.5752), 105); // Галифакс
        d_genes[106] = gen(vec2(46.8139, -71.2080), 106); // Квебек
        d_genes[107] = gen(vec2(43.8563, -79.3370), 107); // Маркем
        d_genes[108] = gen(vec2(43.5890, -79.6441), 108); // Миссиссога
        d_genes[109] = gen(vec2(49.1666, -123.1336), 109); // Ричмонд
        d_genes[110] = gen(vec2(49.1913, -122.8490), 110); // Суррей
        d_genes[111] = gen(vec2(-34.6037, -58.3816), 111); // Буэнос-Айрес
        d_genes[112] = gen(vec2(-23.5505, -46.6333), 112); // Сан-Паулу
        d_genes[113] = gen(vec2(-22.9068, -43.1729), 113); // Рио-де-Жанейро
        d_genes[114] = gen(vec2(-15.7975, -47.8919), 114); // Бразилиа
        d_genes[115] = gen(vec2(-12.0464, -77.0428), 115); // Лима
        d_genes[116] = gen(vec2(-33.4489, -70.6693), 116); // Сантьяго
        d_genes[117] = gen(vec2(4.7110, -74.0721), 117); // Богота
        d_genes[118] = gen(vec2(10.4806, -66.9036), 118); // Каракас
        d_genes[119] = gen(vec2(-0.1807, -78.4678), 119); // Кито
        d_genes[120] = gen(vec2(-34.9011, -56.1645), 120); // Монтевидео
        d_genes[121] = gen(vec2(-25.2637, -57.5759), 121); // Асунсьон
        d_genes[122] = gen(vec2(-16.5000, -68.1500), 122); // Ла-Пас
        d_genes[123] = gen(vec2(-17.3895, -66.1568), 123); // Кочабамба
        d_genes[124] = gen(vec2(-12.0464, -77.0428), 124); // Лима
        d_genes[125] = gen(vec2(-1.8312, -78.1834), 125); // Куэнка
        d_genes[126] = gen(vec2(-8.7832, -55.4915), 126); // Сантарен
        d_genes[127] = gen(vec2(-3.1190, -60.0217), 127); // Манаус
        d_genes[128] = gen(vec2(-1.4558, -48.4902), 128); // Белен
        d_genes[129] = gen(vec2(-3.7172, -38.5434), 129); // Форталеза
        d_genes[130] = gen(vec2(-8.0476, -34.8770), 130); // Ресифи
        d_genes[131] = gen(vec2(-9.9747, -67.8100), 131); // Риу-Бранку
        d_genes[132] = gen(vec2(-12.9704, -38.5124), 132); // Салвадор
        d_genes[133] = gen(vec2(-19.9167, -43.9345), 133); // Белу-Оризонти
        d_genes[134] = gen(vec2(-20.2870, -40.3087), 134); // Витория
        d_genes[135] = gen(vec2(-22.9068, -43.1729), 135); // Рио-де-Жанейро
        d_genes[136] = gen(vec2(-23.5505, -46.6333), 136); // Сан-Паулу
        d_genes[137] = gen(vec2(-25.4278, -49.2731), 137); // Куритиба
        d_genes[138] = gen(vec2(-26.3044, -48.8461), 138); // Жоинвиль
        d_genes[139] = gen(vec2(-27.5954, -48.5480), 139); // Флорианополис
        d_genes[140] = gen(vec2(-30.0346, -51.2177), 140); // Порту-Алегри
        d_genes[141] = gen(vec2(-31.4167, -64.1833), 141); // Кордова
        d_genes[142] = gen(vec2(-32.8895, -68.8458), 142); // Мендоса
        d_genes[143] = gen(vec2(-34.6037, -58.3816), 143); // Буэнос-Айрес
        d_genes[144] = gen(vec2(-34.9011, -56.1645), 144); // Монтевидео
        d_genes[145] = gen(vec2(-38.4161, -63.6167), 145); // Баия-Бланка
        d_genes[146] = gen(vec2(-41.1335, -71.3103), 146); // Сан-Карлос-де-Барилоче
        d_genes[147] = gen(vec2(-45.8641, -67.4966), 147); // Комодоро-Ривадавия
        d_genes[148] = gen(vec2(-51.6230, -69.2168), 148); // Рио-Гальегос
        d_genes[149] = gen(vec2(-53.1638, -70.9171), 149); // Пунта-Аренас
        d_genes[150] = gen(vec2(-33.9249, 18.4241), 150); // Кейптаун
        d_genes[151] = gen(vec2(-26.2041, 28.0473), 151); // Йоханнесбург
        d_genes[152] = gen(vec2(-29.8587, 31.0218), 152); // Дурбан
        d_genes[153] = gen(vec2(-25.7461, 28.1881), 153); // Претория
        d_genes[154] = gen(vec2(-33.9608, 25.6022), 154); // Порт-Элизабет
        d_genes[155] = gen(vec2(-18.1416, 178.4419), 155); // Сува
        d_genes[156] = gen(vec2(-17.7415, 168.3154), 156); // Порт-Вила
        d_genes[157] = gen(vec2(-9.4438, 147.1803), 157); // Порт-Морсби
        d_genes[158] = gen(vec2(-35.2809, 149.1300), 158); // Канберра
        d_genes[159] = gen(vec2(-33.8688, 151.2093), 159); // Сидней
        d_genes[160] = gen(vec2(-37.8136, 144.9631), 160); // Мельбурн
        d_genes[161] = gen(vec2(-27.4698, 153.0251), 161); // Брисбен
        d_genes[162] = gen(vec2(-31.9505, 115.8605), 162); // Перт
        d_genes[163] = gen(vec2(-34.9285, 138.6007), 163); // Аделаида
        d_genes[164] = gen(vec2(-12.4634, 130.8456), 164); // Дарвин
        d_genes[165] = gen(vec2(-42.8821, 147.3272), 165); // Хобарт
        d_genes[166] = gen(vec2(-36.8485, 174.7633), 166); // Окленд
        d_genes[167] = gen(vec2(-41.2865, 174.7762), 167); // Веллингтон
        d_genes[168] = gen(vec2(-43.5321, 172.6362), 168); // Крайстчерч
        d_genes[169] = gen(vec2(-45.8788, 170.5028), 169); // Данидин
        d_genes[170] = gen(vec2(-29.0534, 167.9589), 170); // Норфолк
        d_genes[171] = gen(vec2(-9.4456, 159.9729), 171); // Хониара
        d_genes[172] = gen(vec2(-13.2820, -176.1764), 172); // Мата-Уту
        d_genes[173] = gen(vec2(-21.1343, -175.2018), 173); // Нукуалофа
        d_genes[174] = gen(vec2(-13.8507, -171.7514), 174); // Апиа
        d_genes[175] = gen(vec2(1.3521, 103.8198), 175); // Сингапур
        d_genes[176] = gen(vec2(3.1390, 101.6869), 176); // Куала-Лумпур
        d_genes[177] = gen(vec2(13.7563, 100.5018), 177); // Бангкок
        d_genes[178] = gen(vec2(21.0278, 105.8342), 178); // Ханой
        d_genes[179] = gen(vec2(10.8231, 106.6297), 179); // Хошимин
        d_genes[180] = gen(vec2(14.5995, 120.9842), 180); // Манила
        d_genes[181] = gen(vec2(1.2921, 103.8558), 181); // Сингапур
        d_genes[182] = gen(vec2(-6.2088, 106.8456), 182); // Джакарта
        d_genes[183] = gen(vec2(16.0544, 108.2022), 183); // Дананг
        d_genes[184] = gen(vec2(12.8797, 121.7740), 184); // Филиппины
        d_genes[185] = gen(vec2(5.4141, 100.3288), 185); // Пенанг
        d_genes[186] = gen(vec2(4.2105, 101.9758), 186); // Ипох
        d_genes[187] = gen(vec2(1.5580, 103.6381), 187); // Джохор-Бару
        d_genes[188] = gen(vec2(6.1254, 100.3670), 188); // Алор-Сетар
        d_genes[189] = gen(vec2(2.1896, 102.2501), 189); // Малакка
        d_genes[190] = gen(vec2(3.0738, 101.5183), 190); // Петалинг-Джая
        d_genes[191] = gen(vec2(5.9788, 116.0753), 191); // Кота-Кинабалу
        d_genes[192] = gen(vec2(1.5535, 110.3593), 192); // Кучинг
        d_genes[193] = gen(vec2(4.8857, 114.9317), 193); // Бандар-Сери-Бегаван
        d_genes[194] = gen(vec2(13.4125, 103.8665), 194); // Сиемреап
        d_genes[195] = gen(vec2(11.5564, 104.9282), 195); // Пномпень
        d_genes[196] = gen(vec2(17.9757, 102.6331), 196); // Вьентьян
        d_genes[197] = gen(vec2(20.5937, 78.9629), 197); // Индия
        d_genes[198] = gen(vec2(28.6139, 77.2090), 198); // Нью-Дели
        d_genes[199] = gen(vec2(19.0760, 72.8777), 199); // Мумбаи
        d_genes[200] = gen(vec2(13.0827, 80.2707), 200); // Ченнаи
        d_genes[201] = gen(vec2(12.9716, 77.5946), 201); // Бангалор
        d_genes[202] = gen(vec2(17.3850, 78.4867), 202); // Хайдарабад
        d_genes[203] = gen(vec2(22.5726, 88.3639), 203); // Калькутта
        d_genes[204] = gen(vec2(26.9124, 75.7873), 204); // Джайпур
        d_genes[205] = gen(vec2(30.7333, 76.7794), 205); // Чандигарх
        d_genes[206] = gen(vec2(23.0225, 72.5714), 206); // Ахмадабад
        d_genes[207] = gen(vec2(18.5204, 73.8567), 207); // Пуна
        d_genes[208] = gen(vec2(15.2993, 74.1240), 208); // Гоа
        d_genes[209] = gen(vec2(9.9312, 76.2673), 209); // Кочин
        d_genes[210] = gen(vec2(11.0168, 76.9558), 210); // Коимбатур
        d_genes[211] = gen(vec2(10.7905, 78.7047), 211); // Тируччираппалли
        d_genes[212] = gen(vec2(25.5941, 85.1376), 212); // Патна
        d_genes[213] = gen(vec2(26.4499, 80.3319), 213); // Канпур
        d_genes[214] = gen(vec2(27.1767, 78.0081), 214); // Агра
        d_genes[215] = gen(vec2(28.6791, 77.0697), 215); // Газиабад
        d_genes[216] = gen(vec2(30.3165, 78.0322), 216); // Дехрадун
        d_genes[217] = gen(vec2(31.1048, 77.1734), 217); // Шимла
        d_genes[218] = gen(vec2(32.7266, 74.8570), 218); // Джамму
        d_genes[219] = gen(vec2(34.0837, 74.7973), 219); // Сринагар
        d_genes[220] = gen(vec2(35.9110, 79.0084), 220); // Лех
        d_genes[221] = gen(vec2(11.1271, 78.6569), 221); // Салем
        d_genes[222] = gen(vec2(8.0883, 77.5385), 222); // Каньякумари
        d_genes[223] = gen(vec2(15.8281, 78.0373), 223); // Курнул
        d_genes[224] = gen(vec2(16.5062, 80.6480), 224); // Виджаявада
        d_genes[225] = gen(vec2(17.6868, 83.2185), 225); // Вишакхапатнам
        d_genes[226] = gen(vec2(20.2961, 85.8245), 226); // Бхубанешвар
        d_genes[227] = gen(vec2(22.5726, 88.3639), 227); // Калькутта
        d_genes[228] = gen(vec2(23.8103, 90.4125), 228); // Дакка
        d_genes[229] = gen(vec2(24.3636, 88.6241), 229); // Раджшахи
        d_genes[230] = gen(vec2(22.3569, 91.7832), 230); // Читтагонг
        d_genes[231] = gen(vec2(21.4272, 92.0058), 231); // Кокс-Базар
        d_genes[232] = gen(vec2(27.7172, 85.3240), 232); // Катманду
        d_genes[233] = gen(vec2(27.4728, 89.6390), 233); // Тхимпху
        d_genes[234] = gen(vec2(6.9271, 79.8612), 234); // Коломбо
        d_genes[235] = gen(vec2(7.8731, 80.7718), 235); // Канди
        d_genes[236] = gen(vec2(9.6615, 80.0255), 236); // Джафна
        d_genes[237] = gen(vec2(6.0535, 80.2210), 237); // Галле
        d_genes[238] = gen(vec2(25.2048, 55.2708), 238); // Дубай
        d_genes[239] = gen(vec2(24.4539, 54.3773), 239); // Абу-Даби
        d_genes[240] = gen(vec2(25.2854, 51.5310), 240); // Доха
        d_genes[241] = gen(vec2(29.3759, 47.9774), 241); // Эль-Кувейт
        d_genes[242] = gen(vec2(26.2235, 50.5876), 242); // Манама
        d_genes[243] = gen(vec2(24.7136, 46.6753), 243); // Эр-Рияд
        d_genes[244] = gen(vec2(21.4858, 39.1925), 244); // Мекка
        d_genes[245] = gen(vec2(24.4672, 39.6111), 245); // Медина
        d_genes[246] = gen(vec2(30.0444, 31.2357), 246); // Каир
        d_genes[247] = gen(vec2(31.2001, 29.9187), 247); // Александрия
        d_genes[248] = gen(vec2(36.8969, 30.7133), 248); // Анталья
        d_genes[249] = gen(vec2(41.0082, 28.9784), 249); // Стамбул
        d_genes[250] = gen(vec2(39.9334, 32.8597), 250); // Анкара
        d_genes[251] = gen(vec2(38.4237, 27.1428), 251); // Измир
        d_genes[252] = gen(vec2(36.9864, 35.3253), 252); // Адана
        d_genes[253] = gen(vec2(40.1885, 29.0610), 253); // Бурса
        d_genes[254] = gen(vec2(37.8720, 32.4841), 254); // Конья
        d_genes[255] = gen(vec2(41.0151, 28.9795), 255); // Стамбул
        d_genes[256] = gen(vec2(33.5138, 36.2765), 256); // Дамаск
        d_genes[257] = gen(vec2(33.8938, 35.5018), 257); // Бейрут
        d_genes[258] = gen(vec2(31.9466, 35.3027), 258); // Амман
        d_genes[259] = gen(vec2(31.7683, 35.2137), 259); // Иерусалим
        d_genes[260] = gen(vec2(32.0853, 34.7818), 260); // Тель-Авив
        d_genes[261] = gen(vec2(31.0461, 34.8516), 261); // Беэр-Шева
        d_genes[262] = gen(vec2(32.7940, 34.9896), 262); // Хайфа
        d_genes[263] = gen(vec2(29.5581, 34.9482), 263); // Эйлат
        d_genes[264] = gen(vec2(33.3100, 44.3460), 264); // Багдад
        d_genes[265] = gen(vec2(33.3152, 44.3661), 265); // Багдад
        d_genes[266] = gen(vec2(36.1911, 44.0092), 266); // Эрбиль
        d_genes[267] = gen(vec2(35.6892, 51.3890), 267); // Тегеран
        d_genes[268] = gen(vec2(36.2687, 59.5677), 268); // Мешхед
        d_genes[269] = gen(vec2(32.6546, 51.6680), 269); // Исфахан
        d_genes[270] = gen(vec2(29.5918, 52.5837), 270); // Шираз
        d_genes[271] = gen(vec2(38.0962, 46.2738), 271); // Тебриз
        d_genes[272] = gen(vec2(34.3416, 47.0861), 272); // Керманшах
        d_genes[273] = gen(vec2(35.7000, 51.4000), 273); // Кередж
        d_genes[274] = gen(vec2(30.2839, 57.0833), 274); // Керман
        d_genes[275] = gen(vec2(27.1832, 56.2666), 275); // Бендер-Аббас
        d_genes[276] = gen(vec2(25.2854, 51.5310), 276); // Доха
        d_genes[277] = gen(vec2(24.4539, 54.3773), 277); // Абу-Даби
        d_genes[278] = gen(vec2(25.2048, 55.2708), 278); // Дубай
        d_genes[279] = gen(vec2(25.0759, 55.1348), 279); // Шарджа
        d_genes[280] = gen(vec2(24.4667, 54.3667), 280); // Абу-Даби
        d_genes[281] = gen(vec2(23.6142, 58.5925), 281); // Маскат
        d_genes[282] = gen(vec2(26.2235, 50.5876), 282); // Манама
        d_genes[283] = gen(vec2(29.3759, 47.9774), 283); // Эль-Кувейт
        d_genes[284] = gen(vec2(24.7136, 46.6753), 284); // Эр-Рияд
        d_genes[285] = gen(vec2(21.4858, 39.1925), 285); // Мекка
        d_genes[286] = gen(vec2(24.4672, 39.6111), 286); // Медина
        d_genes[287] = gen(vec2(30.0444, 31.2357), 287); // Каир
        d_genes[288] = gen(vec2(31.2001, 29.9187), 288); // Александрия
        d_genes[289] = gen(vec2(25.6872, 32.6396), 289); // Луксор
        d_genes[290] = gen(vec2(24.0889, 32.8998), 290); // Асуан
        d_genes[291] = gen(vec2(31.2058, 29.9249), 291); // Александрия
        d_genes[292] = gen(vec2(30.0444, 31.2357), 292); // Каир
        d_genes[293] = gen(vec2(36.8969, 30.7133), 293); // Анталья
        d_genes[294] = gen(vec2(41.0082, 28.9784), 294); // Стамбул
        d_genes[295] = gen(vec2(39.9334, 32.8597), 295); // Анкара
        d_genes[296] = gen(vec2(38.4237, 27.1428), 296); // Измир
        d_genes[297] = gen(vec2(36.9864, 35.3253), 297); // Адана
        d_genes[298] = gen(vec2(40.1885, 29.0610), 298); // Бурса
        d_genes[299] = gen(vec2(37.8720, 32.4841), 299); // Конья
        d_genes[300] = gen(vec2(41.0151, 28.9795), 300); // Стамбул
        d_genes[301] = gen(vec2(33.5138, 36.2765), 301); // Дамаск
        d_genes[302] = gen(vec2(33.8938, 35.5018), 302); // Бейрут
        d_genes[303] = gen(vec2(31.9466, 35.3027), 303); // Амман
        d_genes[304] = gen(vec2(31.7683, 35.2137), 304); // Иерусалим
        d_genes[305] = gen(vec2(32.0853, 34.7818), 305); // Тель-Авив
        d_genes[306] = gen(vec2(31.0461, 34.8516), 306); // Беэр-Шева
        d_genes[307] = gen(vec2(32.7940, 34.9896), 307); // Хайфа
        d_genes[308] = gen(vec2(29.5581, 34.9482), 308); // Эйлат
        d_genes[309] = gen(vec2(33.3100, 44.3460), 309); // Багдад
        d_genes[310] = gen(vec2(33.3152, 44.3661), 310); // Багдад
        d_genes[311] = gen(vec2(36.1911, 44.0092), 311); // Эрбиль
        d_genes[312] = gen(vec2(35.6892, 51.3890), 312); // Тегеран
        d_genes[313] = gen(vec2(36.2687, 59.5677), 313); // Мешхед
        d_genes[314] = gen(vec2(32.6546, 51.6680), 314); // Исфахан
        d_genes[315] = gen(vec2(29.5918, 52.5837), 315); // Шираз
        d_genes[316] = gen(vec2(38.0962, 46.2738), 316); // Тебриз
        d_genes[317] = gen(vec2(34.3416, 47.0861), 317); // Керманшах
        d_genes[318] = gen(vec2(35.7000, 51.4000), 318); // Кередж
        d_genes[319] = gen(vec2(30.2839, 57.0833), 319); // Керман
        d_genes[320] = gen(vec2(27.1832, 56.2666), 320); // Бендер-Аббас
        d_genes[321] = gen(vec2(25.2854, 51.5310), 321); // Доха
        d_genes[322] = gen(vec2(24.4539, 54.3773), 322); // Абу-Даби
        d_genes[323] = gen(vec2(25.2048, 55.2708), 323); // Дубай
        d_genes[324] = gen(vec2(25.0759, 55.1348), 324); // Шарджа
        d_genes[325] = gen(vec2(24.4667, 54.3667), 325); // Абу-Даби
        d_genes[326] = gen(vec2(23.6142, 58.5925), 326); // Маскат
        d_genes[327] = gen(vec2(26.2235, 50.5876), 327); // Манама
        d_genes[328] = gen(vec2(29.3759, 47.9774), 328); // Эль-Кувейт
        d_genes[329] = gen(vec2(24.7136, 46.6753), 329); // Эр-Рияд
        d_genes[330] = gen(vec2(21.4858, 39.1925), 330); // Мекка
        d_genes[331] = gen(vec2(24.4672, 39.6111), 331); // Медина
        d_genes[332] = gen(vec2(30.0444, 31.2357), 332); // Каир
        d_genes[333] = gen(vec2(31.2001, 29.9187), 333); // Александрия
        d_genes[334] = gen(vec2(25.6872, 32.6396), 334); // Луксор
        d_genes[335] = gen(vec2(24.0889, 32.8998), 335); // Асуан
        d_genes[336] = gen(vec2(31.2058, 29.9249), 336); // Александрия
        d_genes[337] = gen(vec2(30.0444, 31.2357), 337); // Каир
        d_genes[338] = gen(vec2(36.8969, 30.7133), 338); // Анталья
        d_genes[339] = gen(vec2(41.0082, 28.9784), 339); // Стамбул
        d_genes[340] = gen(vec2(39.9334, 32.8597), 340); // Анкара
        d_genes[341] = gen(vec2(38.4237, 27.1428), 341); // Измир
        d_genes[342] = gen(vec2(36.9864, 35.3253), 342); // Адана
        d_genes[343] = gen(vec2(40.1885, 29.0610), 343); // Бурса
        d_genes[344] = gen(vec2(37.8720, 32.4841), 344); // Конья
        d_genes[345] = gen(vec2(41.0151, 28.9795), 345); // Стамбул
        d_genes[346] = gen(vec2(33.5138, 36.2765), 346); // Дамаск
        d_genes[347] = gen(vec2(33.8938, 35.5018), 347); // Бейрут
        d_genes[348] = gen(vec2(31.9466, 35.3027), 348); // Амман
        d_genes[349] = gen(vec2(31.7683, 35.2137), 349); // Иерусалим
        d_genes[350] = gen(vec2(32.0853, 34.7818), 350); // Тель-Авив
        d_genes[351] = gen(vec2(31.0461, 34.8516), 351); // Беэр-Шева
        d_genes[352] = gen(vec2(32.7940, 34.9896), 352); // Хайфа
        d_genes[353] = gen(vec2(29.5581, 34.9482), 353); // Эйлат
        d_genes[354] = gen(vec2(33.3100, 44.3460), 354); // Багдад
        d_genes[355] = gen(vec2(33.3152, 44.3661), 355); // Багдад
        d_genes[356] = gen(vec2(36.1911, 44.0092), 356); // Эрбиль
        d_genes[357] = gen(vec2(35.6892, 51.3890), 357); // Тегеран
        d_genes[358] = gen(vec2(36.2687, 59.5677), 358); // Мешхед
        d_genes[359] = gen(vec2(32.6546, 51.6680), 359); // Исфахан
        d_genes[360] = gen(vec2(29.5918, 52.5837), 360); // Шираз
        d_genes[361] = gen(vec2(38.0962, 46.2738), 361); // Тебриз
        d_genes[362] = gen(vec2(34.3416, 47.0861), 362); // Керманшах
        d_genes[363] = gen(vec2(35.7000, 51.4000), 363); // Кередж
        d_genes[364] = gen(vec2(30.2839, 57.0833), 364); // Керман
        d_genes[365] = gen(vec2(27.1832, 56.2666), 365); // Бендер-Аббас
        d_genes[366] = gen(vec2(25.2854, 51.5310), 366); // Доха
        d_genes[367] = gen(vec2(24.4539, 54.3773), 367); // Абу-Даби
        d_genes[368] = gen(vec2(25.2048, 55.2708), 368); // Дубай
        d_genes[369] = gen(vec2(25.0759, 55.1348), 369); // Шарджа
        d_genes[370] = gen(vec2(24.4667, 54.3667), 370); // Абу-Даби
        d_genes[371] = gen(vec2(23.6142, 58.5925), 371); // Маскат
        d_genes[372] = gen(vec2(26.2235, 50.5876), 372); // Манама
        d_genes[373] = gen(vec2(29.3759, 47.9774), 373); // Эль-Кувейт
        d_genes[374] = gen(vec2(24.7136, 46.6753), 374); // Эр-Рияд
        d_genes[375] = gen(vec2(21.4858, 39.1925), 375); // Мекка
        d_genes[376] = gen(vec2(24.4672, 39.6111), 376); // Медина
        d_genes[377] = gen(vec2(30.0444, 31.2357), 377); // Каир
        d_genes[378] = gen(vec2(31.2001, 29.9187), 378); // Александрия
        d_genes[379] = gen(vec2(25.6872, 32.6396), 379); // Луксор
        d_genes[380] = gen(vec2(24.0889, 32.8998), 380); // Асуан
        d_genes[381] = gen(vec2(31.2058, 29.9249), 381); // Александрия
        d_genes[382] = gen(vec2(30.0444, 31.2357), 382); // Каир
        d_genes[383] = gen(vec2(36.8969, 30.7133), 383); // Анталья
        d_genes[384] = gen(vec2(41.0082, 28.9784), 384); // Стамбул
        d_genes[385] = gen(vec2(39.9334, 32.8597), 385); // Анкара
        d_genes[386] = gen(vec2(38.4237, 27.1428), 386); // Измир
        d_genes[387] = gen(vec2(36.9864, 35.3253), 387); // Адана
        d_genes[388] = gen(vec2(40.1885, 29.0610), 388); // Бурса
        d_genes[389] = gen(vec2(37.8720, 32.4841), 389); // Конья
        d_genes[390] = gen(vec2(41.0151, 28.9795), 390); // Стамбул
        d_genes[391] = gen(vec2(33.5138, 36.2765), 391); // Дамаск
        d_genes[392] = gen(vec2(33.8938, 35.5018), 392); // Бейрут
        d_genes[393] = gen(vec2(31.9466, 35.3027), 393); // Амман
        d_genes[394] = gen(vec2(31.7683, 35.2137), 394); // Иерусалим
        d_genes[395] = gen(vec2(32.0853, 34.7818), 395); // Тель-Авив
        d_genes[396] = gen(vec2(31.0461, 34.8516), 396); // Беэр-Шева
        d_genes[397] = gen(vec2(32.7940, 34.9896), 397); // Хайфа
        d_genes[398] = gen(vec2(29.5581, 34.9482), 398); // Эйлат
        d_genes[399] = gen(vec2(33.3100, 44.3460), 399); // Багдад
        d_genes[400] = gen(vec2(33.3152, 44.3661), 400); // Багдад
        d_genes[401] = gen(vec2(36.1911, 44.0092), 401); // Эрбиль
        d_genes[402] = gen(vec2(35.6892, 51.3890), 402); // Тегеран
        d_genes[403] = gen(vec2(36.2687, 59.5677), 403); // Мешхед
        d_genes[404] = gen(vec2(32.6546, 51.6680), 404); // Исфахан
        d_genes[405] = gen(vec2(29.5918, 52.5837), 405); // Шираз
        d_genes[406] = gen(vec2(38.0962, 46.2738), 406); // Тебриз
        d_genes[407] = gen(vec2(34.3416, 47.0861), 407); // Керманшах
        d_genes[408] = gen(vec2(35.7000, 51.4000), 408); // Кередж
        d_genes[409] = gen(vec2(30.2839, 57.0833), 409); // Керман
        d_genes[410] = gen(vec2(27.1832, 56.2666), 410); // Бендер-Аббас
        d_genes[411] = gen(vec2(25.2854, 51.5310), 411); // Доха
        d_genes[412] = gen(vec2(24.4539, 54.3773), 412); // Абу-Даби
        d_genes[413] = gen(vec2(25.2048, 55.2708), 413); // Дубай
        d_genes[414] = gen(vec2(25.0759, 55.1348), 414); // Шарджа
        d_genes[415] = gen(vec2(24.4667, 54.3667), 415); // Абу-Даби
        d_genes[416] = gen(vec2(23.6142, 58.5925), 416); // Маскат
        d_genes[417] = gen(vec2(26.2235, 50.5876), 417); // Манама
        d_genes[418] = gen(vec2(29.3759, 47.9774), 418); // Эль-Кувейт
        d_genes[419] = gen(vec2(24.7136, 46.6753), 419); // Эр-Рияд
        d_genes[420] = gen(vec2(21.4858, 39.1925), 420); // Мекка
        d_genes[421] = gen(vec2(24.4672, 39.6111), 421); // Медина
        d_genes[422] = gen(vec2(30.0444, 31.2357), 422); // Каир
        d_genes[423] = gen(vec2(31.2001, 29.9187), 423); // Александрия
        d_genes[424] = gen(vec2(25.6872, 32.6396), 424); // Луксор
        d_genes[425] = gen(vec2(24.0889, 32.8998), 425); // Асуан
        d_genes[426] = gen(vec2(31.2058, 29.9249), 426); // Александрия
        d_genes[427] = gen(vec2(30.0444, 31.2357), 427); // Каир
        d_genes[428] = gen(vec2(36.8969, 30.7133), 428); // Анталья
        d_genes[429] = gen(vec2(41.0082, 28.9784), 429); // Стамбул
        d_genes[430] = gen(vec2(39.9334, 32.8597), 430); // Анкара
        d_genes[431] = gen(vec2(38.4237, 27.1428), 431); // Измир
        d_genes[432] = gen(vec2(36.9864, 35.3253), 432); // Адана
        d_genes[433] = gen(vec2(40.1885, 29.0610), 433); // Бурса
        d_genes[434] = gen(vec2(37.8720, 32.4841), 434); // Конья
        d_genes[435] = gen(vec2(41.0151, 28.9795), 435); // Стамбул
        d_genes[436] = gen(vec2(33.5138, 36.2765), 436); // Дамаск
        d_genes[437] = gen(vec2(33.8938, 35.5018), 437); // Бейрут
        d_genes[438] = gen(vec2(31.9466, 35.3027), 438); // Амман
        d_genes[439] = gen(vec2(31.7683, 35.2137), 439); // Иерусалим
        d_genes[440] = gen(vec2(32.0853, 34.7818), 440); // Тель-Авив
        d_genes[441] = gen(vec2(31.0461, 34.8516), 441); // Беэр-Шева
        d_genes[442] = gen(vec2(32.7940, 34.9896), 442); // Хайфа
        d_genes[443] = gen(vec2(29.5581, 34.9482), 443); // Эйлат
        d_genes[444] = gen(vec2(33.3100, 44.3460), 444); // Багдад
        d_genes[445] = gen(vec2(33.3152, 44.3661), 445); // Багдад
        d_genes[446] = gen(vec2(36.1911, 44.0092), 446); // Эрбиль
        d_genes[447] = gen(vec2(35.6892, 51.3890), 447); // Тегеран
        d_genes[448] = gen(vec2(36.2687, 59.5677), 448); // Мешхед
        d_genes[449] = gen(vec2(32.6546, 51.6680), 449); // Исфахан
        d_genes[450] = gen(vec2(29.5918, 52.5837), 450); // Шираз
        d_genes[451] = gen(vec2(38.0962, 46.2738), 451); // Тебриз
        d_genes[452] = gen(vec2(34.3416, 47.0861), 452); // Керманшах
        d_genes[453] = gen(vec2(35.7000, 51.4000), 453); // Кередж
        d_genes[454] = gen(vec2(30.2839, 57.0833), 454); // Керман
        d_genes[455] = gen(vec2(27.1832, 56.2666), 455); // Бендер-Аббас
        d_genes[456] = gen(vec2(25.2854, 51.5310), 456); // Доха
        d_genes[457] = gen(vec2(24.4539, 54.3773), 457); // Абу-Даби
        d_genes[458] = gen(vec2(25.2048, 55.2708), 458); // Дубай
        d_genes[459] = gen(vec2(25.0759, 55.1348), 459); // Шарджа
        d_genes[460] = gen(vec2(24.4667, 54.3667), 460); // Абу-Даби
        d_genes[461] = gen(vec2(23.6142, 58.5925), 461); // Маскат
        d_genes[462] = gen(vec2(26.2235, 50.5876), 462); // Манама
        d_genes[463] = gen(vec2(29.3759, 47.9774), 463); // Эль-Кувейт
        d_genes[464] = gen(vec2(24.7136, 46.6753), 464); // Эр-Рияд
        d_genes[465] = gen(vec2(21.4858, 39.1925), 465); // Мекка
        d_genes[466] = gen(vec2(24.4672, 39.6111), 466); // Медина
        d_genes[467] = gen(vec2(30.0444, 31.2357), 467); // Каир
        d_genes[468] = gen(vec2(31.2001, 29.9187), 468); // Александрия
        d_genes[469] = gen(vec2(25.6872, 32.6396), 469); // Луксор
        d_genes[470] = gen(vec2(24.0889, 32.8998), 470); // Асуан
        d_genes[471] = gen(vec2(31.2058, 29.9249), 471); // Александрия
        d_genes[472] = gen(vec2(30.0444, 31.2357), 472); // Каир
        d_genes[473] = gen(vec2(36.8969, 30.7133), 473); // Анталья
        d_genes[474] = gen(vec2(41.0082, 28.9784), 474); // Стамбул
        d_genes[475] = gen(vec2(39.9334, 32.8597), 475); // Анкара
        d_genes[476] = gen(vec2(38.4237, 27.1428), 476); // Измир
        d_genes[477] = gen(vec2(36.9864, 35.3253), 477); // Адана
        d_genes[478] = gen(vec2(40.1885, 29.0610), 478); // Бурса
        d_genes[479] = gen(vec2(37.8720, 32.4841), 479); // Конья
        d_genes[480] = gen(vec2(41.0151, 28.9795), 480); // Стамбул
        d_genes[481] = gen(vec2(33.5138, 36.2765), 481); // Дамаск
        d_genes[482] = gen(vec2(33.8938, 35.5018), 482); // Бейрут
        d_genes[483] = gen(vec2(31.9466, 35.3027), 483); // Амман
        d_genes[484] = gen(vec2(31.7683, 35.2137), 484); // Иерусалим
        d_genes[485] = gen(vec2(32.0853, 34.7818), 485); // Тель-Авив
        d_genes[486] = gen(vec2(31.0461, 34.8516), 486); // Беэр-Шева
        d_genes[487] = gen(vec2(32.7940, 34.9896), 487); // Хайфа
        d_genes[488] = gen(vec2(29.5581, 34.9482), 488); // Эйлат
        d_genes[489] = gen(vec2(33.3100, 44.3460), 489); // Багдад
        d_genes[490] = gen(vec2(33.3152, 44.3661), 490); // Багдад
        d_genes[491] = gen(vec2(36.1911, 44.0092), 491); // Эрбиль
        d_genes[492] = gen(vec2(35.6892, 51.3890), 492); // Тегеран
        d_genes[493] = gen(vec2(36.2687, 59.5677), 493); // Мешхед
        d_genes[494] = gen(vec2(32.6546, 51.6680), 494); // Исфахан
        d_genes[495] = gen(vec2(29.5918, 52.5837), 495); // Шираз
        d_genes[496] = gen(vec2(38.0962, 46.2738), 496); // Тебриз
        d_genes[497] = gen(vec2(34.3416, 47.0861), 497); // Керманшах
        d_genes[498] = gen(vec2(35.7000, 51.4000), 498); // Кередж
        d_genes[499] = gen(vec2(30.2839, 57.0833), 499); // Керман
        d_genes[500] = gen(vec2(27.1832, 56.2666), 500); // Бендер-Аббас
        d_genes[501] = gen(vec2(25.2854, 51.5310), 501); // Доха
        d_genes[502] = gen(vec2(24.4539, 54.3773), 502); // Абу-Даби
        d_genes[503] = gen(vec2(25.2048, 55.2708), 503); // Дубай
        d_genes[504] = gen(vec2(25.0759, 55.1348), 504); // Шарджа
        d_genes[505] = gen(vec2(24.4667, 54.3667), 505); // Абу-Даби
        d_genes[506] = gen(vec2(23.6142, 58.5925), 506); // Маскат
        d_genes[507] = gen(vec2(26.2235, 50.5876), 507); // Манама
        d_genes[508] = gen(vec2(29.3759, 47.9774), 508); // Эль-Кувейт
        d_genes[509] = gen(vec2(24.7136, 46.6753), 509); // Эр-Рияд
        d_genes[510] = gen(vec2(21.4858, 39.1925), 510); // Мекка
        d_genes[511] = gen(vec2(24.4672, 39.6111), 511); // Медина
        d_genes[512] = gen(vec2(30.0444, 31.2357), 512); // Каир
        d_genes[513] = gen(vec2(31.2001, 29.9187), 513); // Александрия
        d_genes[514] = gen(vec2(25.6872, 32.6396), 514); // Луксор
        d_genes[515] = gen(vec2(24.0889, 32.8998), 515); // Асуан
        d_genes[516] = gen(vec2(31.2058, 29.9249), 516); // Александрия
        d_genes[517] = gen(vec2(30.0444, 31.2357), 517); // Каир
        d_genes[518] = gen(vec2(36.8969, 30.7133), 518); // Анталья
        d_genes[519] = gen(vec2(41.0082, 28.9784), 519); // Стамбул
        d_genes[520] = gen(vec2(39.9334, 32.8597), 520); // Анкара
        d_genes[521] = gen(vec2(38.4237, 27.1428), 521); // Измир
        d_genes[522] = gen(vec2(36.9864, 35.3253), 522); // Адана
        d_genes[523] = gen(vec2(40.1885, 29.0610), 523); // Бурса
        d_genes[524] = gen(vec2(37.8720, 32.4841), 524); // Конья
        d_genes[525] = gen(vec2(41.0151, 28.9795), 525); // Стамбул
        d_genes[526] = gen(vec2(33.5138, 36.2765), 526); // Дамаск
        d_genes[527] = gen(vec2(33.8938, 35.5018), 527); // Бейрут
        d_genes[528] = gen(vec2(31.9466, 35.3027), 528); // Амман
        d_genes[529] = gen(vec2(31.7683, 35.2137), 529); // Иерусалим
        d_genes[530] = gen(vec2(32.0853, 34.7818), 530); // Тель-Авив
        d_genes[531] = gen(vec2(31.0461, 34.8516), 531); // Беэр-Шева
        d_genes[532] = gen(vec2(32.7940, 34.9896), 532); // Хайфа
        d_genes[533] = gen(vec2(29.5581, 34.9482), 533); // Эйлат
        d_genes[534] = gen(vec2(33.3100, 44.3460), 534); // Багдад
        d_genes[535] = gen(vec2(33.3152, 44.3661), 535); // Багдад
        d_genes[536] = gen(vec2(36.1911, 44.0092), 536); // Эрбиль
        d_genes[537] = gen(vec2(35.6892, 51.3890), 537); // Тегеран
        d_genes[538] = gen(vec2(36.2687, 59.5677), 538); // Мешхед
        d_genes[539] = gen(vec2(32.6546, 51.6680), 539); // Исфахан
        d_genes[540] = gen(vec2(29.5918, 52.5837), 540); // Шираз
        d_genes[541] = gen(vec2(38.0962, 46.2738), 541); // Тебриз
        d_genes[542] = gen(vec2(34.3416, 47.0861), 542); // Керманшах
        d_genes[543] = gen(vec2(35.7000, 51.4000), 543); // Кередж
        d_genes[544] = gen(vec2(30.2839, 57.0833), 544); // Керман
        d_genes[545] = gen(vec2(27.1832, 56.2666), 545); // Бендер-Аббас
        d_genes[546] = gen(vec2(25.2854, 51.5310), 546); // Доха
        d_genes[547] = gen(vec2(24.4539, 54.3773), 547); // Абу-Даби
        d_genes[548] = gen(vec2(25.2048, 55.2708), 548); // Дубай
        d_genes[549] = gen(vec2(25.0759, 55.1348), 549); // Шарджа
        d_genes[550] = gen(vec2(24.4667, 54.3667), 550); // Абу-Даби
        d_genes[551] = gen(vec2(23.6142, 58.5925), 551); // Маскат
        d_genes[552] = gen(vec2(26.2235, 50.5876), 552); // Манама
        d_genes[553] = gen(vec2(29.3759, 47.9774), 553); // Эль-Кувейт
        d_genes[554] = gen(vec2(24.7136, 46.6753), 554); // Эр-Рияд
        d_genes[555] = gen(vec2(21.4858, 39.1925), 555); // Мекка
        d_genes[556] = gen(vec2(24.4672, 39.6111), 556); // Медина
        d_genes[557] = gen(vec2(30.0444, 31.2357), 557); // Каир
        d_genes[558] = gen(vec2(31.2001, 29.9187), 558); // Александрия
        d_genes[559] = gen(vec2(25.6872, 32.6396), 559); // Луксор
        d_genes[560] = gen(vec2(24.0889, 32.8998), 560); // Асуан
        d_genes[561] = gen(vec2(31.2058, 29.9249), 561); // Александрия
        d_genes[562] = gen(vec2(30.0444, 31.2357), 562); // Каир
        d_genes[563] = gen(vec2(36.8969, 30.7133), 563); // Анталья
        d_genes[564] = gen(vec2(41.0082, 28.9784), 564); // Стамбул
        d_genes[565] = gen(vec2(39.9334, 32.8597), 565); // Анкара
        d_genes[566] = gen(vec2(38.4237, 27.1428), 566); // Измир
        d_genes[567] = gen(vec2(36.9864, 35.3253), 567); // Адана
        d_genes[568] = gen(vec2(40.1885, 29.0610), 568); // Бурса
        d_genes[569] = gen(vec2(37.8720, 32.4841), 569); // Конья
        d_genes[570] = gen(vec2(41.0151, 28.9795), 570); // Стамбул
        d_genes[571] = gen(vec2(33.5138, 36.2765), 571); // Дамаск
        d_genes[572] = gen(vec2(33.8938, 35.5018), 572); // Бейрут
        d_genes[573] = gen(vec2(31.9466, 35.3027), 573); // Амман
        d_genes[574] = gen(vec2(31.7683, 35.2137), 574); // Иерусалим
        d_genes[575] = gen(vec2(32.0853, 34.7818), 575); // Тель-Авив
        d_genes[576] = gen(vec2(31.0461, 34.8516), 576); // Беэр-Шева
        d_genes[577] = gen(vec2(32.7940, 34.9896), 577); // Хайфа
        d_genes[578] = gen(vec2(29.5581, 34.9482), 578); // Эйлат
        d_genes[579] = gen(vec2(33.3100, 44.3460), 579); // Багдад
        d_genes[580] = gen(vec2(33.3152, 44.3661), 580); // Багдад
        d_genes[581] = gen(vec2(36.1911, 44.0092), 581); // Эрбиль
        d_genes[582] = gen(vec2(35.6892, 51.3890), 582); // Тегеран
        d_genes[583] = gen(vec2(36.2687, 59.5677), 583); // Мешхед
        d_genes[584] = gen(vec2(32.6546, 51.6680), 584); // Исфахан
        d_genes[585] = gen(vec2(29.5918, 52.5837), 585); // Шираз
        d_genes[586] = gen(vec2(38.0962, 46.2738), 586); // Тебриз
        d_genes[587] = gen(vec2(34.3416, 47.0861), 587); // Керманшах
        d_genes[588] = gen(vec2(35.7000, 51.4000), 588); // Кередж
        d_genes[589] = gen(vec2(30.2839, 57.0833), 589); // Керман
        d_genes[590] = gen(vec2(27.1832, 56.2666), 590); // Бендер-Аббас
        d_genes[591] = gen(vec2(25.2854, 51.5310), 591); // Доха
        d_genes[592] = gen(vec2(24.4539, 54.3773), 592); // Абу-Даби
        d_genes[593] = gen(vec2(25.2048, 55.2708), 593); // Дубай
        d_genes[594] = gen(vec2(25.0759, 55.1348), 594); // Шарджа
        d_genes[595] = gen(vec2(24.4667, 54.3667), 595); // Абу-Даби
        d_genes[596] = gen(vec2(23.6142, 58.5925), 596); // Маскат
        d_genes[597] = gen(vec2(26.2235, 50.5876), 597); // Манама
        d_genes[598] = gen(vec2(29.3759, 47.9774), 598); // Эль-Кувейт
        d_genes[599] = gen(vec2(24.7136, 46.6753), 599); // Эр-Рияд
        d_genes[600] = gen(vec2(21.4858, 39.1925), 600); // Мекка
        d_genes[601] = gen(vec2(24.4672, 39.6111), 601); // Медина
        d_genes[602] = gen(vec2(30.0444, 31.2357), 602); // Каир
        d_genes[603] = gen(vec2(31.2001, 29.9187), 603); // Александрия
        d_genes[604] = gen(vec2(25.6872, 32.6396), 604); // Луксор
        d_genes[605] = gen(vec2(24.0889, 32.8998), 605); // Асуан
        d_genes[606] = gen(vec2(31.2058, 29.9249), 606); // Александрия
        d_genes[607] = gen(vec2(30.0444, 31.2357), 607); // Каир
        d_genes[608] = gen(vec2(36.8969, 30.7133), 608); // Анталья
        d_genes[609] = gen(vec2(41.0082, 28.9784), 609); // Стамбул
        d_genes[610] = gen(vec2(39.9334, 32.8597), 610); // Анкара
        d_genes[611] = gen(vec2(38.4237, 27.1428), 611); // Измир
        d_genes[612] = gen(vec2(36.9864, 35.3253), 612); // Адана
        d_genes[613] = gen(vec2(40.1885, 29.0610), 613); // Бурса
        d_genes[614] = gen(vec2(37.8720, 32.4841), 614); // Конья
        d_genes[615] = gen(vec2(41.0151, 28.9795), 615); // Стамбул
        d_genes[616] = gen(vec2(33.5138, 36.2765), 616); // Дамаск
        d_genes[617] = gen(vec2(33.8938, 35.5018), 617); // Бейрут
        d_genes[618] = gen(vec2(31.9466, 35.3027), 618); // Амман
        d_genes[619] = gen(vec2(31.7683, 35.2137), 619); // Иерусалим
        d_genes[620] = gen(vec2(32.0853, 34.7818), 620); // Тель-Авив
        d_genes[621] = gen(vec2(31.0461, 34.8516), 621); // Беэр-Шева
        d_genes[622] = gen(vec2(32.7940, 34.9896), 622); // Хайфа
        d_genes[623] = gen(vec2(29.5581, 34.9482), 623); // Эйлат
        d_genes[624] = gen(vec2(33.3100, 44.3460), 624); // Багдад
        d_genes[625] = gen(vec2(33.3152, 44.3661), 625); // Багдад
        d_genes[626] = gen(vec2(36.1911, 44.0092), 626); // Эрбиль
        d_genes[627] = gen(vec2(35.6892, 51.3890), 627); // Тегеран
        d_genes[628] = gen(vec2(36.2687, 59.5677), 628); // Мешхед
        d_genes[629] = gen(vec2(32.6546, 51.6680), 629); // Исфахан
        d_genes[630] = gen(vec2(29.5918, 52.5837), 630); // Шираз
        d_genes[631] = gen(vec2(38.0962, 46.2738), 631); // Тебриз
        d_genes[632] = gen(vec2(34.3416, 47.0861), 632); // Керманшах
        d_genes[633] = gen(vec2(35.7000, 51.4000), 633); // Кередж
        d_genes[634] = gen(vec2(30.2839, 57.0833), 634); // Керман
        d_genes[635] = gen(vec2(27.1832, 56.2666), 635); // Бендер-Аббас
        d_genes[636] = gen(vec2(25.2854, 51.5310), 636); // Доха
        d_genes[637] = gen(vec2(24.4539, 54.3773), 637); // Абу-Даби
        d_genes[638] = gen(vec2(25.2048, 55.2708), 638); // Дубай
        d_genes[639] = gen(vec2(25.0759, 55.1348), 639); // Шарджа
        d_genes[640] = gen(vec2(24.4667, 54.3667), 640); // Абу-Даби
        d_genes[641] = gen(vec2(23.6142, 58.5925), 641); // Маскат
        d_genes[642] = gen(vec2(26.2235, 50.5876), 642); // Манама
        d_genes[643] = gen(vec2(29.3759, 47.9774), 643); // Эль-Кувейт
        d_genes[644] = gen(vec2(24.7136, 46.6753), 644); // Эр-Рияд
        d_genes[645] = gen(vec2(21.4858, 39.1925), 645); // Мекка
        d_genes[646] = gen(vec2(24.4672, 39.6111), 646); // Медина
        d_genes[647] = gen(vec2(30.0444, 31.2357), 647); // Каир
        d_genes[648] = gen(vec2(31.2001, 29.9187), 648); // Александрия
        d_genes[649] = gen(vec2(25.6872, 32.6396), 649); // Луксор
        d_genes[650] = gen(vec2(24.0889, 32.8998), 650); // Асуан
        d_genes[651] = gen(vec2(31.2058, 29.9249), 651); // Александрия
        d_genes[652] = gen(vec2(30.0444, 31.2357), 652); // Каир
        d_genes[653] = gen(vec2(36.8969, 30.7133), 653); // Анталья
        d_genes[654] = gen(vec2(41.0082, 28.9784), 654); // Стамбул
        d_genes[655] = gen(vec2(39.9334, 32.8597), 655); // Анкара
        d_genes[656] = gen(vec2(38.4237, 27.1428), 656); // Измир
        d_genes[657] = gen(vec2(36.9864, 35.3253), 657); // Адана
        d_genes[658] = gen(vec2(40.1885, 29.0610), 658); // Бурса
        d_genes[659] = gen(vec2(37.8720, 32.4841), 659); // Конья
        d_genes[660] = gen(vec2(41.0151, 28.9795), 660); // Стамбул
        d_genes[661] = gen(vec2(33.5138, 36.2765), 661); // Дамаск
        d_genes[662] = gen(vec2(33.8938, 35.5018), 662); // Бейрут
        d_genes[663] = gen(vec2(31.9466, 35.3027), 663); // Амман
        d_genes[664] = gen(vec2(31.7683, 35.2137), 664); // Иерусалим
        d_genes[665] = gen(vec2(32.0853, 34.7818), 665); // Тель-Авив
        d_genes[666] = gen(vec2(31.0461, 34.8516), 666); // Беэр-Шева
        d_genes[667] = gen(vec2(32.7940, 34.9896), 667); // Хайфа
        d_genes[668] = gen(vec2(29.5581, 34.9482), 668); // Эйлат
        d_genes[669] = gen(vec2(33.3100, 44.3460), 669); // Багдад
        d_genes[670] = gen(vec2(33.3152, 44.3661), 670); // Багдад
        d_genes[671] = gen(vec2(36.1911, 44.0092), 671); // Эрбиль
        d_genes[672] = gen(vec2(35.6892, 51.3890), 672); // Тегеран
        d_genes[673] = gen(vec2(36.2687, 59.5677), 673); // Мешхед
        d_genes[674] = gen(vec2(32.6546, 51.6680), 674); // Исфахан
        d_genes[675] = gen(vec2(29.5918, 52.5837), 675); // Шираз
        d_genes[676] = gen(vec2(38.0962, 46.2738), 676); // Тебриз
        d_genes[677] = gen(vec2(34.3416, 47.0861), 677); // Керманшах
        d_genes[678] = gen(vec2(35.7000, 51.4000), 678); // Кередж
        d_genes[679] = gen(vec2(30.2839, 57.0833), 679); // Керман
        d_genes[680] = gen(vec2(27.1832, 56.2666), 680); // Бендер-Аббас
        d_genes[681] = gen(vec2(25.2854, 51.5310), 681); // Доха
        d_genes[682] = gen(vec2(24.4539, 54.3773), 682); // Абу-Даби
        d_genes[683] = gen(vec2(25.2048, 55.2708), 683); // Дубай
        d_genes[684] = gen(vec2(25.0759, 55.1348), 684); // Шарджа
        d_genes[685] = gen(vec2(24.4667, 54.3667), 685); // Абу-Даби
        d_genes[686] = gen(vec2(23.6142, 58.5925), 686); // Маскат
        d_genes[687] = gen(vec2(26.2235, 50.5876), 687); // Манама
        d_genes[688] = gen(vec2(29.3759, 47.9774), 688); // Эль-Кувейт
        d_genes[689] = gen(vec2(24.7136, 46.6753), 689); // Эр-Рияд
        d_genes[690] = gen(vec2(21.4858, 39.1925), 690); // Мекка
        d_genes[691] = gen(vec2(24.4672, 39.6111), 691); // Медина
        d_genes[692] = gen(vec2(30.0444, 31.2357), 692); // Каир
        d_genes[693] = gen(vec2(31.2001, 29.9187), 693); // Александрия
        d_genes[694] = gen(vec2(25.6872, 32.6396), 694); // Луксор
        d_genes[695] = gen(vec2(24.0889, 32.8998), 695); // Асуан
        d_genes[696] = gen(vec2(31.2058, 29.9249), 696); // Александрия
        d_genes[697] = gen(vec2(30.0444, 31.2357), 697); // Каир
        d_genes[698] = gen(vec2(36.8969, 30.7133), 698); // Анталья
        d_genes[699] = gen(vec2(41.0082, 28.9784), 699); // Стамбул
        d_genes[700] = gen(vec2(39.9334, 32.8597), 700); // Анкара
        d_genes[701] = gen(vec2(38.4237, 27.1428), 701); // Измир
        d_genes[702] = gen(vec2(36.9864, 35.3253), 702); // Адана
        d_genes[703] = gen(vec2(40.1885, 29.0610), 703); // Бурса
        d_genes[704] = gen(vec2(37.8720, 32.4841), 704); // Конья
        d_genes[705] = gen(vec2(41.0151, 28.9795), 705); // Стамбул
        d_genes[706] = gen(vec2(33.5138, 36.2765), 706); // Дамаск
        d_genes[707] = gen(vec2(33.8938, 35.5018), 707); // Бейрут
        d_genes[708] = gen(vec2(31.9466, 35.3027), 708); // Амман
        d_genes[709] = gen(vec2(31.7683, 35.2137), 709); // Иерусалим
        d_genes[710] = gen(vec2(32.0853, 34.7818), 710); // Тель-Авив
        d_genes[711] = gen(vec2(31.0461, 34.8516), 711); // Беэр-Шева
        d_genes[712] = gen(vec2(32.7940, 34.9896), 712); // Хайфа
        d_genes[713] = gen(vec2(29.5581, 34.9482), 713); // Эйлат
        d_genes[714] = gen(vec2(33.3100, 44.3460), 714); // Багдад
        d_genes[715] = gen(vec2(33.3152, 44.3661), 715); // Багдад
        d_genes[716] = gen(vec2(36.1911, 44.0092), 716); // Эрбиль
        d_genes[717] = gen(vec2(35.6892, 51.3890), 717); // Тегеран
        d_genes[718] = gen(vec2(36.2687, 59.5677), 718); // Мешхед
        d_genes[719] = gen(vec2(32.6546, 51.6680), 719); // Исфахан
        d_genes[720] = gen(vec2(29.5918, 52.5837), 720); // Шираз
        d_genes[721] = gen(vec2(38.0962, 46.2738), 721); // Тебриз
        d_genes[722] = gen(vec2(34.3416, 47.0861), 722); // Керманшах
        d_genes[723] = gen(vec2(35.7000, 51.4000), 723); // Кередж
        d_genes[724] = gen(vec2(30.2839, 57.0833), 724); // Керман
        d_genes[725] = gen(vec2(27.1832, 56.2666), 725); // Бендер-Аббас
        d_genes[726] = gen(vec2(25.2854, 51.5310), 726); // Доха
        d_genes[727] = gen(vec2(24.4539, 54.3773), 727); // Абу-Даби
        d_genes[728] = gen(vec2(25.2048, 55.2708), 728); // Дубай
        d_genes[729] = gen(vec2(25.0759, 55.1348), 729); // Шарджа
        d_genes[730] = gen(vec2(24.4667, 54.3667), 730); // Абу-Даби
        d_genes[731] = gen(vec2(23.6142, 58.5925), 731); // Маскат
        d_genes[732] = gen(vec2(26.2235, 50.5876), 732); // Манама
        d_genes[733] = gen(vec2(29.3759, 47.9774), 733); // Эль-Кувейт
        d_genes[734] = gen(vec2(24.7136, 46.6753), 734); // Эр-Рияд
        d_genes[735] = gen(vec2(21.4858, 39.1925), 735); // Мекка
        d_genes[736] = gen(vec2(24.4672, 39.6111), 736); // Медина
        d_genes[737] = gen(vec2(30.0444, 31.2357), 737); // Каир
        d_genes[738] = gen(vec2(31.2001, 29.9187), 738); // Александрия
        d_genes[739] = gen(vec2(25.6872, 32.6396), 739); // Луксор
        d_genes[740] = gen(vec2(24.0889, 32.8998), 740); // Асуан
        d_genes[741] = gen(vec2(31.2058, 29.9249), 741); // Александрия
        d_genes[742] = gen(vec2(30.0444, 31.2357), 742); // Каир
        d_genes[743] = gen(vec2(36.8969, 30.7133), 743); // Анталья
        d_genes[744] = gen(vec2(41.0082, 28.9784), 744); // Стамбул
        d_genes[745] = gen(vec2(39.9334, 32.8597), 745); // Анкара
        d_genes[746] = gen(vec2(38.4237, 27.1428), 746); // Измир
        d_genes[747] = gen(vec2(36.9864, 35.3253), 747); // Адана
        d_genes[748] = gen(vec2(40.1885, 29.0610), 748); // Бурса
        d_genes[749] = gen(vec2(37.8720, 32.4841), 749); // Конья
        d_genes[750] = gen(vec2(41.0151, 28.9795), 750); // Стамбул
        d_genes[751] = gen(vec2(33.5138, 36.2765), 751); // Дамаск
        d_genes[752] = gen(vec2(33.8938, 35.5018), 752); // Бейрут
        d_genes[753] = gen(vec2(31.9466, 35.3027), 753); // Амман
        d_genes[754] = gen(vec2(31.7683, 35.2137), 754); // Иерусалим
        d_genes[755] = gen(vec2(32.0853, 34.7818), 755); // Тель-Авив
        d_genes[756] = gen(vec2(31.0461, 34.8516), 756); // Беэр-Шева
        d_genes[757] = gen(vec2(32.7940, 34.9896), 757); // Хайфа
        d_genes[758] = gen(vec2(29.5581, 34.9482), 758); // Эйлат
        d_genes[759] = gen(vec2(33.3100, 44.3460), 759); // Багдад
        d_genes[760] = gen(vec2(33.3152, 44.3661), 760); // Багдад
        d_genes[761] = gen(vec2(36.1911, 44.0092), 761); // Эрбиль
        d_genes[762] = gen(vec2(35.6892, 51.3890), 762); // Тегеран
        d_genes[763] = gen(vec2(36.2687, 59.5677), 763); // Мешхед
        d_genes[764] = gen(vec2(32.6546, 51.6680), 764); // Исфахан
        d_genes[765] = gen(vec2(29.5918, 52.5837), 765); // Шираз
        d_genes[766] = gen(vec2(38.0962, 46.2738), 766); // Тебриз
        d_genes[767] = gen(vec2(34.3416, 47.0861), 767); // Керманшах
        d_genes[768] = gen(vec2(35.7000, 51.4000), 768); // Кередж
        d_genes[769] = gen(vec2(30.2839, 57.0833), 769); // Керман
        d_genes[770] = gen(vec2(27.1832, 56.2666), 770); // Бендер-Аббас
        d_genes[771] = gen(vec2(25.2854, 51.5310), 771); // Доха
        d_genes[772] = gen(vec2(24.4539, 54.3773), 772); // Абу-Даби
        d_genes[773] = gen(vec2(25.2048, 55.2708), 773); // Дубай
        d_genes[774] = gen(vec2(25.0759, 55.1348), 774); // Шарджа
        d_genes[775] = gen(vec2(24.4667, 54.3667), 775); // Абу-Даби
        d_genes[776] = gen(vec2(23.6142, 58.5925), 776); // Маскат
        d_genes[777] = gen(vec2(26.2235, 50.5876), 777); // Манама
        d_genes[778] = gen(vec2(29.3759, 47.9774), 778); // Эль-Кувейт
        d_genes[779] = gen(vec2(24.7136, 46.6753), 779); // Эр-Рияд
        d_genes[780] = gen(vec2(21.4858, 39.1925), 780); // Мекка
        d_genes[781] = gen(vec2(24.4672, 39.6111), 781); // Медина
        d_genes[782] = gen(vec2(30.0444, 31.2357), 782); // Каир
        d_genes[783] = gen(vec2(31.2001, 29.9187), 783); // Александрия
        d_genes[784] = gen(vec2(25.6872, 32.6396), 784); // Луксор
        d_genes[785] = gen(vec2(24.0889, 32.8998), 785); // Асуан
        d_genes[786] = gen(vec2(31.2058, 29.9249), 786); // Александрия
        d_genes[787] = gen(vec2(30.0444, 31.2357), 787); // Каир
        d_genes[788] = gen(vec2(36.8969, 30.7133), 788); // Анталья
        d_genes[789] = gen(vec2(41.0082, 28.9784), 789); // Стамбул
        d_genes[790] = gen(vec2(39.9334, 32.8597), 790); // Анкара
        d_genes[791] = gen(vec2(38.4237, 27.1428), 791); // Измир
        d_genes[792] = gen(vec2(36.9864, 35.3253), 792); // Адана
        d_genes[793] = gen(vec2(40.1885, 29.0610), 793); // Бурса
        d_genes[794] = gen(vec2(37.8720, 32.4841), 794); // Конья
        d_genes[795] = gen(vec2(41.0151, 28.9795), 795); // Стамбул
        d_genes[796] = gen(vec2(33.5138, 36.2765), 796); // Дамаск
        d_genes[797] = gen(vec2(33.8938, 35.5018), 797); // Бейрут
        d_genes[798] = gen(vec2(31.9466, 35.3027), 798); // Амман
        d_genes[799] = gen(vec2(31.7683, 35.2137), 799); // Иерусалим
        d_genes[800] = gen(vec2(32.0853, 34.7818), 800); // Тель-Авив
        d_genes[801] = gen(vec2(31.0461, 34.8516), 801); // Беэр-Шева
        d_genes[802] = gen(vec2(32.7940, 34.9896), 802); // Хайфа
        d_genes[803] = gen(vec2(29.5581, 34.9482), 803); // Эйлат
        d_genes[804] = gen(vec2(33.3100, 44.3460), 804); // Багдад
        d_genes[805] = gen(vec2(33.3152, 44.3661), 805); // Багдад
        d_genes[806] = gen(vec2(36.1911, 44.0092), 806); // Эрбиль
        d_genes[807] = gen(vec2(35.6892, 51.3890), 807); // Тегеран
        d_genes[808] = gen(vec2(36.2687, 59.5677), 808); // Мешхед
        d_genes[809] = gen(vec2(32.6546, 51.6680), 809); // Исфахан
        d_genes[810] = gen(vec2(29.5918, 52.5837), 810); // Шираз
        d_genes[811] = gen(vec2(38.0962, 46.2738), 811); // Тебриз
        d_genes[812] = gen(vec2(34.3416, 47.0861), 812); // Керманшах
        d_genes[813] = gen(vec2(35.7000, 51.4000), 813); // Кередж
        d_genes[814] = gen(vec2(30.2839, 57.0833), 814); // Керман
        d_genes[815] = gen(vec2(27.1832, 56.2666), 815); // Бендер-Аббас
        d_genes[816] = gen(vec2(25.2854, 51.5310), 816); // Доха
        d_genes[817] = gen(vec2(24.4539, 54.3773), 817); // Абу-Даби
        d_genes[818] = gen(vec2(25.2048, 55.2708), 818); // Дубай
        d_genes[819] = gen(vec2(25.0759, 55.1348), 819); // Шарджа
        d_genes[820] = gen(vec2(24.4667, 54.3667), 820); // Абу-Даби
        d_genes[821] = gen(vec2(23.6142, 58.5925), 821); // Маскат
        d_genes[822] = gen(vec2(26.2235, 50.5876), 822); // Манама
        d_genes[823] = gen(vec2(29.3759, 47.9774), 823); // Эль-Кувейт
        d_genes[824] = gen(vec2(24.7136, 46.6753), 824); // Эр-Рияд
        d_genes[825] = gen(vec2(21.4858, 39.1925), 825); // Мекка
        d_genes[826] = gen(vec2(24.4672, 39.6111), 826); // Медина
        d_genes[827] = gen(vec2(30.0444, 31.2357), 827); // Каир
        d_genes[828] = gen(vec2(31.2001, 29.9187), 828); // Александрия
        d_genes[829] = gen(vec2(25.6872, 32.6396), 829); // Луксор
        d_genes[830] = gen(vec2(24.0889, 32.8998), 830); // Асуан
        d_genes[831] = gen(vec2(31.2058, 29.9249), 831); // Александрия
        d_genes[832] = gen(vec2(30.0444, 31.2357), 832); // Каир
        d_genes[833] = gen(vec2(36.8969, 30.7133), 833); // Анталья
        d_genes[834] = gen(vec2(41.0082, 28.9784), 834); // Стамбул
        d_genes[835] = gen(vec2(39.9334, 32.8597), 835); // Анкара
        d_genes[836] = gen(vec2(38.4237, 27.1428), 836); // Измир
        d_genes[837] = gen(vec2(36.9864, 35.3253), 837); // Адана
        d_genes[838] = gen(vec2(40.1885, 29.0610), 838); // Бурса
        d_genes[839] = gen(vec2(37.8720, 32.4841), 839); // Конья
        d_genes[840] = gen(vec2(41.0151, 28.9795), 840); // Стамбул
        d_genes[841] = gen(vec2(33.5138, 36.2765), 841); // Дамаск
        d_genes[842] = gen(vec2(33.8938, 35.5018), 842); // Бейрут
        d_genes[843] = gen(vec2(31.9466, 35.3027), 843); // Амман
        d_genes[844] = gen(vec2(31.7683, 35.2137), 844); // Иерусалим
        d_genes[845] = gen(vec2(32.0853, 34.7818), 845); // Тель-Авив
        d_genes[846] = gen(vec2(31.0461, 34.8516), 846); // Беэр-Шева
        d_genes[847] = gen(vec2(32.7940, 34.9896), 847); // Хайфа
        d_genes[848] = gen(vec2(29.5581, 34.9482), 848); // Эйлат
        d_genes[849] = gen(vec2(33.3100, 44.3460), 849); // Багдад
        d_genes[850] = gen(vec2(33.3152, 44.3661), 850); // Багдад
        d_genes[851] = gen(vec2(36.1911, 44.0092), 851); // Эрбиль
        d_genes[852] = gen(vec2(35.6892, 51.3890), 852); // Тегеран
        d_genes[853] = gen(vec2(36.2687, 59.5677), 853); // Мешхед
        d_genes[854] = gen(vec2(32.6546, 51.6680), 854); // Исфахан
        d_genes[855] = gen(vec2(29.5918, 52.5837), 855); // Шираз
        d_genes[856] = gen(vec2(38.0962, 46.2738), 856); // Тебриз
        d_genes[857] = gen(vec2(34.3416, 47.0861), 857); // Керманшах
        d_genes[858] = gen(vec2(35.7000, 51.4000), 858); // Кередж
        d_genes[859] = gen(vec2(30.2839, 57.0833), 859); // Керман
        d_genes[860] = gen(vec2(27.1832, 56.2666), 860); // Бендер-Аббас
        d_genes[861] = gen(vec2(25.2854, 51.5310), 861); // Доха
        d_genes[862] = gen(vec2(24.4539, 54.3773), 862); // Абу-Даби
        d_genes[863] = gen(vec2(25.2048, 55.2708), 863); // Дубай
        d_genes[864] = gen(vec2(25.0759, 55.1348), 864); // Шарджа
        d_genes[865] = gen(vec2(24.4667, 54.3667), 865); // Абу-Даби
        d_genes[866] = gen(vec2(23.6142, 58.5925), 866); // Маскат
        d_genes[867] = gen(vec2(26.2235, 50.5876), 867); // Манама
        d_genes[868] = gen(vec2(29.3759, 47.9774), 868); // Эль-Кувейт
        d_genes[869] = gen(vec2(24.7136, 46.6753), 869); // Эр-Рияд
        d_genes[870] = gen(vec2(21.4858, 39.1925), 870); // Мекка
        d_genes[871] = gen(vec2(24.4672, 39.6111), 871); // Медина
        d_genes[872] = gen(vec2(30.0444, 31.2357), 872); // Каир
        d_genes[873] = gen(vec2(31.2001, 29.9187), 873); // Александрия
        d_genes[874] = gen(vec2(25.6872, 32.6396), 874); // Луксор
        d_genes[875] = gen(vec2(24.0889, 32.8998), 875); // Асуан
        d_genes[876] = gen(vec2(31.2058, 29.9249), 876); // Александрия
        d_genes[877] = gen(vec2(30.0444, 31.2357), 877); // Каир
        d_genes[878] = gen(vec2(36.8969, 30.7133), 878); // Анталья
        d_genes[879] = gen(vec2(41.0082, 28.9784), 879); // Стамбул
        d_genes[880] = gen(vec2(39.9334, 32.8597), 880); // Анкара
        d_genes[881] = gen(vec2(38.4237, 27.1428), 881); // Измир
        d_genes[882] = gen(vec2(36.9864, 35.3253), 882); // Адана
        d_genes[883] = gen(vec2(40.1885, 29.0610), 883); // Бурса
        d_genes[884] = gen(vec2(37.8720, 32.4841), 884); // Конья
        d_genes[885] = gen(vec2(41.0151, 28.9795), 885); // Стамбул
        d_genes[886] = gen(vec2(33.5138, 36.2765), 886); // Дамаск
        d_genes[887] = gen(vec2(33.8938, 35.5018), 887); // Бейрут
        d_genes[888] = gen(vec2(31.9466, 35.3027), 888); // Амман
        d_genes[889] = gen(vec2(31.7683, 35.2137), 889); // Иерусалим
        d_genes[890] = gen(vec2(32.0853, 34.7818), 890); // Тель-Авив
        d_genes[891] = gen(vec2(31.0461, 34.8516), 891); // Беэр-Шева
        d_genes[892] = gen(vec2(32.7940, 34.9896), 892); // Хайфа
        d_genes[893] = gen(vec2(29.5581, 34.9482), 893); // Эйлат
        d_genes[894] = gen(vec2(33.3100, 44.3460), 894); // Багдад
        d_genes[895] = gen(vec2(33.3152, 44.3661), 895); // Багдад
        d_genes[896] = gen(vec2(36.1911, 44.0092), 896); // Эрбиль
        d_genes[897] = gen(vec2(35.6892, 51.3890), 897); // Тегеран
        d_genes[898] = gen(vec2(36.2687, 59.5677), 898); // Мешхед
        d_genes[899] = gen(vec2(32.6546, 51.6680), 899); // Исфахан
        d_genes[900] = gen(vec2(29.5918, 52.5837), 900); // Шираз
        d_genes[901] = gen(vec2(38.0962, 46.2738), 901); // Тебриз
        d_genes[902] = gen(vec2(34.3416, 47.0861), 902); // Керманшах
        d_genes[903] = gen(vec2(35.7000, 51.4000), 903); // Кередж
        d_genes[904] = gen(vec2(30.2839, 57.0833), 904); // Керман
        d_genes[905] = gen(vec2(27.1832, 56.2666), 905); // Бендер-Аббас
        d_genes[906] = gen(vec2(25.2854, 51.5310), 906); // Доха
        d_genes[907] = gen(vec2(24.4539, 54.3773), 907); // Абу-Даби
        d_genes[908] = gen(vec2(25.2048, 55.2708), 908); // Дубай
        d_genes[909] = gen(vec2(25.0759, 55.1348), 909); // Шарджа
        d_genes[910] = gen(vec2(24.4667, 54.3667), 910); // Абу-Даби
        d_genes[911] = gen(vec2(23.6142, 58.5925), 911); // Маскат
        d_genes[912] = gen(vec2(26.2235, 50.5876), 912); // Манама
        d_genes[913] = gen(vec2(29.3759, 47.9774), 913); // Эль-Кувейт
        d_genes[914] = gen(vec2(24.7136, 46.6753), 914); // Эр-Рияд
        d_genes[915] = gen(vec2(21.4858, 39.1925), 915); // Мекка
        d_genes[916] = gen(vec2(24.4672, 39.6111), 916); // Медина
        d_genes[917] = gen(vec2(30.0444, 31.2357), 917); // Каир
        d_genes[918] = gen(vec2(31.2001, 29.9187), 918); // Александрия
        d_genes[919] = gen(vec2(25.6872, 32.6396), 919); // Луксор
        d_genes[920] = gen(vec2(24.0889, 32.8998), 920); // Асуан
        d_genes[921] = gen(vec2(31.2058, 29.9249), 921); // Александрия
        d_genes[922] = gen(vec2(30.0444, 31.2357), 922); // Каир
        d_genes[923] = gen(vec2(36.8969, 30.7133), 923); // Анталья
        d_genes[924] = gen(vec2(41.0082, 28.9784), 924); // Стамбул
        d_genes[925] = gen(vec2(39.9334, 32.8597), 925); // Анкара
        d_genes[926] = gen(vec2(38.4237, 27.1428), 926); // Измир
        d_genes[927] = gen(vec2(36.9864, 35.3253), 927); // Адана
        d_genes[928] = gen(vec2(40.1885, 29.0610), 928); // Бурса
        d_genes[929] = gen(vec2(37.8720, 32.4841), 929); // Конья
        d_genes[930] = gen(vec2(41.0151, 28.9795), 930); // Стамбул
        d_genes[931] = gen(vec2(33.5138, 36.2765), 931); // Дамаск
        d_genes[932] = gen(vec2(33.8938, 35.5018), 932); // Бейрут
        d_genes[933] = gen(vec2(31.9466, 35.3027), 933); // Амман
        d_genes[934] = gen(vec2(31.7683, 35.2137), 934); // Иерусалим
        d_genes[935] = gen(vec2(32.0853, 34.7818), 935); // Тель-Авив
        d_genes[936] = gen(vec2(31.0461, 34.8516), 936); // Беэр-Шева
        d_genes[937] = gen(vec2(32.7940, 34.9896), 937); // Хайфа
        d_genes[938] = gen(vec2(29.5581, 34.9482), 938); // Эйлат
        d_genes[939] = gen(vec2(33.3100, 44.3460), 939); // Багдад
        d_genes[940] = gen(vec2(33.3152, 44.3661), 940); // Багдад
        d_genes[941] = gen(vec2(36.1911, 44.0092), 941); // Эрбиль
        d_genes[942] = gen(vec2(35.6892, 51.3890), 942); // Тегеран
        d_genes[943] = gen(vec2(36.2687, 59.5677), 943); // Мешхед
        d_genes[944] = gen(vec2(32.6546, 51.6680), 944); // Исфахан
        d_genes[945] = gen(vec2(29.5918, 52.5837), 945); // Шираз
        d_genes[946] = gen(vec2(38.0962, 46.2738), 946); // Тебриз
        d_genes[947] = gen(vec2(34.3416, 47.0861), 947); // Керманшах
        d_genes[948] = gen(vec2(35.7000, 51.4000), 948); // Кередж
        d_genes[949] = gen(vec2(30.2839, 57.0833), 949); // Керман
        d_genes[950] = gen(vec2(27.1832, 56.2666), 950); // Бендер-Аббас
        d_genes[951] = gen(vec2(25.2854, 51.5310), 951); // Доха
        d_genes[952] = gen(vec2(24.4539, 54.3773), 952); // Абу-Даби
        d_genes[953] = gen(vec2(25.2048, 55.2708), 953); // Дубай
        d_genes[954] = gen(vec2(25.0759, 55.1348), 954); // Шарджа
        d_genes[955] = gen(vec2(24.4667, 54.3667), 955); // Абу-Даби
        d_genes[956] = gen(vec2(23.6142, 58.5925), 956); // Маскат
        d_genes[957] = gen(vec2(26.2235, 50.5876), 957); // Манама
        d_genes[958] = gen(vec2(29.3759, 47.9774), 958); // Эль-Кувейт
        d_genes[959] = gen(vec2(24.7136, 46.6753), 959); // Эр-Рияд
        d_genes[960] = gen(vec2(21.4858, 39.1925), 960); // Мекка
        d_genes[961] = gen(vec2(24.4672, 39.6111), 961); // Медина
        d_genes[962] = gen(vec2(30.0444, 31.2357), 962); // Каир
        d_genes[963] = gen(vec2(31.2001, 29.9187), 963); // Александрия
        d_genes[964] = gen(vec2(25.6872, 32.6396), 964); // Луксор
        d_genes[965] = gen(vec2(24.0889, 32.8998), 965); // Асуан
        d_genes[966] = gen(vec2(31.2058, 29.9249), 966); // Александрия
        d_genes[967] = gen(vec2(30.0444, 31.2357), 967); // Каир
        d_genes[968] = gen(vec2(36.8969, 30.7133), 968); // Анталья
        d_genes[969] = gen(vec2(41.0082, 28.9784), 969); // Стамбул
        d_genes[970] = gen(vec2(39.9334, 32.8597), 970); // Анкара
        d_genes[971] = gen(vec2(38.4237, 27.1428), 971); // Измир
        d_genes[972] = gen(vec2(36.9864, 35.3253), 972); // Адана
        d_genes[973] = gen(vec2(40.1885, 29.0610), 973); // Бурса
        d_genes[974] = gen(vec2(37.8720, 32.4841), 974); // Конья
        d_genes[975] = gen(vec2(41.0151, 28.9795), 975); // Стамбул
        d_genes[976] = gen(vec2(33.5138, 36.2765), 976); // Дамаск
        d_genes[977] = gen(vec2(33.8938, 35.5018), 977); // Бейрут
        d_genes[978] = gen(vec2(31.9466, 35.3027), 978); // Амман
        d_genes[979] = gen(vec2(31.7683, 35.2137), 979); // Иерусалим
        d_genes[980] = gen(vec2(32.0853, 34.7818), 980); // Тель-Авив
        d_genes[981] = gen(vec2(31.0461, 34.8516), 981); // Беэр-Шева
        d_genes[982] = gen(vec2(32.7940, 34.9896), 982); // Хайфа
        d_genes[983] = gen(vec2(29.5581, 34.9482), 983); // Эйлат
        d_genes[984] = gen(vec2(33.3100, 44.3460), 984); // Багдад
        d_genes[985] = gen(vec2(33.3152, 44.3661), 985); // Багдад
        d_genes[986] = gen(vec2(36.1911, 44.0092), 986); // Эрбиль
        d_genes[987] = gen(vec2(35.6892, 51.3890), 987); // Тегеран
        d_genes[988] = gen(vec2(36.2687, 59.5677), 988); // Мешхед
        d_genes[989] = gen(vec2(32.6546, 51.6680), 989); // Исфахан
        d_genes[990] = gen(vec2(29.5918, 52.5837), 990); // Шираз
        d_genes[991] = gen(vec2(38.0962, 46.2738), 991); // Тебриз
        d_genes[992] = gen(vec2(34.3416, 47.0861), 992); // Керманшах
        d_genes[993] = gen(vec2(35.7000, 51.4000), 993); // Кередж
        d_genes[994] = gen(vec2(30.2839, 57.0833), 994); // Керман
        d_genes[995] = gen(vec2(27.1832, 56.2666), 995); // Бендер-Аббас
        d_genes[996] = gen(vec2(25.2854, 51.5310), 996); // Доха
    }
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
     
    time_t start, stop;

    // Запуск генетического алгоритма.
    //------------------------------------------------------------------------------------------------
    std::cerr << "Start gatsp in " << THREADS_PER_BLOCKS << " threads and " << BLOCKS_PER_GRID << " blocks.\n";

    start = clock();
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

    stop = clock();
    
    if (DEBUG) { print_debug(d_population); }
    
    print_result(fitnesses);

    double timer = ((double)(stop-start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer << " seconds.\n";
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
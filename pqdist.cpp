#include "pqdist.hpp"
#include <cstring>
#include <memory>
#include <immintrin.h>
#include <emmintrin.h>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <climits>
#include <chrono>
#include <math.h>
#include <c++/9/iomanip>
#include <vector>
using namespace std;
using namespace std::chrono;
Tester::Tester(int d_, int m_, int nbits_, int pqdist_type_, int data_num_, int query_num_, int batch_size_) : d(d_), m(m_), nbits(nbits_), batch_size(batch_size_), pqdist_type(pqdist_type_), data_num(data_num_), query_num(query_num_)
{
    if (pqdist_type == PQDist_TYPE_NAIVE)
        pqdist = new PQDistNaive(d_, m_, nbits_);
    else if (pqdist_type == PQDist_TYPE_SIMD)
        pqdist = new PQDistSIMD(d_, m_, nbits_);
#ifdef __AVX512F__
    else if (pqdist_type == PQDist_TYPE_SIMD_QUANTIZE)
        pqdist = new PQDistSIMDQuantize(d_, m_, nbits_);
#endif
    else
        perror("pqdist_type error");

    encodes = (uint8_t *)aligned_alloc(64, data_num / 8 * m * nbits * sizeof(uint8_t));
    datas = (float *)aligned_alloc(64, data_num * d * sizeof(float));
    querys = (float *)aligned_alloc(64, query_num * d * sizeof(float));
    realDistances = (float *)aligned_alloc(64, query_num * data_num * sizeof(float));
    memset(realDistances, 0, query_num * data_num * sizeof(float));
    PQDistances = (float *)aligned_alloc(64, query_num * data_num * sizeof(float));
    memset(PQDistances, 0, data_num * query_num * sizeof(float));
}

Tester::~Tester()
{

    if (pqdist != nullptr)
        delete pqdist;
    if (encodes != nullptr)
        free(encodes);
    if (datas != nullptr)
        free(datas);
    if (querys != nullptr)
        free(querys);
    if (realDistances != nullptr)
        free(realDistances);
    if (PQDistances != nullptr)
        free(PQDistances);

}

void Tester::load_PQ(string filename)
{
    ifstream fin(filename, std::ios::binary);
    if (!fin.is_open())
    {
        cout << "open " << filename << " fail\n";
        exit(-1);
    }
    int N, d, m, nbits;
    fin.read(reinterpret_cast<char *>(&N), 4);
    fin.read(reinterpret_cast<char *>(&d), 4);
    fin.read(reinterpret_cast<char *>(&m), 4);
    fin.read(reinterpret_cast<char *>(&nbits), 4);
    if (pqdist->d != d || pqdist->m != m || pqdist->nbits != nbits)
    {
        cout << "读入文件的参数与初始化参数不符\n";
        exit(-1);
    }
    cout << "load: " << N << ' ' << d << " " << m << " " << nbits << endl;
    assert(8 % pqdist->nbits == 0);

    fin.read(reinterpret_cast<char *>(pqdist->centroids), d * (1 << nbits) * sizeof(float));

    fin.read(reinterpret_cast<char *>(encodes), min(N, data_num) / 8 * m * nbits);

    fin.close();
}

void read_dataset(string filename, float *datas, int num_limit)
{
    ifstream fin(filename, std::ios::binary);
    if (!fin.is_open())
    {
        cout << "open " << filename << " fail\n";
        exit(-1);
    }
    if (num_limit == -1)
        num_limit = INT_MAX;
    int d;
    int count = 0;
    while (!fin.eof())
    {
        fin.read(reinterpret_cast<char *>(&d), 4);
        fin.read(reinterpret_cast<char *>(datas + count * d), d * sizeof(float));
        count++;
        if (count >= num_limit)
            break;
    }
    fin.close();
}
void Tester::load_query(string filename)
{
    read_dataset(filename, querys, query_num);
    cout << "Load Query is done\n";
}
void Tester::load_data(string filename)
{
    read_dataset(filename, datas, data_num);
    cout << "Load Data is done\n";
}

PQDist::PQDist(int d_, int m_, int nbits_) : d(d_), m(m_), nbits(nbits_)
{
    code_nums = 1 << nbits;
    d_pq = d / m;
    table_size = m * code_nums;
    encode_size = m * nbits / 8;
    if (nbits > 8)
    {
        cout << "Warning nbits exceeds 8: " << nbits << "\n";
    }
    else if (8 % nbits != 0)
    {
        perror("nbits must be divided by 8!");
    }
    centroids = (float *)aligned_alloc(64, d * code_nums * sizeof(float));
}
PQDistNaive::PQDistNaive(int d_, int m_, int nbits_) : PQDist(d_, m_, nbits_)
{
    pq_dist_cache_data = (float *)aligned_alloc(64, sizeof(float) * table_size);
}
PQDistSIMD::PQDistSIMD(int d_, int m_, int nbits_) : PQDist(d_, m_, nbits_)
{
    pq_dist_cache_data = (float *)aligned_alloc(64, sizeof(float) * table_size);
}
#ifdef __AVX512F__
PQDistSIMDQuantize::PQDistSIMDQuantize(int d_, int m_, int nbits_) : PQDist(d_, m_, nbits_)
{
    pq_dist_cache_data = (float *)aligned_alloc(64, sizeof(float) * table_size);
    pq_dist_cache_data_uint8 = (uint8_t *)aligned_alloc(64, sizeof(uint8_t) * table_size);
}
#endif
PQDistNaive::~PQDistNaive()
{
    if (pq_dist_cache_data != nullptr)
        free(pq_dist_cache_data);
    if (centroids != nullptr)
        free(centroids);
}
PQDistSIMD::~PQDistSIMD()
{
    if (pq_dist_cache_data != nullptr)
        free(pq_dist_cache_data);
    if (centroids != nullptr)
        free(centroids);
}
#ifdef __AVX512F__
PQDistSIMDQuantize::~PQDistSIMDQuantize()
{

    if (pq_dist_cache_data != nullptr)
        free(pq_dist_cache_data);
    if (centroids != nullptr)
        free(centroids);
    if (pq_dist_cache_data_uint8 != nullptr)
        free(pq_dist_cache_data_uint8);


}
#endif

void PQDistNaive::load_query(float *query)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < code_nums; j++)
        {

            pq_dist_cache_data[i * code_nums + j] = 0;
            for (int k = 0; k < d_pq; k++)
            {
                float dist = query[i * d_pq + k] - centroids[i * code_nums * d_pq + j * d_pq + k];
                pq_dist_cache_data[i * code_nums + j] += dist * dist;
            }
        }
}

void PQDistSIMD::load_query(float *query)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < code_nums; j++)
        {
            pq_dist_cache_data[i * code_nums + j] = 0;
            for (int k = 0; k < d_pq; k++)
            {
                float dist = query[i * d_pq + k] - centroids[i * code_nums * d_pq + j * d_pq + k];
                pq_dist_cache_data[i * code_nums + j] += dist * dist;
            }
        }
}
#ifdef __AVX512F__
void PQDistSIMDQuantize::load_query(float *query)
{
    minx = INT_MAX;
    maxx = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < code_nums; j++)
        {
            float dist = 0;
            for (int k = 0; k < d_pq; k++)
            {
                float diff = query[i * d_pq + k] - centroids[i * code_nums * d_pq + j * d_pq + k];
                dist += diff * diff;
            }
            pq_dist_cache_data[i * code_nums + j] = dist;
            minx = min(minx, dist);
            maxx = max(maxx, dist);
        }
    }

    scale = (maxx - minx) / 255.0;

    for (int i = 0; i < m; i++)
        for (int j = 0; j < code_nums; j++)
        {
            pq_dist_cache_data_uint8[i * code_nums + j] = round((pq_dist_cache_data[i * code_nums + j] - minx) / scale);
        }

    // print_m512i_uint8(simd_registers[0]);
}
#endif

void PQDistNaive::calc_dist(uint8_t *encodes, int data_num, int batch_size, float *dists)
{
    for (int b = 0; b < data_num; b += batch_size)
    {
        uint8_t* b_encodes =  encodes + b * m * nbits / 8;
        float* b_dists = dists + b;

        if (nbits == 4)
        {
            for (int i = 0; i < batch_size; i++)
            {
                float dist = 0;
                uint8_t *encode = b_encodes + i * m / 2;
                for (int j = 0; j < m; j += 2)
                {
                    uint8_t id = encode[j / 2];
                    uint8_t high_id = (id & 0xF0) >> 4;

                    uint8_t low_id = id & 0x0F;

                    dist += pq_dist_cache_data[j * code_nums + low_id] + pq_dist_cache_data[(j + 1) * code_nums + high_id];
                }
                b_dists[i] = dist;
            }
        }
        else if (nbits == 8)
        {
            for (int i = 0; i < batch_size; i++)
            {
                float dist = 0;
                uint8_t *encode = b_encodes + i * m;
                for (int j = 0; j < m; j++)
                {
                    dist += pq_dist_cache_data[j * code_nums + encode[j]];
                }
                b_dists[i] = dist;
            }
        }
        else
        {
            perror("nbits error");
        }
    }
}

// 使用转置之后的encodes计算pq距离
void PQDistSIMD::calc_dist(uint8_t *encodes, int data_num, int batch_size, float *dists)
{
    for (int b = 0; b < data_num; b += batch_size)
    {
        uint8_t* b_encodes =  encodes + b * m * nbits / 8;
        float* b_dists = dists + b;
        if (nbits == 4)
        {
            for (int i = 0; i < m; i += 2)
            {
                for (int j = 0; j < batch_size; j++)
                {
                    uint8_t id = b_encodes[(i / 2) * batch_size + j];
                    uint8_t low_id = id & 0x0F;
                    uint8_t high_id = (id & 0xF0) >> 4;
                    b_dists[j] += pq_dist_cache_data[i * code_nums + low_id] + pq_dist_cache_data[(i + 1) * code_nums + high_id];
                }
            }
        }
        else if (nbits == 8)
        {
            for (int i = 0; i < m; i++)
            {
                uint8_t *encode = b_encodes + i * batch_size;
                __builtin_prefetch(pq_dist_cache_data + i * code_nums, 0, 3);
                __builtin_prefetch(b_encodes + i * batch_size, 0, 3);
                for (int j = 0; j < batch_size; j++)
                {
                    b_dists[j] += pq_dist_cache_data[i * code_nums + encode[j]];
                }
            }
        }
        else
        {
            perror("nbits error");
        }
    }
}

// encodes已经转置。
#ifdef __AVX512F__
void PQDistSIMDQuantize::calc_dist(uint8_t *encodes, int data_num, int batch_size, float *dists)
{
    assert(nbits == 4);
    // print_m512i_uint8i(simd_registers[i / 8 + 1]);
    __m512i simd_registers[m / 4];
    uint8_t *temp_buffers = (uint8_t *)aligned_alloc(64, sizeof(uint8_t) * 4 * (1 << nbits));
    for (int i = 0; i < m; i += 8)
    {

        memcpy(temp_buffers, pq_dist_cache_data_uint8 + i * code_nums, sizeof(uint8_t) * code_nums);
        memcpy(temp_buffers + code_nums, pq_dist_cache_data_uint8 + (i + 2) * code_nums, sizeof(uint8_t) * code_nums);
        memcpy(temp_buffers + 2 * code_nums, pq_dist_cache_data_uint8 + (i + 4) * code_nums, sizeof(uint8_t) * code_nums);
        memcpy(temp_buffers + 3 * code_nums, pq_dist_cache_data_uint8 + (i + 6) * code_nums, sizeof(uint8_t) * code_nums);
        simd_registers[2 * (i / 8)] = _mm512_load_si512(temp_buffers);
        // print_m512i_uint8(simd_registers[i / 8]);
        memcpy(temp_buffers, pq_dist_cache_data_uint8 + (i + 1) * code_nums, sizeof(uint8_t) * code_nums);
        memcpy(temp_buffers + code_nums, pq_dist_cache_data_uint8 + (i + 3) * code_nums, sizeof(uint8_t) * code_nums);
        memcpy(temp_buffers + 2 * code_nums, pq_dist_cache_data_uint8 + (i + 5) * code_nums, sizeof(uint8_t) * code_nums);
        memcpy(temp_buffers + 3 * code_nums, pq_dist_cache_data_uint8 + (i + 7) * code_nums, sizeof(uint8_t) * code_nums);
        simd_registers[2 * (i / 8) + 1] = _mm512_load_si512(temp_buffers);
    }
    free(temp_buffers);
    __m512i mask = _mm512_set1_epi8(0x0F);
    __m512 scale_f = _mm512_set1_ps(scale);
    __m512 minx_f = _mm512_set1_ps(minx);
    __m512i index = _mm512_setzero_si512();
    __m512i dist = _mm512_setzero_si512();

    // print_m512i_uint8(simd_registers[20]);
    for (int b = 0; b < data_num; b += batch_size)
    {
        

        uint8_t* b_encodes = encodes + b * m * nbits / 8;
        float* b_dists = dists + b;

        __m512i acc = _mm512_setzero_si512();
        for (int i = 0; i < m; i += 8)
        {

            index = _mm512_load_si512(b_encodes + i * batch_size * nbits / 8);
            // print_m512i_uint8(index);
            __m512i partial_id = _mm512_and_si512(index, mask);
            // print_m512i_uint8(partial_id);

            __m512i dist = _mm512_shuffle_epi8(simd_registers[2 * (i / 8)], partial_id);
            // print_m512i_uint8(dist);
            extract_and_upcast_and_add(acc, dist);
            // 饱和加法
            // dist = _mm512_adds_epu8(dist, partial_dist);
            // 将index右移4位。
            index = _mm512_srli_epi16(index, 4);
            partial_id = _mm512_and_si512(index, mask);
            // print_m512i_uint8(partial_id);
            dist = _mm512_shuffle_epi8(simd_registers[2 * (i / 8) + 1], partial_id);
            // print_m512i_uint8(dist);
            extract_and_upcast_and_add(acc, dist);
        }

        __m512 acc_f = _mm512_cvtepi32_ps(acc);

        acc_f = _mm512_mul_ps(acc_f, scale_f);
        acc_f = _mm512_add_ps(acc_f, minx_f);
        // 将acc存入dists
        _mm512_store_ps(b_dists, acc_f);
    }
}

#endif

// query是一个,datas是多个。。。。
void Tester::calc_real_dist(float *query, float *datas, float *dists)
{
    // L2 distance
    for (int b = 0; b < data_num; b += batch_size)
    {
        int size = min(batch_size, data_num - b);

        float* b_datas = datas + b * d;
        float* b_dists = dists + b;

        for (int i = 0; i < size; i++)
        {
            float dist = 0;
            for (int j = 0; j < d; j++)
            {
                float diff = query[j] - b_datas[i * d + j];
                dist += diff * diff;
            }
            b_dists[i] = dist;
        }
    }
}

void Tester::transpose_encodes_bybatch(int batch_size)
{
    // 转置encodes， 假设encodes大小是batch_size的倍数，
    // 将encodes由N * (m * nbits / 8)转置为(N / batch_size * m * nbits / 8) * batch_size
    // 只支持nbits = 4

    uint8_t *tmp = (uint8_t *)aligned_alloc(64, m * batch_size * nbits / 8 * sizeof(uint8_t));

    for (int i = 0; i < data_num; i += batch_size)
    {
        for (int j = 0; j < batch_size; j++)
            for (int k = 0; k < m * nbits / 8; k++)
            {
                tmp[k * batch_size + j] = encodes[(i + j) * m * nbits / 8 + k];
            }
        memcpy(encodes + i * m * nbits / 8, tmp, m * batch_size * nbits / 8);
    }
    free(tmp);
}
// 与PQDistNaive相同。

// load文件之后调用test函数。
void Tester::test()
{

    // 计算真实距离
    auto start_real_dist = high_resolution_clock::now();
    for (int i = 0; i < query_num; i++)
        calc_real_dist(querys + i * d, datas, realDistances + i * data_num);

    auto end_real_dist = high_resolution_clock::now();
    cout << "计算真实距离时间: "
         << duration_cast<milliseconds>(end_real_dist - start_real_dist).count()
         << " ms" << endl;

    // 计算 PQ 近似距离
    if (pqdist_type != PQDist_TYPE_NAIVE)
    {
        transpose_encodes_bybatch(batch_size);
    }
    auto start_pq_dist = high_resolution_clock::now();
    for (int i = 0; i < query_num; i++)
    {
        pqdist->load_query(querys + i * d);
        pqdist->calc_dist(encodes, data_num, batch_size, PQDistances + i * data_num);
    }
    auto end_pq_dist = high_resolution_clock::now();
    cout << "计算 PQ 近似距离时间: "
         << duration_cast<milliseconds>(end_pq_dist - start_pq_dist).count()
         << " ms" << endl;

    // 计算误差
    auto start_error_calc = high_resolution_clock::now();
    double mse = 0;
    double abs = 0;
    for (int i = 0; i < query_num; i++)
    {
        for (int j = 0; j < data_num; j++)
        {
            // cout << realDistances[i * data_num + j] << " " << PQDistances[i * data_num + j] << endl;
            float abs_error = std::abs(sqrt(realDistances[i * data_num + j]) - sqrt(PQDistances[i * data_num + j]));
            mse += abs_error * abs_error;
            abs += abs_error / sqrt(realDistances[i * data_num + j]);
        }
    }
    mse /= (query_num * data_num);
    abs /= (query_num * data_num);

    // 输出最终误差
    cout << "平均均方误差：" << mse << endl;
    cout << "平均偏离：" << abs << endl;
}

#ifdef __AVX512F__
void print_m512i_uint8(__m512i reg)
{
    alignas(64) uint8_t buffer[64];             // 对齐的缓冲区，用于存储寄存器内容
    _mm512_store_si512((__m512i *)buffer, reg); // 将寄存器内容存储到缓冲区

    std::cout << "Contents of __m512i (64 uint8_t values):" << std::endl;
    for (int i = 0; i < 64; i++)
    {
        std::cout << std::setw(3) << (int)buffer[i] << " "; // 打印每个字节的值
        if ((i + 1) % 16 == 0)
        {
            std::cout << std::endl; // 每 16 个值换行，便于阅读
        }
    }
    std::cout << std::endl;
}
#endif

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <dataset_name> (gist/sift)" << endl;
        return 1;
    }

    string name = argv[1]; // 从命令行获取数据集名称
    cout << "Testing " << name << " Dataset\n";
    string output = "res_" + name + ".txt";
    freopen(output.c_str(), "w", stdout);
    if (name == "gist")
    {
        vector<int> pq_m_list = {120, 240, 320, 480};
        vector<int> nbits_list = {4, 8};
        vector<int> pqdist_type_list = {PQDist_TYPE_NAIVE, PQDist_TYPE_SIMD_QUANTIZE};
        vector<int> size_list = {1 << 9, 1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14};
        for (int size : size_list)
        {
            cout << "**************Testing with size = " << size << " **************\n";
            cout << endl;
            for (int pqdist_type : pqdist_type_list)
            {
                if (pqdist_type == PQDist_TYPE_NAIVE)
                {
                    cout << "*****Testing with PQDist_TYPE_NAIVE *****\n";
                    for (int m : pq_m_list)
                        for (int nbits : nbits_list)
                        {
                            Tester *tester = new Tester(960, m, nbits, pqdist_type, size, 1000, 16);
                            // uint8_t* encodes = (uint8_t*)malloc(sizeof(uint8_t) * 16);
                            string pq_file_name = "/root/pqdist_2/gist_encoded_data_1000000_" + to_string(m) + "_" + to_string(nbits);
                            tester->load_PQ(pq_file_name);
                            tester->load_query("/root/gist/test.fvecs");
                            tester->load_data("/root/gist/train.fvecs");
                            tester->test();
                            delete tester;
                            cout << endl;
                        }
                }
                else if (pqdist_type == PQDist_TYPE_SIMD_QUANTIZE)
                {
                    cout << "******Testing with PQDist_TYPE_SIMD_QUANTIZE *****\n";
                    Tester *tester = new Tester(960, 120, 4, pqdist_type, size, 1000, 16);
                    string pq_file_name = "/root/pqdist_2/gist_encoded_data_1000000_120_4";
                    tester->load_PQ(pq_file_name);
                    tester->load_query("/root/gist/test.fvecs");
                    tester->load_data("/root/gist/train.fvecs");
                    tester->test();
                    delete tester;
                    cout << endl;
                }
            }
        }
    }
    else if (name == "sift")
    {
        vector<int> pq_m_list = {16, 32, 64};
        vector<int> nbits_list = {4, 8};
        vector<int> pqdist_type_list = {PQDist_TYPE_NAIVE, PQDist_TYPE_SIMD_QUANTIZE};
        vector<int> size_list = {1 << 9, 1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14};
        for (int size : size_list)
        {
            cout << "**************Testing with size = " << size << " **************\n";
            cout << endl;
            for (int pqdist_type : pqdist_type_list)
            {
                if (pqdist_type == PQDist_TYPE_NAIVE)
                {
                    cout << "*******Testing with PQDist_TYPE_NAIVE *******\n";
                    for (int m : pq_m_list)
                        for (int nbits : nbits_list)
                        {
                            Tester *tester = new Tester(128, m, nbits, pqdist_type, size, 1000, 16);
                            // uint8_t* encodes = (uint8_t*)malloc(sizeof(uint8_t) * 16);
                            string pq_file_name = "/root/pqdist_2/sift_encoded_data_1000000_" + to_string(m) + "_" + to_string(nbits);
                            tester->load_PQ(pq_file_name);
                            tester->load_query("/root/sift/test.fvecs");
                            tester->load_data("/root/sift/train.fvecs");
                            tester->test();
                            delete tester;
                            cout << endl;
                        }
                }
                else if (pqdist_type == PQDist_TYPE_SIMD_QUANTIZE)
                {
                    cout << "*******Testing with PQDist_TYPE_SIMD_QUANTIZE *******\n";
                    for (int m : pq_m_list)
                    {
                        int nbits = 4;
                        Tester *tester = new Tester(128, m, nbits, pqdist_type, size, 1000, 16);
                        // uint8_t* encodes = (uint8_t*)malloc(sizeof(uint8_t) * 16);
                        string pq_file_name = "/root/pqdist_2/sift_encoded_data_1000000_" + to_string(m) + "_" + to_string(nbits);
                        tester->load_PQ(pq_file_name);
                        tester->load_query("/root/sift/test.fvecs");
                        tester->load_data("/root/sift/train.fvecs");
                        tester->test();
                        delete tester;
                        cout << endl;
                    }
                }
            }
        }
    }
    return 0;
}

#pragma once
#ifdef __AVX512F__
#include <immintrin.h>
#endif
#include <string>
#include <iostream>
#define PQDist_TYPE_NAIVE 0
#define PQDist_TYPE_SIMD 1
#define PQDist_TYPE_SIMD_QUANTIZE 2

using namespace std;


class PQDist {
    public:
    virtual ~PQDist() = default;
    PQDist(int d_, int m_, int nbits_);

    int d;
    int m;
    int nbits;
    int d_pq;
    size_t table_size;
    int code_nums;
    float* centroids;
    //一条数据encode的大小
    int encode_size;
    virtual void calc_dist(uint8_t* encodes, int data_num, int batch_size, float* dists) = 0;
    virtual void load_query(float* query) = 0;

};
class PQDistNaive : public PQDist{
    public:
    PQDistNaive(int d_, int m_, int nbits_);
    float* pq_dist_cache_data;
    void calc_dist(uint8_t* encodes, int data_num, int batch_size, float* dists) override;
    void load_query(float* query) override;
    ~PQDistNaive() override;
    
};
class PQDistSIMD : public PQDist{
    public:
    PQDistSIMD(int d_, int m_, int nbits_);
    float* pq_dist_cache_data;
    void calc_dist(uint8_t* encodes, int data_num, int batch_size, float* dists) override;
    void load_query(float* query) override;
    ~PQDistSIMD() override;
};
//仅在支持avx12f时编译pqdistsimdquantize，方便测试不使用simd的时间。
#ifdef __AVX512F__
class PQDistSIMDQuantize : public PQDist{
    public:
    PQDistSIMDQuantize(int d_, int m_, int nbits_);
    float* pq_dist_cache_data;
    uint8_t* pq_dist_cache_data_uint8;
    void calc_dist(uint8_t* encodes, int data_num, int batch_size, float* dists) override;
    void load_query(float* query) override;
    ~PQDistSIMDQuantize() override;
    float minx;
    float maxx;
    float scale;

};
#endif

class  Tester {
    public:
    Tester(int d, int m, int nbits, int pqdist_type, int data_num, int query_num, int batch_size);
    ~Tester();
    uint8_t* encodes;
    void load_PQ(string filename);
    void load_query(string filename);
    void load_data(string filename);
    PQDist* pqdist;
    int pqdist_type;
    int d;
    int m;
    int nbits;
    int batch_size;
    int data_num;
    int query_num;
    float* datas;
    float* querys;
    float* realDistances;
    float* PQDistances;
    void test();
    void calc_real_dist(float* query, float* data, float* dists);
    void transpose_encodes_bybatch(int batch_size);
};
void read_dataset(string filename, float* datas, int num_limit);

//For Debug
#ifdef __AVX512F__
void print_m512i_uint8(__m512i reg);
inline void print_m512i_int32(__m512i reg){
    int32_t* data = (int32_t*)&reg;
    for(int i = 0; i < 16; i++){
        cout << data[i] << " ";
    }
    cout << endl;
}
inline void extract_and_upcast_and_add(__m512i& acc, __m512i& a){
    acc = _mm512_add_epi32(acc, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(a, 0)));
    acc = _mm512_add_epi32(acc, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(a, 1)));
    acc = _mm512_add_epi32(acc, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(a, 2)));
    acc = _mm512_add_epi32(acc, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(a, 3)));
}
#endif
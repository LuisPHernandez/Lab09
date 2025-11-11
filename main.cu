// CC3086 - Lab 9: Chat-Box con IA (CUDA) - Smart Home
// Compilar: nvcc -O3 -std=c++17 main.cu -o chatbox_cuda
// ejecutar: ./chatbox_cuda
// Genera las gráficas: pip install pandas matplotlib seaborn python generate_plots.py


#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <array>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <chrono>

// ------------------------ Utilidades ------------------------

static double percentile(std::vector<float> v, double p){
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  double r = (p/100.0)*(v.size()-1);
  size_t lo = (size_t)std::floor(r), hi = (size_t)std::ceil(r);
  if (lo == hi) return v[lo];
  double w = r - lo;
  return (1.0 - w)*v[lo] + w*v[hi];
}

static std::vector<int> topk_indices(const float* scores, int K, int k){
  std::vector<int> idx(K);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin()+std::min(k,K), idx.end(),
                    [&](int a, int b){ return scores[a] > scores[b]; });
  idx.resize(std::min(k,K));
  return idx;
}

static void open_metrics_csv(std::ofstream& ofs, const std::string& path){
  ofs.open(path, std::ios::out);
  ofs << "query_id,query_text,top_intent,top2_intent,top3_intent,decision,new_alarm,new_luces,new_vent,new_desh,"
         "mean_mov,mean_lux,mean_temp,mean_ruido,mean_hum,"
         "lat_total_ms,lat_nlu_ms,lat_data_ms,lat_fuse_ms\n";
}

static void append_metrics_csv(std::ofstream& ofs,
                               int qid, const std::string& qtext,
                               const std::vector<std::string>& top_intents,
                               int decision,
                               int new_alarm, int new_luces, int new_vent, int new_desh,
                               const float meanC[5],
                               float t_total, float t_nlu, float t_data, float t_fuse){
  ofs << qid << ","
      << "\"" << qtext << "\"" << ",";
  
  // TOP-3 intenciones
  for (size_t i = 0; i < 3; ++i) {
    if (i < top_intents.size()) {
      ofs << top_intents[i];
    }
    if (i < 2) ofs << ",";
  }
  
  ofs << "," << decision << ","
      << new_alarm << "," << new_luces << "," << new_vent  << "," << new_desh << ","
      << std::fixed << std::setprecision(3)
      << meanC[0] << "," << meanC[1] << "," << meanC[2] << ","
      << meanC[3] << "," << meanC[4] << ","
      << std::setprecision(3)
      << t_total << "," << t_nlu << "," << t_data << "," << t_fuse << "\n";
}

#define CUDA_OK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line){
  if (code != cudaSuccess){
    fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

inline int ceilDiv(int a, int b){ return (a + b - 1) / b; }

// ------------------------ Parámetros ------------------------
constexpr int D = 8192;
constexpr int K = 5;
constexpr int TOPK = 3; 
constexpr int MAX_QUERY = 512;
constexpr int C = 5;      
constexpr int N = 1<<20;  
constexpr int W = 1024;   

// ------------------------ Hash 3-gramas ------------------------
__device__ __forceinline__
uint32_t hash3(uint8_t a, uint8_t b, uint8_t c){
  uint32_t h = 2166136261u;
  h = (h ^ a) * 16777619u;
  h = (h ^ b) * 16777619u;
  h = (h ^ c) * 16777619u;
  return h % D;
}

// ------------------------ Kernels NLU ------------------------
__global__
void tokenize3grams(const char* __restrict__ query, int n,
                    float* __restrict__ vq){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i+2 >= n) return;
  uint32_t idx = hash3((uint8_t)query[i], (uint8_t)query[i+1], (uint8_t)query[i+2]);
  atomicAdd(&vq[idx], 1.0f);
}

__global__
void l2normalize(float* __restrict__ v, int d){
  __shared__ float ssum[256];
  float acc = 0.f;
  for (int j = threadIdx.x; j < d; j += blockDim.x){
    float x = v[j];
    acc += x*x;
  }
  ssum[threadIdx.x] = acc;
  __syncthreads();
  for (int offset = blockDim.x>>1; offset > 0; offset >>= 1){
    if (threadIdx.x < offset) ssum[threadIdx.x] += ssum[threadIdx.x+offset];
    __syncthreads();
  }
  float norm = sqrtf(ssum[0] + 1e-12f);
  __syncthreads();
  for (int j = threadIdx.x; j < d; j += blockDim.x){
    v[j] = v[j] / norm;
  }
}

__global__
void matvecDotCos(const float* __restrict__ M, const float* __restrict__ vq,
                  float* __restrict__ scores, int K, int D){
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  float acc = 0.f;
  for (int j = 0; j < D; ++j) acc += M[k*D + j] * vq[j];
  scores[k] = acc;
}

// ------------------------ Kernels Sensores ------------------------
__global__
void window_stats_last(const float* __restrict__ X,
                       int N, int C, int W,
                       float* __restrict__ mean_out, float* __restrict__ std_out){
  int c = blockIdx.x;
  if (c >= C) return;
  float sum = 0.f, sum2 = 0.f;
  int start = max(0, N - W);
  for (int i = threadIdx.x; i < W; i += blockDim.x){
    float v = X[(start + i)*C + c];
    sum  += v;
    sum2 += v*v;
  }
  __shared__ float ssum[256], ssum2[256];
  ssum[threadIdx.x] = sum;
  ssum2[threadIdx.x] = sum2;
  __syncthreads();
  for (int off = blockDim.x>>1; off > 0; off >>= 1){
    if (threadIdx.x < off){
      ssum[threadIdx.x]  += ssum[threadIdx.x+off];
      ssum2[threadIdx.x] += ssum2[threadIdx.x+off];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0){
    float m = ssum[0] / W;
    float var = fmaxf(ssum2[0]/W - m*m, 0.f);
    mean_out[c] = m;
    std_out[c]  = sqrtf(var);
  }
}

// ------------------------ Kernel Fusión / Decisión ------------------------
enum Intent { 
  TOGGLE_ALARMA = 0,
  TOGGLE_LUCES = 1,
  TOGGLE_VENTILADOR = 2,
  TOGGLE_DESHUMIDIFICADOR = 3,
  CONSULTAR_ESTADO = 4
};

static const std::array<std::vector<std::string>, K> TEMPLATES = {{
  {"activar alarma", "apagar alarma", "alarma on", "alarma off", "encender alarma", "desactivar alarma", "prender alarma", "silenciar alarma", "sonar alarma"},
  {"prender luces", "apagar luces", "encender iluminacion", "luces afuera", "luz exterior", "iluminar entrada", "activar luces", "desactivar luces"},
  {"encender ventilador", "apagar ventilador", "activar ventilacion", "ventilador on", "ventilador off", "encender enfriamiento"},
  {"activar deshumidificador", "apagar deshumidificador", "deshumidificador on", "deshumidificador off", "reducir humedad"},
  {"como esta la casa", "estado del sistema", "que esta encendido", "resumen", "status general", "datos de sensores", "estado de la casa", "resumen de registros", "estadisticas del sistema"}
}};

__global__
void fuseDecision(const float* __restrict__ scores, int K,
                  const float* __restrict__ meanC,
                  float thrLuz, float thrTemp, float thrRuido, float thrHum,
                  const int* __restrict__ curAlarm, const int* __restrict__ curLuces,
                  const int* __restrict__ curVent,  const int* __restrict__ curDesh,
                  int* __restrict__ outTop,
                  int* __restrict__ newAlarm, int* __restrict__ newLuces,
                  int* __restrict__ newVent,  int* __restrict__ newDesh)
{
  __shared__ int topIdx;
  __shared__ float topScore;

  if (threadIdx.x == 0){ topIdx = 0; topScore = scores[0]; }
  __syncthreads();

  for (int k = threadIdx.x; k < K; k += blockDim.x){
    float s = scores[k];
    if (s > topScore){ topScore = s; topIdx = k; }
  }
  __syncthreads();

  if (threadIdx.x == 0){
    *outTop = topIdx;

    float mov   = meanC[0];
    float luz   = meanC[1];
    float temp  = meanC[2];
    float ruido = meanC[3];
    float hum   = meanC[4];

    int sAlarm = *curAlarm;
    int sLuces = *curLuces;
    int sVent  = *curVent;
    int sDesh  = *curDesh;

    int nAlarm = sAlarm, nLuces = sLuces, nVent = sVent, nDesh = sDesh;

    bool oscuro   = (luz <  thrLuz);
    bool caliente = (temp > thrTemp);
    bool ruidoso  = (ruido > thrRuido);
    bool humedo   = (hum >  thrHum);
    bool hayMov   = (mov >= 0.5f);

    switch (topIdx){
      case TOGGLE_ALARMA:
        if (hayMov || ruidoso) nAlarm = 1;
        else nAlarm = 1 - sAlarm;
        break;
      case TOGGLE_LUCES:
        if (oscuro) nLuces = 1;
        else nLuces = 1 - sLuces;
        break;
      case TOGGLE_VENTILADOR:
        if (caliente) nVent = 1;
        else nVent = 1 - sVent;
        break;
      case TOGGLE_DESHUMIDIFICADOR:
        if (humedo) nDesh = 1;
        else nDesh = 1 - sDesh;
        break;
      case CONSULTAR_ESTADO:
      default:
        break;
    }

    *newAlarm = nAlarm;
    *newLuces = nLuces;
    *newVent  = nVent;
    *newDesh  = nDesh;
  }
}

// ------------------------ Host helpers ------------------------
static inline uint32_t hash3_host(uint8_t a, uint8_t b, uint8_t c){
  uint32_t h = 2166136261u;
  h = (h ^ a) * 16777619u;
  h = (h ^ b) * 16777619u;
  h = (h ^ c) * 16777619u;
  return h % D;
}

static void l2normalize_host(std::vector<float>& v){
  double acc = 0.0;
  for (float x : v) acc += double(x)*double(x);
  float n = float(std::sqrt(acc) + 1e-12);
  if (n == 0.f) return;
  for (auto& x : v) x /= n;
}

static void tokenize3grams_host(const std::string& s, std::vector<float>& out){
  std::string q = s;
  if (q.size() < 3) return;
  for (size_t i = 0; i + 2 < q.size(); ++i){
    uint32_t idx = hash3_host((uint8_t)q[i], (uint8_t)q[i+1], (uint8_t)q[i+2]);
    out[idx] += 1.0f;
  }
}

static void buildIntentMatrixFromTemplates(std::vector<float>& M){
  M.assign(K * D, 0.f);
  for (int k = 0; k < K; ++k){
    std::vector<float> row(D, 0.f);
    const auto& phrases = TEMPLATES[k];
    for (const auto& p : phrases){
      std::vector<float> v(D, 0.f);
      tokenize3grams_host(p, v);
      l2normalize_host(v);
      for (int j = 0; j < D; ++j) row[j] += v[j];
    }
    float inv = phrases.empty() ? 1.f : (1.f / phrases.size());
    for (int j = 0; j < D; ++j) row[j] *= inv;
    {
      double acc=0.0; for (float x:row) acc += double(x)*double(x);
      float n = float(std::sqrt(acc) + 1e-12);
      for (float& x:row) x /= n;
    }
    for (int j = 0; j < D; ++j) M[k*D + j] = row[j];
  }
}

void synthSensors(std::vector<float>& X){
  X.resize(size_t(N)*C);
  srand(12345);

  int burstLen = 0;
  const int minBurst = 50, maxBurst = 500;
  const float pStartBurst = 0.001f;
  const float luxBaseMin = 50.f, luxBaseMax = 800.f;
  const float tempMean = 27.0f, tempJitter = 3.0f;
  const float noiseBaseMin = 35.f, noiseBaseMax = 45.f;
  const float noiseSpikeMin = 65.f, noiseSpikeMax = 85.f;
  const float pNoiseSpike = 0.0008f;
  float humidity = 55.f;
  const float humMin = 40.f, humMax = 70.f;

  for (int i = 0; i < N; ++i){
    int motion = 0;
    if (burstLen > 0){
      motion = 1;
      burstLen--;
    } else {
      float r = (rand() % 10000) / 10000.f;
      if (r < pStartBurst){
        burstLen = minBurst + (rand() % (maxBurst - minBurst + 1));
        motion = 1;
        burstLen--;
      }
    }

    float phase = (i % (1<<16)) * (2.f * 3.14159265f / float(1<<16));
    float diurnal = 0.5f * (1.f + sinf(phase));
    float lux = luxBaseMin + diurnal * (luxBaseMax - luxBaseMin);
    lux += ((rand()%1000)/1000.f - 0.5f) * 30.f;
    if (lux < 0.f) lux = 0.f;

    float temp = tempMean + ((rand()%1000)/1000.f - 0.5f) * (2.f*tempJitter);

    float noise = noiseBaseMin + (rand()%1000)/1000.f * (noiseBaseMax - noiseBaseMin);
    float rp = (rand()%100000) / 100000.f;
    if (rp < pNoiseSpike){
      noise = noiseSpikeMin + (rand()%1000)/1000.f * (noiseSpikeMax - noiseSpikeMin);
    }

    humidity += ((rand()%1000)/1000.f - 0.5f) * 0.2f;
    if (humidity < humMin) humidity = humMin + 0.5f;
    if (humidity > humMax) humidity = humMax - 0.5f;

    X[i*C + 0] = float(motion);
    X[i*C + 1] = lux;
    X[i*C + 2] = temp;
    X[i*C + 3] = noise;
    X[i*C + 4] = humidity;
  }
}

// ------------------------ Función de Benchmark por Configuración de Streams ------------------------
struct BenchmarkResult {
  int num_streams;
  double p50_ms;
  double p95_ms;
  double qps;
  double avg_nlu_ms;
  double avg_data_ms;
  double avg_fuse_ms;
};

BenchmarkResult runBenchmark(int numStreams, const std::vector<std::string>& QUERIES, 
                              const std::vector<float>& hM, float* dM){
  const int Q = (int)QUERIES.size();
  static const char* intentNames[K] = {
    "TOGGLE_ALARMA", "TOGGLE_LUCES", "TOGGLE_VENTILADOR",
    "TOGGLE_DESHUMIDIFICADOR", "CONSULTAR_ESTADO"
  };

  // Crear streams dinámicamente
  std::vector<cudaStream_t> streams(numStreams);
  for (int i = 0; i < numStreams; ++i) {
    CUDA_OK(cudaStreamCreate(&streams[i]));
  }

  cudaEvent_t evStart, evStop;
  CUDA_OK(cudaEventCreate(&evStart));
  CUDA_OK(cudaEventCreate(&evStop));

  cudaEvent_t evNLUStart, evNLUStop, evDATAStart, evDATAStop, evFUSEStart, evFUSEStop;
  CUDA_OK(cudaEventCreate(&evNLUStart));
  CUDA_OK(cudaEventCreate(&evNLUStop));
  CUDA_OK(cudaEventCreate(&evDATAStart));
  CUDA_OK(cudaEventCreate(&evDATAStop));
  CUDA_OK(cudaEventCreate(&evFUSEStart));
  CUDA_OK(cudaEventCreate(&evFUSEStop));

  std::vector<float> all_total_ms, all_nlu_ms, all_data_ms, all_fuse_ms;
  all_total_ms.reserve(Q);
  all_nlu_ms.reserve(Q);
  all_data_ms.reserve(Q);
  all_fuse_ms.reserve(Q);

  char *hQ=nullptr, *dQ=nullptr;
  float *hVQ=nullptr, *dVQ=nullptr, *dScores=nullptr, *hScores=nullptr;
  CUDA_OK(cudaHostAlloc(&hQ, MAX_QUERY, cudaHostAllocDefault));
  CUDA_OK(cudaMalloc(&dQ, MAX_QUERY));
  CUDA_OK(cudaHostAlloc(&hVQ, D*sizeof(float), cudaHostAllocDefault));
  CUDA_OK(cudaHostAlloc(&hScores, K*sizeof(float), cudaHostAllocDefault));
  CUDA_OK(cudaMalloc(&dVQ, D*sizeof(float)));
  CUDA_OK(cudaMalloc(&dScores, K*sizeof(float)));

  std::vector<float> hXvec;
  synthSensors(hXvec);
  float *hX=nullptr;
  CUDA_OK(cudaHostAlloc(&hX, size_t(N)*C*sizeof(float), cudaHostAllocDefault));
  memcpy(hX, hXvec.data(), size_t(N)*C*sizeof(float));

  float *dX=nullptr, *dMean=nullptr, *dStd=nullptr;
  float hMean[C]={0}, hStd[C]={0};
  CUDA_OK(cudaMalloc(&dX, size_t(N)*C*sizeof(float)));
  CUDA_OK(cudaMalloc(&dMean, C*sizeof(float)));
  CUDA_OK(cudaMalloc(&dStd,  C*sizeof(float)));

  for (int qi = 0; qi < Q; ++qi){
    const std::string& q = QUERIES[qi];
    const int qn = std::min<int>((int)q.size(), MAX_QUERY);
    memset(hQ, 0, MAX_QUERY);
    memcpy(hQ, q.data(), qn);

    // Seleccionar stream cíclicamente
    int streamIdx = qi % numStreams;
    cudaStream_t currentStream = streams[streamIdx];

    CUDA_OK(cudaEventRecord(evStart, 0));

    // NLU
    CUDA_OK(cudaEventRecord(evNLUStart, currentStream));
    CUDA_OK(cudaMemsetAsync(dVQ, 0, D*sizeof(float), currentStream));
    CUDA_OK(cudaMemcpyAsync(dQ, hQ, MAX_QUERY, cudaMemcpyHostToDevice, currentStream));
    {
      dim3 blk(256), grd(ceilDiv(qn, (int)blk.x));
      tokenize3grams<<<grd, blk, 0, currentStream>>>(dQ, qn, dVQ);
    }
    l2normalize<<<1,256,0,currentStream>>>(dVQ, D);
    {
      dim3 blk(128), grd(ceilDiv(K,(int)blk.x));
      matvecDotCos<<<grd, blk, 0, currentStream>>>(dM, dVQ, dScores, K, D);
    }
    CUDA_OK(cudaMemcpyAsync(hScores, dScores, K*sizeof(float), cudaMemcpyDeviceToHost, currentStream));
    CUDA_OK(cudaEventRecord(evNLUStop, currentStream));

    // DATA
    CUDA_OK(cudaEventRecord(evDATAStart, currentStream));
    std::vector<float> hXvecIter;
    hXvecIter.reserve((size_t)N*C);
    synthSensors(hXvecIter);
    memcpy(hX, hXvecIter.data(), size_t(N)*C*sizeof(float));
    CUDA_OK(cudaMemcpyAsync(dX, hX, size_t(N)*C*sizeof(float), cudaMemcpyHostToDevice, currentStream));
    window_stats_last<<<C, 256, 0, currentStream>>>(dX, N, C, W, dMean, dStd);
    CUDA_OK(cudaMemcpyAsync(hMean, dMean, C*sizeof(float), cudaMemcpyDeviceToHost, currentStream));
    CUDA_OK(cudaMemcpyAsync(hStd,  dStd,  C*sizeof(float), cudaMemcpyDeviceToHost, currentStream));
    CUDA_OK(cudaEventRecord(evDATAStop, currentStream));

    CUDA_OK(cudaStreamSynchronize(currentStream));

    // FUSE
    CUDA_OK(cudaEventRecord(evFUSEStart, currentStream));
    float *dMeanHost=nullptr;
    CUDA_OK(cudaMalloc(&dMeanHost, C*sizeof(float)));
    CUDA_OK(cudaMemcpyAsync(dMeanHost, hMean, C*sizeof(float), cudaMemcpyHostToDevice, currentStream));

    int hStateAlarm = 0, hStateLuces = 0, hStateVent = 0, hStateDesh = 0;
    const float thrLuz = 250.0f, thrTemp = 27.0f, thrRuido = 60.0f, thrHum = 0.60f;

    int *dStateAlarm=nullptr, *dStateLuces=nullptr, *dStateVent=nullptr, *dStateDesh=nullptr;
    int *dNewAlarm=nullptr, *dNewLuces=nullptr, *dNewVent=nullptr, *dNewDesh=nullptr;
    CUDA_OK(cudaMalloc(&dStateAlarm, sizeof(int)));
    CUDA_OK(cudaMalloc(&dStateLuces, sizeof(int)));
    CUDA_OK(cudaMalloc(&dStateVent,  sizeof(int)));
    CUDA_OK(cudaMalloc(&dStateDesh,  sizeof(int)));
    CUDA_OK(cudaMalloc(&dNewAlarm, sizeof(int)));
    CUDA_OK(cudaMalloc(&dNewLuces, sizeof(int)));
    CUDA_OK(cudaMalloc(&dNewVent,  sizeof(int)));
    CUDA_OK(cudaMalloc(&dNewDesh,  sizeof(int)));

    CUDA_OK(cudaMemcpyAsync(dStateAlarm, &hStateAlarm, sizeof(int), cudaMemcpyHostToDevice, currentStream));
    CUDA_OK(cudaMemcpyAsync(dStateLuces, &hStateLuces, sizeof(int), cudaMemcpyHostToDevice, currentStream));
    CUDA_OK(cudaMemcpyAsync(dStateVent,  &hStateVent,  sizeof(int), cudaMemcpyHostToDevice, currentStream));
    CUDA_OK(cudaMemcpyAsync(dStateDesh,  &hStateDesh,  sizeof(int), cudaMemcpyHostToDevice, currentStream));

    int *dTop=nullptr;
    CUDA_OK(cudaMalloc(&dTop, sizeof(int)));

    fuseDecision<<<1, 128, 0, currentStream>>>(
      dScores, K, dMeanHost, thrLuz, thrTemp, thrRuido, thrHum,
      dStateAlarm, dStateLuces, dStateVent, dStateDesh, dTop,
      dNewAlarm, dNewLuces, dNewVent, dNewDesh
    );

    int hTop=-1, hNewAlarm=0, hNewLuces=0, hNewVent=0, hNewDesh=0;
    CUDA_OK(cudaMemcpyAsync(&hTop,      dTop,       sizeof(int), cudaMemcpyDeviceToHost, currentStream));
    CUDA_OK(cudaMemcpyAsync(&hNewAlarm, dNewAlarm,  sizeof(int), cudaMemcpyDeviceToHost, currentStream));
    CUDA_OK(cudaMemcpyAsync(&hNewLuces, dNewLuces,  sizeof(int), cudaMemcpyDeviceToHost, currentStream));
    CUDA_OK(cudaMemcpyAsync(&hNewVent,  dNewVent,   sizeof(int), cudaMemcpyDeviceToHost, currentStream));
    CUDA_OK(cudaMemcpyAsync(&hNewDesh,  dNewDesh,   sizeof(int), cudaMemcpyDeviceToHost, currentStream));

    CUDA_OK(cudaEventRecord(evFUSEStop, currentStream));
    CUDA_OK(cudaStreamSynchronize(currentStream));

    CUDA_OK(cudaEventRecord(evStop, 0));
    CUDA_OK(cudaEventSynchronize(evStop));

    float msNLU=0.f, msDATA=0.f, msFUSE=0.f, msTOTAL=0.f;
    CUDA_OK(cudaEventElapsedTime(&msNLU,  evNLUStart,  evNLUStop));
    CUDA_OK(cudaEventElapsedTime(&msDATA, evDATAStart, evDATAStop));
    CUDA_OK(cudaEventElapsedTime(&msFUSE, evFUSEStart, evFUSEStop));
    CUDA_OK(cudaEventElapsedTime(&msTOTAL, evStart, evStop));

    all_nlu_ms.push_back(msNLU);
    all_data_ms.push_back(msDATA);
    all_fuse_ms.push_back(msFUSE);
    all_total_ms.push_back(msTOTAL);

    cudaFree(dTop);
    cudaFree(dMeanHost);
    cudaFree(dStateAlarm); cudaFree(dStateLuces); cudaFree(dStateVent); cudaFree(dStateDesh);
    cudaFree(dNewAlarm);   cudaFree(dNewLuces);   cudaFree(dNewVent);   cudaFree(dNewDesh);
  }

  double p50 = percentile(all_total_ms, 50.0);
  double p95 = percentile(all_total_ms, 95.0);
  double sum_ms = std::accumulate(all_total_ms.begin(), all_total_ms.end(), 0.0);
  double qps = (sum_ms>0.0) ? (1000.0 * Q / sum_ms) : 0.0;

  BenchmarkResult result;
  result.num_streams = numStreams;
  result.p50_ms = p50;
  result.p95_ms = p95;
  result.qps = qps;
  result.avg_nlu_ms = std::accumulate(all_nlu_ms.begin(), all_nlu_ms.end(), 0.0) / Q;
  result.avg_data_ms = std::accumulate(all_data_ms.begin(), all_data_ms.end(), 0.0) / Q;
  result.avg_fuse_ms = std::accumulate(all_fuse_ms.begin(), all_fuse_ms.end(), 0.0) / Q;

  cudaFree(dQ); cudaFree(dVQ); cudaFree(dScores);
  cudaFree(dX); cudaFree(dMean); cudaFree(dStd);
  cudaFreeHost(hQ); cudaFreeHost(hVQ); cudaFreeHost(hScores); cudaFreeHost(hX);
  cudaEventDestroy(evNLUStart); cudaEventDestroy(evNLUStop);
  cudaEventDestroy(evDATAStart); cudaEventDestroy(evDATAStop);
  cudaEventDestroy(evFUSEStart); cudaEventDestroy(evFUSEStop);
  cudaEventDestroy(evStart); cudaEventDestroy(evStop);
  
  for (auto& s : streams) {
    cudaStreamDestroy(s);
  }

  return result;
}

// ------------------------ Main ------------------------
int main(){
  printf("╔═══════════════════════════════════════════════════════════╗\n");
  printf("║   LAB 9  CHAT-BOX CUDA SMART HOME  ║\n");
  printf("╚═══════════════════════════════════════════════════════════╝\n\n");

  std::vector<float> hM;
  buildIntentMatrixFromTemplates(hM);
  float *dM=nullptr;
  CUDA_OK(cudaMalloc(&dM, K*D*sizeof(float)));
  CUDA_OK(cudaMemcpy(dM, hM.data(), K*D*sizeof(float), cudaMemcpyHostToDevice));

  std::vector<std::string> QUERIES = {
    "activa la alarma",
    "apaga la alarma",
    "enciende luces exteriores",
    "apaga luces",
    "activa el ventilador",
    "apaga el ventilador",
    "activa el deshumidificador",
    "consulta estado",
    "muestra mediciones",
    "enciende luces si esta oscuro",
    "activar sistema de seguridad",
    "desactivar ventilacion",
    "status de la casa",
    "encender iluminacion exterior",
    "apagar deshumidificador"
  };

  // ========== TAREA 2: EVALUACIÓN CON 1, 2, 4, 8 STREAMS ==========
  std::vector<int> streamConfigs = {1, 2, 4, 8};
  std::vector<BenchmarkResult> results;
  
  printf("Ejecutando benchmarks con diferentes configuraciones de streams...\n\n");
  
  for (int numStreams : streamConfigs) {
    printf("→ Probando con %d stream(s)...\n", numStreams);
    auto result = runBenchmark(numStreams, QUERIES, hM, dM);
    results.push_back(result);
    
    printf("  ✓ Completado: p50=%.3f ms, p95=%.3f ms, QPS=%.2f\n\n", 
           result.p50_ms, result.p95_ms, result.qps);
  }

  // ========== GUARDAR RESULTADOS EN CSV PARA GRÁFICAS ==========
  std::ofstream benchCsv("benchmark_results.csv");
  benchCsv << "num_streams,p50_ms,p95_ms,qps,avg_nlu_ms,avg_data_ms,avg_fuse_ms\n";
  for (const auto& r : results) {
    benchCsv << r.num_streams << ","
             << std::fixed << std::setprecision(3)
             << r.p50_ms << "," << r.p95_ms << ","
             << r.qps << "," << r.avg_nlu_ms << ","
             << r.avg_data_ms << "," << r.avg_fuse_ms << "\n";
  }
  benchCsv.close();

  // ========== EJECUTAR UNA CORRIDA COMPLETA CON BITÁCORA DETALLADA ==========
  printf("═══════════════════════════════════════════════════════════\n");
  printf("Generando bitácora detallada con 3 streams...\n");
  printf("═══════════════════════════════════════════════════════════\n\n");

  cudaStream_t sNLU, sDATA, sFUSE;
  CUDA_OK(cudaStreamCreate(&sNLU));
  CUDA_OK(cudaStreamCreate(&sDATA));
  CUDA_OK(cudaStreamCreate(&sFUSE));

  cudaEvent_t evStart, evStop;
  CUDA_OK(cudaEventCreate(&evStart));
  CUDA_OK(cudaEventCreate(&evStop));

  cudaEvent_t evNLUStart, evNLUStop, evDATAStart, evDATAStop, evFUSEStart, evFUSEStop;
  CUDA_OK(cudaEventCreate(&evNLUStart));
  CUDA_OK(cudaEventCreate(&evNLUStop));
  CUDA_OK(cudaEventCreate(&evDATAStart));
  CUDA_OK(cudaEventCreate(&evDATAStop));
  CUDA_OK(cudaEventCreate(&evFUSEStart));
  CUDA_OK(cudaEventCreate(&evFUSEStop));

  const int Q = (int)QUERIES.size();

  char *hQ=nullptr; CUDA_OK(cudaHostAlloc(&hQ, MAX_QUERY, cudaHostAllocDefault));
  char *dQ=nullptr; CUDA_OK(cudaMalloc(&dQ, MAX_QUERY));

  std::ofstream metricsCsv;
  open_metrics_csv(metricsCsv, "metrics.csv");

  std::vector<float> all_total_ms, all_nlu_ms, all_data_ms, all_fuse_ms;
  all_total_ms.reserve(Q);
  all_nlu_ms.reserve(Q);
  all_data_ms.reserve(Q);
  all_fuse_ms.reserve(Q);

  float *hVQ=nullptr, *dVQ=nullptr, *dScores=nullptr, *hScores=nullptr;
  CUDA_OK(cudaHostAlloc(&hVQ, D*sizeof(float), cudaHostAllocDefault));
  CUDA_OK(cudaHostAlloc(&hScores, K*sizeof(float), cudaHostAllocDefault));
  CUDA_OK(cudaMalloc(&dVQ, D*sizeof(float)));
  CUDA_OK(cudaMalloc(&dScores, K*sizeof(float)));

  std::vector<float> hXvec; synthSensors(hXvec);
  float *hX=nullptr; CUDA_OK(cudaHostAlloc(&hX, size_t(N)*C*sizeof(float), cudaHostAllocDefault));
  memcpy(hX, hXvec.data(), size_t(N)*C*sizeof(float));

  float *dX=nullptr, *dMean=nullptr, *dStd=nullptr;
  float hMean[C]={0}, hStd[C]={0};
  CUDA_OK(cudaMalloc(&dX, size_t(N)*C*sizeof(float)));
  CUDA_OK(cudaMalloc(&dMean, C*sizeof(float)));
  CUDA_OK(cudaMalloc(&dStd,  C*sizeof(float)));

  static const char* intentNames[K] = {
    "TOGGLE_ALARMA",
    "TOGGLE_LUCES",
    "TOGGLE_VENTILADOR",
    "TOGGLE_DESHUMIDIFICADOR",
    "CONSULTAR_ESTADO"
  };

  for (int qi = 0; qi < Q; ++qi){
    const std::string& q = QUERIES[qi];
    const int qn = std::min<int>((int)q.size(), MAX_QUERY);
    memset(hQ, 0, MAX_QUERY);
    memcpy(hQ, q.data(), qn);

    CUDA_OK(cudaEventRecord(evStart, 0));

    // --------- NLU ---------
    CUDA_OK(cudaEventRecord(evNLUStart, sNLU));
    CUDA_OK(cudaMemsetAsync(dVQ, 0, D*sizeof(float), sNLU));
    CUDA_OK(cudaMemcpyAsync(dQ, hQ, MAX_QUERY, cudaMemcpyHostToDevice, sNLU));

    {
      dim3 blk(256), grd(ceilDiv(qn, (int)blk.x));
      tokenize3grams<<<grd, blk, 0, sNLU>>>(dQ, qn, dVQ);
    }
    l2normalize<<<1,256,0,sNLU>>>(dVQ, D);
    {
      dim3 blk(128), grd(ceilDiv(K,(int)blk.x));
      matvecDotCos<<<grd, blk, 0, sNLU>>>(dM, dVQ, dScores, K, D);
    }
    CUDA_OK(cudaMemcpyAsync(hScores, dScores, K*sizeof(float), cudaMemcpyDeviceToHost, sNLU));
    CUDA_OK(cudaEventRecord(evNLUStop, sNLU));

    // --------- DATA ---------
    CUDA_OK(cudaEventRecord(evDATAStart, sDATA));
    std::vector<float> hXvecIter; hXvecIter.reserve((size_t)N*C);
    synthSensors(hXvecIter);
    memcpy(hX, hXvecIter.data(), size_t(N)*C*sizeof(float));

    CUDA_OK(cudaMemcpyAsync(dX, hX, size_t(N)*C*sizeof(float), cudaMemcpyHostToDevice, sDATA));
    window_stats_last<<<C, 256, 0, sDATA>>>(dX, N, C, W, dMean, dStd);
    CUDA_OK(cudaMemcpyAsync(hMean, dMean, C*sizeof(float), cudaMemcpyDeviceToHost, sDATA));
    CUDA_OK(cudaMemcpyAsync(hStd,  dStd,  C*sizeof(float), cudaMemcpyDeviceToHost, sDATA));
    CUDA_OK(cudaEventRecord(evDATAStop, sDATA));

    CUDA_OK(cudaStreamSynchronize(sNLU));
    CUDA_OK(cudaStreamSynchronize(sDATA));

    // --------- FUSE ---------
    CUDA_OK(cudaEventRecord(evFUSEStart, sFUSE));

    float *dMeanHost=nullptr;
    CUDA_OK(cudaMalloc(&dMeanHost, C*sizeof(float)));
    CUDA_OK(cudaMemcpyAsync(dMeanHost, hMean, C*sizeof(float), cudaMemcpyHostToDevice, sFUSE));

    int hStateAlarm = 0, hStateLuces = 0, hStateVent = 0, hStateDesh = 0;
    const float thrLuz   = 250.0f;
    const float thrTemp  = 27.0f;
    const float thrRuido = 60.0f;
    const float thrHum   = 0.60f;

    int *dStateAlarm=nullptr, *dStateLuces=nullptr, *dStateVent=nullptr, *dStateDesh=nullptr;
    int *dNewAlarm=nullptr, *dNewLuces=nullptr, *dNewVent=nullptr, *dNewDesh=nullptr;
    CUDA_OK(cudaMalloc(&dStateAlarm, sizeof(int)));
    CUDA_OK(cudaMalloc(&dStateLuces, sizeof(int)));
    CUDA_OK(cudaMalloc(&dStateVent,  sizeof(int)));
    CUDA_OK(cudaMalloc(&dStateDesh,  sizeof(int)));
    CUDA_OK(cudaMalloc(&dNewAlarm, sizeof(int)));
    CUDA_OK(cudaMalloc(&dNewLuces, sizeof(int)));
    CUDA_OK(cudaMalloc(&dNewVent,  sizeof(int)));
    CUDA_OK(cudaMalloc(&dNewDesh,  sizeof(int)));

    CUDA_OK(cudaMemcpyAsync(dStateAlarm, &hStateAlarm, sizeof(int), cudaMemcpyHostToDevice, sFUSE));
    CUDA_OK(cudaMemcpyAsync(dStateLuces, &hStateLuces, sizeof(int), cudaMemcpyHostToDevice, sFUSE));
    CUDA_OK(cudaMemcpyAsync(dStateVent,  &hStateVent,  sizeof(int), cudaMemcpyHostToDevice, sFUSE));
    CUDA_OK(cudaMemcpyAsync(dStateDesh,  &hStateDesh,  sizeof(int), cudaMemcpyHostToDevice, sFUSE));

    int *dTop=nullptr; CUDA_OK(cudaMalloc(&dTop, sizeof(int)));

    fuseDecision<<<1, 128, 0, sFUSE>>>(
      dScores, K,
      dMeanHost,
      thrLuz, thrTemp, thrRuido, thrHum,
      dStateAlarm, dStateLuces, dStateVent, dStateDesh,
      dTop,
      dNewAlarm, dNewLuces, dNewVent, dNewDesh
    );

    int hTop=-1, hNewAlarm=0, hNewLuces=0, hNewVent=0, hNewDesh=0;
    CUDA_OK(cudaMemcpyAsync(&hTop,      dTop,       sizeof(int), cudaMemcpyDeviceToHost, sFUSE));
    CUDA_OK(cudaMemcpyAsync(&hNewAlarm, dNewAlarm,  sizeof(int), cudaMemcpyDeviceToHost, sFUSE));
    CUDA_OK(cudaMemcpyAsync(&hNewLuces, dNewLuces,  sizeof(int), cudaMemcpyDeviceToHost, sFUSE));
    CUDA_OK(cudaMemcpyAsync(&hNewVent,  dNewVent,   sizeof(int), cudaMemcpyDeviceToHost, sFUSE));
    CUDA_OK(cudaMemcpyAsync(&hNewDesh,  dNewDesh,   sizeof(int), cudaMemcpyDeviceToHost, sFUSE));

    CUDA_OK(cudaEventRecord(evFUSEStop, sFUSE));
    CUDA_OK(cudaStreamSynchronize(sFUSE));

    CUDA_OK(cudaEventRecord(evStop, 0));
    CUDA_OK(cudaEventSynchronize(evStop));

    float msNLU=0.f, msDATA=0.f, msFUSE=0.f, msTOTAL=0.f;
    CUDA_OK(cudaEventElapsedTime(&msNLU,  evNLUStart,  evNLUStop));
    CUDA_OK(cudaEventElapsedTime(&msDATA, evDATAStart, evDATAStop));
    CUDA_OK(cudaEventElapsedTime(&msFUSE, evFUSEStart, evFUSEStop));
    CUDA_OK(cudaEventElapsedTime(&msTOTAL, evStart, evStop));

    all_nlu_ms.push_back(msNLU);
    all_data_ms.push_back(msDATA);
    all_fuse_ms.push_back(msFUSE);
    all_total_ms.push_back(msTOTAL);

    // TOP-K de sugerencias
    auto top3 = topk_indices(hScores, K, 3);
    std::vector<std::string> topIntents;
    for (int idx : top3) {
      topIntents.push_back(intentNames[idx]);
    }

    int decision = 1;
    append_metrics_csv(metricsCsv, qi, q, topIntents, decision,
                      hNewAlarm, hNewLuces, hNewVent, hNewDesh,
                      hMean, msTOTAL, msNLU, msDATA, msFUSE);

    // Imprimir a consola para verificación
    printf("[Q%02d] \"%s\"\n", qi, q.c_str());
    printf("      Intent: %s | Alarm:%d Luces:%d Vent:%d Desh:%d\n", 
           intentNames[hTop], hNewAlarm, hNewLuces, hNewVent, hNewDesh);
    printf("      Sensores: Mov:%.2f Lux:%.1f Temp:%.1f°C Ruido:%.1fdB Hum:%.1f%%\n",
           hMean[0], hMean[1], hMean[2], hMean[3], hMean[4]*100);
    printf("      Latencias: Total:%.3fms (NLU:%.3f DATA:%.3f FUSE:%.3f)\n\n",
           msTOTAL, msNLU, msDATA, msFUSE);

    cudaFree(dTop);
    cudaFree(dMeanHost);
    cudaFree(dStateAlarm); cudaFree(dStateLuces); cudaFree(dStateVent); cudaFree(dStateDesh);
    cudaFree(dNewAlarm);   cudaFree(dNewLuces);   cudaFree(dNewVent);   cudaFree(dNewDesh);
  }

  double p50 = percentile(all_total_ms, 50.0);
  double p95 = percentile(all_total_ms, 95.0);
  double sum_ms = std::accumulate(all_total_ms.begin(), all_total_ms.end(), 0.0);
  double qps = (sum_ms>0.0) ? (1000.0 * Q / sum_ms) : 0.0;

  printf("═══════════════════════════════════════════════════════════\n");
  printf("                   RESUMEN FINAL\n");
  printf("═══════════════════════════════════════════════════════════\n");
  printf("Total queries: %d\n", Q);
  printf("Latencia p50: %.3f ms | p95: %.3f ms\n", p50, p95);
  printf("QPS aproximado: %.2f queries/segundo\n", qps);
  printf("Promedio por etapa:\n");
  printf("  - NLU:  %.3f ms\n", std::accumulate(all_nlu_ms.begin(),  all_nlu_ms.end(),  0.0)/Q);
  printf("  - DATA: %.3f ms\n", std::accumulate(all_data_ms.begin(), all_data_ms.end(), 0.0)/Q);
  printf("  - FUSE: %.3f ms\n", std::accumulate(all_fuse_ms.begin(), all_fuse_ms.end(), 0.0)/Q);
  printf("═══════════════════════════════════════════════════════════\n\n");

  printf("Archivos generados:\n");
  printf("  - metrics.csv (bitácora detallada por consulta)\n");
  printf("  - benchmark_results.csv (comparación 1-2-4-8 streams)\n\n");

  printf(" Siguiente paso : Ejecutar el script Python para generar gráficas\n");
  printf("   $ python generate_plots.py\n\n");

  metricsCsv.close();
  cudaFree(dQ); cudaFree(dVQ); cudaFree(dScores); cudaFree(dM);
  cudaFree(dX); cudaFree(dMean); cudaFree(dStd);
  cudaFreeHost(hQ); cudaFreeHost(hVQ); cudaFreeHost(hScores); cudaFreeHost(hX);
  cudaEventDestroy(evNLUStart); cudaEventDestroy(evNLUStop);
  cudaEventDestroy(evDATAStart); cudaEventDestroy(evDATAStop);
  cudaEventDestroy(evFUSEStart); cudaEventDestroy(evFUSEStop);
  cudaEventDestroy(evStart); cudaEventDestroy(evStop);
  cudaStreamDestroy(sNLU); cudaStreamDestroy(sDATA); cudaStreamDestroy(sFUSE);
  
  return 0;
}
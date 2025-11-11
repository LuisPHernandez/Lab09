// CC3086 - Lab 9: Chat-Box con IA (CUDA) 
// Compilar: nvcc -O3 -std=c++17 main.cu -o chatbox_cuda

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
#include <sstream>

// ======================== CONFIGURACIÓN ========================
constexpr int D = 8192;
constexpr int K = 5;
constexpr int TOPK = 3;
constexpr int MAX_QUERY = 512;
constexpr int C = 5;
constexpr int N = 1<<20;
constexpr int W = 1024;

#define CUDA_OK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line){
  if (code != cudaSuccess){
    fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

inline int ceilDiv(int a, int b){ return (a + b - 1) / b; }

// ======================== KERNELS ========================

__device__ __forceinline__
uint32_t hash3(uint8_t a, uint8_t b, uint8_t c){
  uint32_t h = 2166136261u;
  h = (h ^ a) * 16777619u;
  h = (h ^ b) * 16777619u;
  h = (h ^ c) * 16777619u;
  return h % D;
}

__global__
void tokenize3grams(const char* __restrict__ query, int n, float* __restrict__ vq){
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

__global__
void window_stats_last(const float* __restrict__ X, int N, int C, int W,
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

enum Intent {
  TOGGLE_ALARMA = 0,
  TOGGLE_LUCES = 1,
  TOGGLE_VENTILADOR = 2,
  TOGGLE_DESHUMIDIFICADOR = 3,
  CONSULTAR_ESTADO = 4
};

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
  if (threadIdx.x == 0){ topIdx = 0; }
  __syncthreads();
  if (threadIdx.x == 0){
    int best = 0;
    float bestv = scores[0];
    for (int k = 1; k < K; ++k){
      if (scores[k] > bestv){ bestv = scores[k]; best = k; }
    }
    topIdx = best;
  }
  __syncthreads();
  if (threadIdx.x == 0){
    *outTop = topIdx;
    float mov   = meanC[0];
    float luz   = meanC[1];
    float temp  = meanC[2];
    float ruido = meanC[3];
    float hum   = meanC[4];
    int sAlarm = *curAlarm, sLuces = *curLuces, sVent = *curVent, sDesh = *curDesh;
    int nAlarm = sAlarm, nLuces = sLuces, nVent = sVent, nDesh = sDesh;
    bool oscuro = (luz < thrLuz), caliente = (temp > thrTemp);
    bool ruidoso = (ruido > thrRuido), humedo = (hum > thrHum);
    bool hayMov = (mov >= 0.5f);
    switch (topIdx){
      case TOGGLE_ALARMA:
        nAlarm = (hayMov || ruidoso) ? 1 : (1 - sAlarm);
        break;
      case TOGGLE_LUCES:
        nLuces = oscuro ? 1 : (1 - sLuces);
        break;
      case TOGGLE_VENTILADOR:
        nVent = caliente ? 1 : (1 - sVent);
        break;
      case TOGGLE_DESHUMIDIFICADOR:
        nDesh = humedo ? 1 : (1 - sDesh);
        break;
      default: break;
    }
    *newAlarm = nAlarm; *newLuces = nLuces; *newVent = nVent; *newDesh = nDesh;
  }
}

// ======================== HOST HELPERS ========================

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
  for (auto& x : v) x /= n;
}

static void tokenize3grams_host(const std::string& s, std::vector<float>& out){
  if (s.size() < 3) return;
  for (size_t i = 0; i + 2 < s.size(); ++i){
    uint32_t idx = hash3_host((uint8_t)s[i], (uint8_t)s[i+1], (uint8_t)s[i+2]);
    out[idx] += 1.0f;
  }
}

static const std::array<std::vector<std::string>, K> TEMPLATES = {{
  {"activar alarma", "apagar alarma", "alarma on", "alarma off", "encender alarma"},
  {"prender luces", "apagar luces", "encender iluminacion", "luces afuera"},
  {"encender ventilador", "apagar ventilador", "activar ventilacion"},
  {"activar deshumidificador", "apagar deshumidificador", "reducir humedad"},
  {"como esta la casa", "estado del sistema", "resumen", "datos de sensores"}
}};

static const char* intentNames[K] = {
  "TOGGLE_ALARMA", "TOGGLE_LUCES", "TOGGLE_VENTILADOR",
  "TOGGLE_DESHUMIDIFICADOR", "CONSULTAR_ESTADO"
};

static void buildIntentMatrix(std::vector<float>& M){
  M.assign(K * D, 0.f);
  for (int k = 0; k < K; ++k){
    std::vector<float> row(D, 0.f);
    for (const auto& p : TEMPLATES[k]){
      std::vector<float> v(D, 0.f);
      tokenize3grams_host(p, v);
      l2normalize_host(v);
      for (int j = 0; j < D; ++j) row[j] += v[j];
    }
    float inv = TEMPLATES[k].empty() ? 1.f : (1.f / TEMPLATES[k].size());
    for (int j = 0; j < D; ++j) row[j] *= inv;
    {
      double acc=0.0; for (float x:row) acc += double(x)*double(x);
      float n = float(std::sqrt(acc) + 1e-12);
      for (float& x:row) x /= n;
    }
    for (int j = 0; j < D; ++j) M[k*D + j] = row[j];
  }
}

// TOP-K con índices
static std::vector<int> topk_indices(const float* scores, int K, int k){
  std::vector<int> idx(K);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin()+std::min(k,K), idx.end(),
                    [&](int a, int b){ return scores[a] > scores[b]; });
  idx.resize(std::min(k,K));
  return idx;
}

void synthSensors(std::vector<float>& X){
  X.resize(size_t(N)*C);
  srand(12345);
  int burstLen = 0;
  for (int i = 0; i < N; ++i){
    int motion = 0;
    if (burstLen > 0){ motion = 1; burstLen--; }
    else if ((rand()%10000)/10000.f < 0.001f){ burstLen = 50 + rand()%450; motion = 1; burstLen--; }
    float phase = (i%(1<<16))*(2.f*3.14159265f/float(1<<16));
    float lux = 50.f + 0.5f*(1.f+sinf(phase))*750.f + ((rand()%1000)/1000.f - 0.5f)*30.f;
    if (lux < 0.f) lux = 0.f;
    float temp = 27.0f + ((rand()%1000)/1000.f - 0.5f)*6.f;
    float noise = 35.f + (rand()%1000)/1000.f*10.f;
    if ((rand()%100000)/100000.f < 0.0008f) noise = 65.f + (rand()%1000)/1000.f*20.f;
    float hum = 55.f + ((rand()%1000)/1000.f - 0.5f)*0.2f;
    X[i*C + 0] = float(motion);
    X[i*C + 1] = lux;
    X[i*C + 2] = temp;
    X[i*C + 3] = noise;
    X[i*C + 4] = hum;
  }
}

static double percentile(std::vector<float> v, double p){
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  double r = (p/100.0)*(v.size()-1);
  size_t lo = (size_t)std::floor(r), hi = (size_t)std::ceil(r);
  if (lo == hi) return v[lo];
  double w = r - lo;
  return (1.0 - w)*v[lo] + w*v[hi];
}

// ======================== QUERY CONTEXT ========================
struct QueryContext {
  char *d_query;
  float *d_vq, *d_scores, *h_scores;
  float *d_X, *d_mean, *d_std, *h_mean;
  int *d_stateAlarm, *d_stateLuces, *d_stateVent, *d_stateDesh;
  int *d_newAlarm, *d_newLuces, *d_newVent, *d_newDesh;
  int *d_top;
  float *d_meanC;
  cudaEvent_t ev_start, ev_end;
  cudaEvent_t ev_nlu_start, ev_nlu_end;
  cudaEvent_t ev_data_start, ev_data_end;
  cudaEvent_t ev_fuse_start, ev_fuse_end;
};

void allocateQueryContext(QueryContext& ctx){
  CUDA_OK(cudaMalloc(&ctx.d_query, MAX_QUERY));
  CUDA_OK(cudaMalloc(&ctx.d_vq, D*sizeof(float)));
  CUDA_OK(cudaMalloc(&ctx.d_scores, K*sizeof(float)));
  CUDA_OK(cudaHostAlloc(&ctx.h_scores, K*sizeof(float), cudaHostAllocDefault));
  CUDA_OK(cudaMalloc(&ctx.d_X, size_t(N)*C*sizeof(float)));
  CUDA_OK(cudaMalloc(&ctx.d_mean, C*sizeof(float)));
  CUDA_OK(cudaMalloc(&ctx.d_std, C*sizeof(float)));
  CUDA_OK(cudaHostAlloc(&ctx.h_mean, C*sizeof(float), cudaHostAllocDefault));
  CUDA_OK(cudaMalloc(&ctx.d_meanC, C*sizeof(float)));
  CUDA_OK(cudaMalloc(&ctx.d_stateAlarm, sizeof(int)));
  CUDA_OK(cudaMalloc(&ctx.d_stateLuces, sizeof(int)));
  CUDA_OK(cudaMalloc(&ctx.d_stateVent, sizeof(int)));
  CUDA_OK(cudaMalloc(&ctx.d_stateDesh, sizeof(int)));
  CUDA_OK(cudaMalloc(&ctx.d_newAlarm, sizeof(int)));
  CUDA_OK(cudaMalloc(&ctx.d_newLuces, sizeof(int)));
  CUDA_OK(cudaMalloc(&ctx.d_newVent, sizeof(int)));
  CUDA_OK(cudaMalloc(&ctx.d_newDesh, sizeof(int)));
  CUDA_OK(cudaMalloc(&ctx.d_top, sizeof(int)));
  CUDA_OK(cudaEventCreate(&ctx.ev_start));
  CUDA_OK(cudaEventCreate(&ctx.ev_end));
  CUDA_OK(cudaEventCreate(&ctx.ev_nlu_start));
  CUDA_OK(cudaEventCreate(&ctx.ev_nlu_end));
  CUDA_OK(cudaEventCreate(&ctx.ev_data_start));
  CUDA_OK(cudaEventCreate(&ctx.ev_data_end));
  CUDA_OK(cudaEventCreate(&ctx.ev_fuse_start));
  CUDA_OK(cudaEventCreate(&ctx.ev_fuse_end));
}

void freeQueryContext(QueryContext& ctx){
  cudaFree(ctx.d_query); cudaFree(ctx.d_vq); cudaFree(ctx.d_scores);
  cudaFree(ctx.d_X); cudaFree(ctx.d_mean); cudaFree(ctx.d_std); cudaFree(ctx.d_meanC);
  cudaFree(ctx.d_stateAlarm); cudaFree(ctx.d_stateLuces);
  cudaFree(ctx.d_stateVent); cudaFree(ctx.d_stateDesh);
  cudaFree(ctx.d_newAlarm); cudaFree(ctx.d_newLuces);
  cudaFree(ctx.d_newVent); cudaFree(ctx.d_newDesh); cudaFree(ctx.d_top);
  cudaFreeHost(ctx.h_scores); cudaFreeHost(ctx.h_mean);
  cudaEventDestroy(ctx.ev_start); cudaEventDestroy(ctx.ev_end);
  cudaEventDestroy(ctx.ev_nlu_start); cudaEventDestroy(ctx.ev_nlu_end);
  cudaEventDestroy(ctx.ev_data_start); cudaEventDestroy(ctx.ev_data_end);
  cudaEventDestroy(ctx.ev_fuse_start); cudaEventDestroy(ctx.ev_fuse_end);
}

// ======================== RESULTADOS ========================
struct QueryResult {
  std::string query;
  int top_intent;
  std::vector<int> topk_intents;
  int decision;
  int new_alarm, new_luces, new_vent, new_desh;
  float mean_sensors[C];
  float lat_total_ms, lat_nlu_ms, lat_data_ms, lat_fuse_ms;
};

// ======================== BENCHMARK ========================
struct BenchResult {
  int num_streams;
  double p50_ms, p95_ms, qps;
  double avg_nlu_ms, avg_data_ms, avg_fuse_ms;
  std::vector<QueryResult> query_results;
};

BenchResult runBenchmark(int num_streams, const std::vector<std::string>& QUERIES, 
                         float *d_M, const std::vector<float>& h_sensorData){
  const int Q = (int)QUERIES.size();
  
  std::vector<cudaStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; ++i) CUDA_OK(cudaStreamCreate(&streams[i]));
  
  std::vector<QueryContext> contexts(num_streams);
  for (auto& ctx : contexts) allocateQueryContext(ctx);
  
  int h_alarm = 0, h_luces = 0, h_vent = 0, h_desh = 0;
  
  std::vector<float> all_total, all_nlu, all_data, all_fuse;
  std::vector<QueryResult> results;
  
  for (int qi = 0; qi < Q; ++qi){
    int sid = qi % num_streams;
    cudaStream_t s = streams[sid];
    QueryContext& ctx = contexts[sid];
    
    std::string q = QUERIES[qi];
    std::transform(q.begin(), q.end(), q.begin(), [](unsigned char c){ return std::tolower(c); });
    int qn = std::min<int>((int)q.size(), MAX_QUERY);
    
    char h_query[MAX_QUERY] = {0};
    memcpy(h_query, q.data(), qn);
    
    CUDA_OK(cudaEventRecord(ctx.ev_start, s));
    
    // === NLU ===
    CUDA_OK(cudaEventRecord(ctx.ev_nlu_start, s));
    CUDA_OK(cudaMemsetAsync(ctx.d_vq, 0, D*sizeof(float), s));
    CUDA_OK(cudaMemcpyAsync(ctx.d_query, h_query, MAX_QUERY, cudaMemcpyHostToDevice, s));
    tokenize3grams<<<ceilDiv(qn,256), 256, 0, s>>>(ctx.d_query, qn, ctx.d_vq);
    l2normalize<<<1, 256, 0, s>>>(ctx.d_vq, D);
    matvecDotCos<<<ceilDiv(K,128), 128, 0, s>>>(d_M, ctx.d_vq, ctx.d_scores, K, D);
    CUDA_OK(cudaMemcpyAsync(ctx.h_scores, ctx.d_scores, K*sizeof(float), cudaMemcpyDeviceToHost, s));
    CUDA_OK(cudaEventRecord(ctx.ev_nlu_end, s));
    
    // === DATA ===
    CUDA_OK(cudaEventRecord(ctx.ev_data_start, s));
    CUDA_OK(cudaMemcpyAsync(ctx.d_X, h_sensorData.data(), size_t(N)*C*sizeof(float), cudaMemcpyHostToDevice, s));
    window_stats_last<<<C, 256, 0, s>>>(ctx.d_X, N, C, W, ctx.d_mean, ctx.d_std);
    CUDA_OK(cudaMemcpyAsync(ctx.d_meanC, ctx.d_mean, C*sizeof(float), cudaMemcpyDeviceToDevice, s));
    CUDA_OK(cudaMemcpyAsync(ctx.h_mean, ctx.d_mean, C*sizeof(float), cudaMemcpyDeviceToHost, s));
    CUDA_OK(cudaEventRecord(ctx.ev_data_end, s));
    
    // === FUSE ===
    CUDA_OK(cudaEventRecord(ctx.ev_fuse_start, s));
    CUDA_OK(cudaMemcpyAsync(ctx.d_stateAlarm, &h_alarm, sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_OK(cudaMemcpyAsync(ctx.d_stateLuces, &h_luces, sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_OK(cudaMemcpyAsync(ctx.d_stateVent, &h_vent, sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_OK(cudaMemcpyAsync(ctx.d_stateDesh, &h_desh, sizeof(int), cudaMemcpyHostToDevice, s));
    
    fuseDecision<<<1, 128, 0, s>>>(
      ctx.d_scores, K, ctx.d_meanC,
      250.f, 27.f, 60.f, 0.60f,
      ctx.d_stateAlarm, ctx.d_stateLuces, ctx.d_stateVent, ctx.d_stateDesh,
      ctx.d_top,
      ctx.d_newAlarm, ctx.d_newLuces, ctx.d_newVent, ctx.d_newDesh
    );
    
    int h_top, h_new[4];
    CUDA_OK(cudaMemcpyAsync(&h_top, ctx.d_top, sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_OK(cudaMemcpyAsync(&h_new[0], ctx.d_newAlarm, sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_OK(cudaMemcpyAsync(&h_new[1], ctx.d_newLuces, sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_OK(cudaMemcpyAsync(&h_new[2], ctx.d_newVent, sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_OK(cudaMemcpyAsync(&h_new[3], ctx.d_newDesh, sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_OK(cudaEventRecord(ctx.ev_fuse_end, s));
    
    CUDA_OK(cudaEventRecord(ctx.ev_end, s));
    CUDA_OK(cudaStreamSynchronize(s));
    
    // Guardar resultado
    QueryResult qr;
    qr.query = QUERIES[qi];
    qr.top_intent = h_top;
    qr.topk_intents = topk_indices(ctx.h_scores, K, TOPK);
    qr.decision = (h_new[0] != h_alarm || h_new[1] != h_luces || h_new[2] != h_vent || h_new[3] != h_desh) ? 1 : 0;
    qr.new_alarm = h_new[0]; qr.new_luces = h_new[1]; qr.new_vent = h_new[2]; qr.new_desh = h_new[3];
    for (int i = 0; i < C; ++i) qr.mean_sensors[i] = ctx.h_mean[i];
    
    float ms_total, ms_nlu, ms_data, ms_fuse;
    CUDA_OK(cudaEventElapsedTime(&ms_total, ctx.ev_start, ctx.ev_end));
    CUDA_OK(cudaEventElapsedTime(&ms_nlu, ctx.ev_nlu_start, ctx.ev_nlu_end));
    CUDA_OK(cudaEventElapsedTime(&ms_data, ctx.ev_data_start, ctx.ev_data_end));
    CUDA_OK(cudaEventElapsedTime(&ms_fuse, ctx.ev_fuse_start, ctx.ev_fuse_end));
    
    qr.lat_total_ms = ms_total;
    qr.lat_nlu_ms = ms_nlu;
    qr.lat_data_ms = ms_data;
    qr.lat_fuse_ms = ms_fuse;
    
    results.push_back(qr);
    
    all_total.push_back(ms_total);
    all_nlu.push_back(ms_nlu);
    all_data.push_back(ms_data);
    all_fuse.push_back(ms_fuse);
    
    h_alarm = h_new[0]; h_luces = h_new[1]; h_vent = h_new[2]; h_desh = h_new[3];
  }
  
  for (auto& ctx : contexts) freeQueryContext(ctx);
  for (auto& s : streams) CUDA_OK(cudaStreamDestroy(s));
  
  BenchResult res;
  res.num_streams = num_streams;
  res.p50_ms = percentile(all_total, 50.0);
  res.p95_ms = percentile(all_total, 95.0);
  double sum_ms = std::accumulate(all_total.begin(), all_total.end(), 0.0);
  res.qps = (sum_ms > 0) ? (1000.0 * Q / sum_ms) : 0.0;
  res.avg_nlu_ms = std::accumulate(all_nlu.begin(), all_nlu.end(), 0.0) / Q;
  res.avg_data_ms = std::accumulate(all_data.begin(), all_data.end(), 0.0) / Q;
  res.avg_fuse_ms = std::accumulate(all_fuse.begin(), all_fuse.end(), 0.0) / Q;
  res.query_results = results;
  
  return res;
}

// ======================== MAIN ========================
int main(){
  printf("╔═══════════════════════════════════════════════════════════╗\n");
  printf("║  LAB 9 - CHAT-BOX CUDA CON MÚLTIPLES STREAMS             ║\n");
  printf("║  Smart Home System                                        ║\n");
  printf("╚═══════════════════════════════════════════════════════════╝\n\n");
  
  // Preparar matriz de intenciones
  printf("[1/4] Construyendo matriz de intenciones desde templates...\n");
  std::vector<float> h_M;
  buildIntentMatrix(h_M);
  float *d_M = nullptr;
  CUDA_OK(cudaMalloc(&d_M, K*D*sizeof(float)));
  CUDA_OK(cudaMemcpy(d_M, h_M.data(), K*D*sizeof(float), cudaMemcpyHostToDevice));
  printf("       Matriz M [%dx%d] construida desde %d intenciones\n\n", K, D, K);
  
  // Generar datos de sensores
  printf("[2/4] Generando datos sintéticos de sensores...\n");
  std::vector<float> h_sensorData;
  synthSensors(h_sensorData);
  printf("       Generados %d samples × %d canales (%.2f MB)\n\n", 
         N, C, (N*C*sizeof(float))/(1024.0*1024.0));
  
  // Queries de prueba
  std::vector<std::string> QUERIES = {
    "activa la alarma", "apaga la alarma",
    "enciende luces exteriores", "apaga luces",
    "activa el ventilador", "apaga el ventilador",
    "activa el deshumidificador", "apaga deshumidificador",
    "consulta estado", "muestra mediciones",
    "enciende luces si esta oscuro", "estado del sistema",
    "alarma on", "ventilador off", "como esta la casa"
  };
  
  printf("[3/4] Ejecutando benchmark con múltiples configuraciones...\n\n");
  
  // Benchmark con diferentes configuraciones
  std::vector<int> configs = {1, 2, 4, 8};
  std::vector<BenchResult> bench_results;
  
  for (int ns : configs){
    printf("   → Configuración: %d stream(s)\n", ns);
    BenchResult r = runBenchmark(ns, QUERIES, d_M, h_sensorData);
    bench_results.push_back(r);
    printf("     p50=%.3fms | p95=%.3fms | QPS=%.2f\n", r.p50_ms, r.p95_ms, r.qps);
    printf("     Etapas: NLU=%.3fms | DATA=%.3fms | FUSE=%.3fms\n\n", 
           r.avg_nlu_ms, r.avg_data_ms, r.avg_fuse_ms);
  }
  
  printf("[4/4] Generando archivos de salida...\n");
  
  // Guardar benchmark_results.csv
  std::ofstream csv("benchmark_results.csv");
  csv << "num_streams,p50_ms,p95_ms,qps,avg_nlu_ms,avg_data_ms,avg_fuse_ms\n";
  for (const auto& r : bench_results){
    csv << r.num_streams << "," << std::fixed << std::setprecision(3)
        << r.p50_ms << "," << r.p95_ms << "," << r.qps << ","
        << r.avg_nlu_ms << "," << r.avg_data_ms << "," << r.avg_fuse_ms << "\n";
  }
  csv.close();
  printf("      benchmark_results.csv\n");
  
  // Guardar metrics.csv con TODOS los detalles de la primera configuración
  std::ofstream mcsv("metrics.csv");
  mcsv << "query_id,query_text,top_intent,top_k_intents,decision,"
          "new_alarm,new_luces,new_vent,new_desh,"
          "mean_mov,mean_lux,mean_temp,mean_ruido,mean_hum,"
          "lat_total_ms,lat_nlu_ms,lat_data_ms,lat_fuse_ms\n";
  
  // Usar resultados de la primera configuración (1 stream) para metrics detallado
  const auto& first_bench = bench_results[0];
  for (size_t qi = 0; qi < first_bench.query_results.size(); ++qi){
    const auto& qr = first_bench.query_results[qi];
    
    // Construir string con top-k intents
    std::stringstream topk_str;
    topk_str << "\"";
    for (size_t i = 0; i < qr.topk_intents.size(); ++i){
      topk_str << intentNames[qr.topk_intents[i]];
      if (i < qr.topk_intents.size() - 1) topk_str << ";";
    }
    topk_str << "\"";
    
    mcsv << qi << ",\"" << qr.query << "\","
         << intentNames[qr.top_intent] << ","
         << topk_str.str() << ","
         << qr.decision << ","
         << qr.new_alarm << "," << qr.new_luces << "," 
         << qr.new_vent << "," << qr.new_desh << ","
         << std::fixed << std::setprecision(3)
         << qr.mean_sensors[0] << "," << qr.mean_sensors[1] << ","
         << qr.mean_sensors[2] << "," << qr.mean_sensors[3] << ","
         << qr.mean_sensors[4] << ","
         << qr.lat_total_ms << "," << qr.lat_nlu_ms << ","
         << qr.lat_data_ms << "," << qr.lat_fuse_ms << "\n";
  }
  mcsv.close();
  printf("      ✓ metrics.csv\n");
  
  // Resumen en consola
  printf("\n╔═══════════════════════════════════════════════════════════╗\n");
  printf("║   BENCHMARK COMPLETADO                                  ║\n");
  printf("╚═══════════════════════════════════════════════════════════╝\n\n");
  
  printf("Resumen de resultados:\n");
  printf("─────────────────────────────────────────────────────────────\n");
  printf(" Streams │   p50 (ms) │   p95 (ms) │    QPS  │  Speedup\n");
  printf("─────────────────────────────────────────────────────────────\n");
  
  double baseline_qps = bench_results[0].qps;
  for (const auto& r : bench_results){
    double speedup = r.qps / baseline_qps;
    printf("   %2d    │   %7.3f  │   %7.3f  │  %6.2f │  %5.2fx\n",
           r.num_streams, r.p50_ms, r.p95_ms, r.qps, speedup);
  }
  printf("─────────────────────────────────────────────────────────────\n\n");
  
  printf("Desglose promedio por etapa (config: 1 stream):\n");
  printf("  • NLU:  %.3f ms (%.1f%%)\n", 
         bench_results[0].avg_nlu_ms,
         100.0 * bench_results[0].avg_nlu_ms / bench_results[0].p50_ms);
  printf("  • DATA: %.3f ms (%.1f%%)\n",
         bench_results[0].avg_data_ms,
         100.0 * bench_results[0].avg_data_ms / bench_results[0].p50_ms);
  printf("  • FUSE: %.3f ms (%.1f%%)\n\n",
         bench_results[0].avg_fuse_ms,
         100.0 * bench_results[0].avg_fuse_ms / bench_results[0].p50_ms);
  
  printf("Ejemplos de queries procesadas:\n");
  printf("─────────────────────────────────────────────────────────────\n");
  for (size_t i = 0; i < std::min<size_t>(5, first_bench.query_results.size()); ++i){
    const auto& qr = first_bench.query_results[i];
    printf("[%zu] \"%s\"\n", i, qr.query.c_str());
    printf("    Intent: %s (Top-3: ", intentNames[qr.top_intent]);
    for (size_t j = 0; j < qr.topk_intents.size(); ++j){
      printf("%s", intentNames[qr.topk_intents[j]]);
      if (j < qr.topk_intents.size()-1) printf(", ");
    }
    printf(")\n");
    printf("    Sensores: mov=%.1f lux=%.0f temp=%.1f°C ruido=%.0fdB hum=%.0f%%\n",
           qr.mean_sensors[0], qr.mean_sensors[1], qr.mean_sensors[2],
           qr.mean_sensors[3], qr.mean_sensors[4]);
    printf("    Acción: Alarm=%d Luces=%d Vent=%d Desh=%d (cambio=%s)\n",
           qr.new_alarm, qr.new_luces, qr.new_vent, qr.new_desh,
           qr.decision ? "SÍ" : "NO");
    printf("    Latencia: %.3fms (NLU=%.3f DATA=%.3f FUSE=%.3f)\n\n",
           qr.lat_total_ms, qr.lat_nlu_ms, qr.lat_data_ms, qr.lat_fuse_ms);
  }
  printf("─────────────────────────────────────────────────────────────\n\n");
  
  printf("Archivos generados:\n");
  printf("  1. benchmark_results.csv - Comparación de 1/2/4/8 streams\n");
  printf("  2. metrics.csv          - Métricas detalladas por query\n\n");
  
  printf("Siguiente paso:\n");
  printf("   python generate_plots.py\n");
  printf("   Revisar gráficas generadas (PNG)\n\n");
  
  cudaFree(d_M);
  
  printf("╔═══════════════════════════════════════════════════════════╗\n");
  printf("║  Chat-Box CUDA finalizado exitosamente                   ║\n");
  printf("╚═══════════════════════════════════════════════════════════╝\n");
  
  return 0;
}
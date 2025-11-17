//# Si usas MSYS2 o MinGW instalar libreria curl :: pacman -S mingw-w64-x86_64-curl
// Compilar: nvcc -O3 -std=c++17 main.cu -o chatbox_cuda -lcurl

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
#include <iostream>
#include <sstream>
#include <curl/curl.h>

// ------------------------ Utilidades ------------------------

// Callback para CURL
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Parser CSV simple
std::vector<std::vector<float>> parseCSV(const std::string& csv_data) {
    std::vector<std::vector<float>> result;
    std::istringstream stream(csv_data);
    std::string line;
    
    // Saltar encabezado
    std::getline(stream, line);
    
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        std::vector<float> row;
        std::istringstream lineStream(line);
        std::string cell;
        
        while (std::getline(lineStream, cell, ',')) {
            try {
                row.push_back(std::stof(cell));
            } catch (...) {
                row.push_back(0.0f);
            }
        }
        if (!row.empty()) result.push_back(row);
    }
    return result;
}

// Descarga Google Sheets como CSV
bool downloadGoogleSheet(const std::string& sheet_url, std::vector<std::vector<float>>& data) {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;
    
    // Convertir URL de Google Sheets a formato CSV export
    std::string csv_url = sheet_url;
    size_t pos = csv_url.find("/edit");
    if (pos != std::string::npos) {
        csv_url = csv_url.substr(0, pos) + "/export?format=csv";
    }
    
    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, csv_url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        
        if(res == CURLE_OK) {
            data = parseCSV(readBuffer);
            return !data.empty();
        }
    }
    return false;
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
  ofs << "query_id,query_text,top_intent,decision,new_alarm,new_luces,new_vent,new_desh,"
         "mean_mov,mean_lux,mean_temp,mean_ruido,mean_hum,"
         "lat_total_ms,lat_nlu_ms,lat_data_ms,lat_fuse_ms\n";
}

static void append_metrics_csv(std::ofstream& ofs,
                               int qid, const std::string& qtext,
                               const std::string& top_intent, int decision,
                               int new_alarm, int new_luces, int new_vent, int new_desh,
                               const float meanC[5],
                               float t_total, float t_nlu, float t_data, float t_fuse){
  ofs << qid << ","
      << "\"" << qtext << "\"" << ","
      << top_intent << ","
      << decision << ","
      << new_alarm << ","
      << new_luces << ","
      << new_vent  << ","
      << new_desh  << ","
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

// Sensores (5 canales)
constexpr int C = 5;
constexpr int W = 1024;  // Ventana de últimas muestras

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
  int actual_w = N - start;
  
  for (int i = threadIdx.x; i < actual_w; i += blockDim.x){
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
    float m = ssum[0] / actual_w;
    float var = fmaxf(ssum2[0]/actual_w - m*m, 0.f);
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
  {"activar alarma", "apagar alarma", "alarma on", "alarma off", "encender alarma", "desactivar alarma"},
  {"prender luces", "apagar luces", "encender iluminacion", "luces afuera", "luz exterior"},
  {"encender ventilador", "apagar ventilador", "activar ventilacion", "ventilador on", "ventilador off"},
  {"activar deshumidificador", "apagar deshumidificador", "deshumidificador on", "reducir humedad"},
  {"como esta la casa", "estado del sistema", "que esta encendido", "resumen", "status general"}
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

  if (threadIdx.x == 0){ topIdx = 0; }
  __syncthreads();

  if (threadIdx.x == 0){
    int best = 0;
    float bestv = scores[0];
    for (int k = 1; k < K; ++k){
      float s = scores[k];
      if (s > bestv){ bestv = s; best = k; }
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

void initIntentPrototypes(std::vector<float>& M){
  buildIntentMatrixFromTemplates(M);
}

// ------------------------ Main ------------------------
int main(){
  std::cout << "=== Chat-Box Interactivo con CUDA ===\n\n";
  
  // Solicitar URL de Google Sheets
  std::string sheet_url;
  std::cout << "Ingresa la URL de tu Google Sheets con los datos de sensores:\n";
  std::cout << "(Formato: movimiento, luz, temperatura, ruido, humedad)\n";
  std::cout << "URL: ";
  std::getline(std::cin, sheet_url);
  
  // Descargar datos de sensores
  std::vector<std::vector<float>> sensorData;
  std::cout << "\nDescargando datos de sensores...\n";
  
  if (!downloadGoogleSheet(sheet_url, sensorData)) {
    std::cerr << "Error: No se pudo descargar el Google Sheet.\n";
    std::cerr << "Asegúrate de que:\n";
    std::cerr << "1. El link sea público o compartido\n";
    std::cerr << "2. El formato sea: movimiento, luz, temp, ruido, humedad\n";
    return 1;
  }
  
  int N = (int)sensorData.size();
  std::cout << "✓ Descargados " << N << " registros de sensores\n\n";
  
  if (N == 0 || sensorData[0].size() < C) {
    std::cerr << "Error: El sheet debe tener al menos " << C << " columnas\n";
    return 1;
  }
  
  // Inicializar CUDA
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

  // Inicializar matriz de intenciones
  std::vector<float> hM;
  initIntentPrototypes(hM);
  float *dM=nullptr;
  CUDA_OK(cudaMalloc(&dM, K*D*sizeof(float)));
  CUDA_OK(cudaMemcpy(dM, hM.data(), K*D*sizeof(float), cudaMemcpyHostToDevice));

  // Buffers para query
  char *hQ=nullptr;
  CUDA_OK(cudaHostAlloc(&hQ, MAX_QUERY, cudaHostAllocDefault));
  char *dQ=nullptr;
  CUDA_OK(cudaMalloc(&dQ, MAX_QUERY));

  // CSV de métricas
  std::ofstream metricsCsv;
  open_metrics_csv(metricsCsv, "metrics.csv");

  std::vector<float> all_total_ms, all_nlu_ms, all_data_ms, all_fuse_ms;

  float *hVQ=nullptr, *dVQ=nullptr, *dScores=nullptr, *hScores=nullptr;
  CUDA_OK(cudaHostAlloc(&hVQ, D*sizeof(float), cudaHostAllocDefault));
  CUDA_OK(cudaHostAlloc(&hScores, K*sizeof(float), cudaHostAllocDefault));
  CUDA_OK(cudaMalloc(&dVQ, D*sizeof(float)));
  CUDA_OK(cudaMalloc(&dScores, K*sizeof(float)));

  // Preparar datos de sensores en formato plano
  float *hX=nullptr;
  CUDA_OK(cudaHostAlloc(&hX, size_t(N)*C*sizeof(float), cudaHostAllocDefault));
  
  for (int i = 0; i < N; ++i) {
    for (int c = 0; c < C && c < (int)sensorData[i].size(); ++c) {
      hX[i*C + c] = sensorData[i][c];
    }
  }

  float *dX=nullptr, *dMean=nullptr, *dStd=nullptr;
  float hMean[C]={0}, hStd[C]={0};
  CUDA_OK(cudaMalloc(&dX, size_t(N)*C*sizeof(float)));
  CUDA_OK(cudaMalloc(&dMean, C*sizeof(float)));
  CUDA_OK(cudaMalloc(&dStd,  C*sizeof(float)));
  
  // Copiar datos de sensores a GPU (una sola vez)
  CUDA_OK(cudaMemcpy(dX, hX, size_t(N)*C*sizeof(float), cudaMemcpyHostToDevice));

  // Estados de dispositivos
  int hStateAlarm = 0, hStateLuces = 0, hStateVent = 0, hStateDesh = 0;

  static const char* intentNames[K] = {
    "TOGGLE_ALARMA",
    "TOGGLE_LUCES",
    "TOGGLE_VENTILADOR",
    "TOGGLE_DESHUMIDIFICADOR",
    "CONSULTAR_ESTADO"
  };

  std::cout << "Sistema listo. Comandos disponibles:\n";
  std::cout << "  - 'activa/apaga la alarma'\n";
  std::cout << "  - 'enciende/apaga las luces'\n";
  std::cout << "  - 'activa/apaga el ventilador'\n";
  std::cout << "  - 'activa/apaga el deshumidificador'\n";
  std::cout << "  - 'consulta estado'\n";
  std::cout << "  - 'salir' para terminar\n\n";

  int qi = 0;
  while (true) {
    std::cout << "Comando> ";
    std::string input;
    std::getline(std::cin, input);
    
    if (input.empty()) continue;
    if (input == "salir" || input == "exit" || input == "quit") break;
    
    std::string q = input;
    std::transform(q.begin(), q.end(), q.begin(),
                  [](unsigned char c){ return std::tolower(c); });
    
    const int qn = std::min<int>((int)q.size(), MAX_QUERY);
    memset(hQ, 0, MAX_QUERY);
    memcpy(hQ, q.data(), qn);

    CUDA_OK(cudaEventRecord(evStart, 0));

    // ===================== NLU =====================
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

    // ===================== DATA =====================
    CUDA_OK(cudaEventRecord(evDATAStart, sDATA));
    window_stats_last<<<C, 256, 0, sDATA>>>(dX, N, C, W, dMean, dStd);
    CUDA_OK(cudaMemcpyAsync(hMean, dMean, C*sizeof(float), cudaMemcpyDeviceToHost, sDATA));
    CUDA_OK(cudaMemcpyAsync(hStd,  dStd,  C*sizeof(float), cudaMemcpyDeviceToHost, sDATA));
    CUDA_OK(cudaEventRecord(evDATAStop, sDATA));
    CUDA_OK(cudaStreamSynchronize(sDATA));

    // ===================== FUSE =====================
    CUDA_OK(cudaEventRecord(evFUSEStart, sFUSE));

    float *dMeanHost=nullptr;
    CUDA_OK(cudaMalloc(&dMeanHost, C*sizeof(float)));
    CUDA_OK(cudaMemcpyAsync(dMeanHost, hMean, C*sizeof(float), cudaMemcpyHostToDevice, sFUSE));

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
    CUDA_OK(cudaStreamSynchronize(sFUSE));

    int *dTop=nullptr;
    CUDA_OK(cudaMalloc(&dTop, sizeof(int)));

    fuseDecision<<<1, 128, 0, sFUSE>>>(
      dScores, K, dMeanHost,
      thrLuz, thrTemp, thrRuido, thrHum,
      dStateAlarm, dStateLuces, dStateVent, dStateDesh,
      dTop, dNewAlarm, dNewLuces, dNewVent, dNewDesh
    );

    int hTop=-1, hNewAlarm=0, hNewLuces=0, hNewVent=0, hNewDesh=0;
    CUDA_OK(cudaMemcpyAsync(&hTop,      dTop,       sizeof(int), cudaMemcpyDeviceToHost, sFUSE));
    CUDA_OK(cudaMemcpyAsync(&hNewAlarm, dNewAlarm,  sizeof(int), cudaMemcpyDeviceToHost, sFUSE));
    CUDA_OK(cudaMemcpyAsync(&hNewLuces, dNewLuces,  sizeof(int), cudaMemcpyDeviceToHost, sFUSE));
    CUDA_OK(cudaMemcpyAsync(&hNewVent,  dNewVent,   sizeof(int), cudaMemcpyDeviceToHost, sFUSE));
    CUDA_OK(cudaMemcpyAsync(&hNewDesh,  dNewDesh,   sizeof(int), cudaMemcpyDeviceToHost, sFUSE));

    CUDA_OK(cudaEventRecord(evFUSEStop, sFUSE));
    CUDA_OK(cudaStreamSynchronize(sFUSE));

    int prevAlarm = hStateAlarm, prevLuces = hStateLuces, prevVent = hStateVent, prevDesh = hStateDesh;

    hStateAlarm = hNewAlarm;
    hStateLuces = hNewLuces;
    hStateVent  = hNewVent;
    hStateDesh  = hNewDesh;

    CUDA_OK(cudaEventRecord(evStop, 0));
    CUDA_OK(cudaEventSynchronize(evStop));

    // Métricas
    float msNLU=0.f, msDATA=0.f, msFUSE=0.f, msTOTAL=0.f;
    CUDA_OK(cudaEventElapsedTime(&msNLU,  evNLUStart,  evNLUStop));
    CUDA_OK(cudaEventElapsedTime(&msDATA, evDATAStart, evDATAStop));
    CUDA_OK(cudaEventElapsedTime(&msFUSE, evFUSEStart, evFUSEStop));
    CUDA_OK(cudaEventElapsedTime(&msTOTAL, evStart, evStop));

    all_nlu_ms.push_back(msNLU);
    all_data_ms.push_back(msDATA);
    all_fuse_ms.push_back(msFUSE);
    all_total_ms.push_back(msTOTAL);

    // Mostrar resultados
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Intención detectada: " << intentNames[hTop] << "\n";
    std::cout << "Confianza: " << std::fixed << std::setprecision(2) << (hScores[hTop] * 100) << "%\n";
    
    std::cout << "\n📊 Sensores (promedio últimas " << std::min(W, N) << " muestras):\n";
    std::cout << "  • Movimiento: " << std::setprecision(3) << hMean[0] << "\n";
    std::cout << "  • Luz: " << std::setprecision(1) << hMean[1] << " lux\n";
    std::cout << "  • Temperatura: " << std::setprecision(1) << hMean[2] << " °C\n";
    std::cout << "  • Ruido: " << std::setprecision(1) << hMean[3] << " dB\n";
    std::cout << "  • Humedad: " << std::setprecision(1) << (hMean[4]*100) << "%\n";
    
    std::cout << "\n🏠 Estado de dispositivos:\n";
    
    auto printDevice = [](const char* name, int prev, int curr) {
      if (prev != curr) {
        std::cout << "  • " << name << ": " << (prev ? "ON" : "OFF") 
                  << " → " << (curr ? "✓ ON" : "✗ OFF") << " [CAMBIO]\n";
      } else {
        std::cout << "  • " << name << ": " << (curr ? "ON" : "OFF") << "\n";
      }
    };
    
    printDevice("Alarma        ", prevAlarm, hNewAlarm);
    printDevice("Luces         ", prevLuces, hNewLuces);
    printDevice("Ventilador    ", prevVent,  hNewVent);
    printDevice("Deshumidif.   ", prevDesh,  hNewDesh);
    
    std::cout << "\n⚡ Latencia: " << std::setprecision(2) << msTOTAL << " ms "
              << "(NLU:" << msNLU << " + DATA:" << msDATA << " + FUSE:" << msFUSE << ")\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    // Guardar en CSV
    int decision = (prevAlarm != hNewAlarm) || (prevLuces != hNewLuces) || 
                   (prevVent != hNewVent) || (prevDesh != hNewDesh);
    append_metrics_csv(metricsCsv, qi, q, intentNames[hTop], decision,
                      hNewAlarm, hNewLuces, hNewVent, hNewDesh,
                      hMean, msTOTAL, msNLU, msDATA, msFUSE);

    // Limpieza
    cudaFree(dTop);
    cudaFree(dMeanHost);
    cudaFree(dStateAlarm); cudaFree(dStateLuces); cudaFree(dStateVent); cudaFree(dStateDesh);
    cudaFree(dNewAlarm);   cudaFree(dNewLuces);   cudaFree(dNewVent);   cudaFree(dNewDesh);
    
    qi++;
  }

  // Resumen final
  if (!all_total_ms.empty()) {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║        RESUMEN DE LA SESIÓN           ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    double p50 = percentile(all_total_ms, 50.0);
    double p95 = percentile(all_total_ms, 95.0);
    double sum_ms = std::accumulate(all_total_ms.begin(), all_total_ms.end(), 0.0);
    double qps = (sum_ms>0.0) ? (1000.0 * qi / sum_ms) : 0.0;

    std::cout << "Total de consultas: " << qi << "\n";
    std::cout << "QPS (consultas/seg): " << std::fixed << std::setprecision(2) << qps << "\n";
    std::cout << "Latencia p50: " << p50 << " ms\n";
    std::cout << "Latencia p95: " << p95 << " ms\n";
    std::cout << "\nPromedios por etapa:\n";
    std::cout << "  • NLU:  " << (std::accumulate(all_nlu_ms.begin(),  all_nlu_ms.end(),  0.0)/qi) << " ms\n";
    std::cout << "  • DATA: " << (std::accumulate(all_data_ms.begin(), all_data_ms.end(), 0.0)/qi) << " ms\n";
    std::cout << "  • FUSE: " << (std::accumulate(all_fuse_ms.begin(), all_fuse_ms.end(), 0.0)/qi) << " ms\n";
    std::cout << "\n✓ Métricas guardadas en 'metrics.csv'\n\n";
  }

  // Limpieza final
  metricsCsv.close();
  cudaFree(dQ);
  cudaFree(dVQ); cudaFree(dScores); cudaFree(dM);
  cudaFree(dX);  cudaFree(dMean);   cudaFree(dStd);
  cudaFreeHost(hQ); cudaFreeHost(hVQ); cudaFreeHost(hScores); cudaFreeHost(hX);
  cudaEventDestroy(evNLUStart); cudaEventDestroy(evNLUStop);
  cudaEventDestroy(evDATAStart); cudaEventDestroy(evDATAStop);
  cudaEventDestroy(evFUSEStart); cudaEventDestroy(evFUSEStop);
  cudaEventDestroy(evStart); cudaEventDestroy(evStop);
  cudaStreamDestroy(sNLU); cudaStreamDestroy(sDATA); cudaStreamDestroy(sFUSE);
  
  std::cout << "¡Hasta luego! 👋\n";
  return 0;
}
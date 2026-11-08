// ldr_features.h — extração de atributos da janela de leituras do LDR.
// C++ puro (sem Arduino): é compilado igual no ESP32 e no teste de paridade em PC.
// Precisa ser IDÊNTICO ao cálculo de ml/train_classifier.py (função extract_features).
#pragma once
#include <math.h>

#define WINDOW_SIZE 10
#define LUX_MIN 0.1f
#define LUX_MAX 100000.0f

// Atributos usados pelo classificador (ordem fixa!)
//   [0] lmean = log10(média da janela em lux)
//   [1] lmin  = log10(menor leitura da janela)
//   [2] lmax  = log10(maior leitura da janela)
//   [3] cv    = desvio padrão / média (coeficiente de variação)
#define N_FEATURES 4

static inline float clampLux(float v) {
  if (!(v > LUX_MIN)) return LUX_MIN;  // também trata NaN
  if (v > LUX_MAX) return LUX_MAX;
  return v;
}

static inline void extractFeatures(const float *lux, int n, float *out) {
  float sum = 0, mn = LUX_MAX, mx = LUX_MIN;
  for (int i = 0; i < n; i++) {
    float v = clampLux(lux[i]);
    sum += v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  float mean = sum / n;
  float var = 0;
  for (int i = 0; i < n; i++) {
    float d = clampLux(lux[i]) - mean;
    var += d * d;
  }
  float sd = sqrtf(var / n);
  out[0] = log10f(mean);
  out[1] = log10f(mn);
  out[2] = log10f(mx);
  out[3] = sd / mean;
}

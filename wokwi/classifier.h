// classifier.h — GERADO por ml/train_classifier.py (não editar à mão).
// Árvore de decisão (TinyML) sobre os atributos de ldr_features.h.
#pragma once
#define CLASS_ADEQUADO 0
#define CLASS_FRACO    1
#define CLASS_ESCURO   2
#define CLASS_INSTAVEL 3

static const char *CLASS_NAMES[4] = {"ADEQUADO", "FRACO", "ESCURO", "INSTAVEL"};

// f[] = {lmean, lmin, lmax, cv} calculado por extractFeatures()
static inline int classifyWindow(const float *f) {
  if (f[0] <= 0.669724f) {
    if (f[0] <= 0.641033f) {
      return 2;  // ESCURO
    } else {
      if (f[2] <= 0.779144f) {
        return 2;  // ESCURO
      } else {
        return 2;  // ESCURO
      }
    }
  } else {
    if (f[3] <= 0.389160f) {
      if (f[0] <= 1.312255f) {
        if (f[0] <= 0.704590f) {
          if (f[2] <= 0.774096f) {
            return 2;  // ESCURO
          } else {
            return 1;  // FRACO
          }
        } else {
          if (f[0] <= 1.297120f) {
            return 1;  // FRACO
          } else {
            return 1;  // FRACO
          }
        }
      } else {
        if (f[1] <= 1.115762f) {
          return 3;  // INSTAVEL
        } else {
          if (f[0] <= 1.341214f) {
            return 0;  // ADEQUADO
          } else {
            return 0;  // ADEQUADO
          }
        }
      }
    } else {
      if (f[3] <= 0.459886f) {
        return 3;  // INSTAVEL
      } else {
        return 3;  // INSTAVEL
      }
    }
  }
}

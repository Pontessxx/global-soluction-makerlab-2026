// Teste em PC: (1) paridade Python x C++ ; (2) perfis do modo TEST caem na classe esperada.
// Compilar: g++ -std=c++17 -I wokwi ml/tests/host_test.cpp -o /tmp/host_test && /tmp/host_test ml/tests/parity_vectors.txt
#include <cstdio>
#include <cstdlib>
#include "ldr_features.h"
#include "classifier.h"

static float noise(float pct) { return 1.0f + pct * ((rand() % 2001 - 1000) / 1000.0f); }

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "uso: host_test parity_vectors.txt\n"); return 2; }
  FILE *fp = fopen(argv[1], "r");
  if (!fp) { perror("abrir vetores"); return 2; }
  int total = 0, ok = 0;
  float w[WINDOW_SIZE], f[N_FEATURES];
  for (;;) {
    int i = 0;
    for (; i < WINDOW_SIZE; i++) if (fscanf(fp, "%f", &w[i]) != 1) break;
    int py;
    if (i < WINDOW_SIZE || fscanf(fp, "%d", &py) != 1) break;
    extractFeatures(w, WINDOW_SIZE, f);
    total++;
    if (classifyWindow(f) == py) ok++;
  }
  fclose(fp);
  printf("Paridade Python x C++: %d/%d iguais\n", ok, total);

  // Mesmos perfis de readLux() em sketch.ino (modo TEST)
  struct { const char *nome; int esperado; } casos[] = {
    {"ADEQUADO", CLASS_ADEQUADO}, {"FRACO", CLASS_FRACO}, {"ESCURO", CLASS_ESCURO}, {"INSTAVEL", CLASS_INSTAVEL}};
  int perfOk = 0, perfTotal = 0;
  for (int c = 0; c < 4; c++) {
    int hit = 0;
    for (int rep = 0; rep < 500; rep++) {
      for (int i = 0; i < WINDOW_SIZE; i++) {
        float v = 0;
        if (c == 0) v = 120.0f * noise(0.06f);
        if (c == 1) v = 11.0f * noise(0.08f);
        if (c == 2) v = 1.5f * noise(0.20f);
        if (c == 3) v = (rand() % 2) ? 150.0f * noise(0.05f) : 0.8f * noise(0.2f);
        w[i] = clampLux(v);
      }
      extractFeatures(w, WINDOW_SIZE, f);
      if (classifyWindow(f) == casos[c].esperado) hit++;
    }
    printf("Perfil %-8s: %d/500 classificados corretamente\n", casos[c].nome, hit);
    perfOk += hit; perfTotal += 500;
  }
  return (ok == total && perfOk >= perfTotal * 0.99) ? 0 : 1;
}

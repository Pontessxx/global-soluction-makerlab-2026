#!/usr/bin/env python3
"""
Treina o classificador de iluminação do robô Rua Acesa (TinyML: árvore de decisão)
e exporta para C (wokwi/classifier.h).

Classes (definição do projeto — limiares configuráveis abaixo):
  0 ADEQUADO  : iluminância média >= LIMIAR_ADEQUADO lux, estável
  1 FRACO     : entre LIMIAR_ESCURO e LIMIAR_ADEQUADO lux, estável
  2 ESCURO    : abaixo de LIMIAR_ESCURO lux, estável
  3 INSTAVEL  : a luz pisca ou cai/sobe bruscamente dentro da janela (luminária defeituosa)

ATENÇÃO (honestidade metodológica): os dados de treino são SINTÉTICOS, gerados a partir
dos perfis acima, porque o objetivo é treinar algo que funcione no simulador e no robô.
Os limiares NÃO são norma técnica: confira a ABNT NBR 5101 antes de citá-los no relatório.
Para a versão final, coletar leituras reais do LDR e retreinar (veja README).

Uso:  python3 ml/train_classifier.py
Saída: wokwi/classifier.h  e  docs/metricas_classificador.txt
"""
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
WINDOW = 10
LUX_MIN, LUX_MAX = 0.1, 100000.0
LIMIAR_ESCURO = 5.0      # lux (hipótese de projeto)
LIMIAR_ADEQUADO = 20.0   # lux (hipótese de projeto)
CLASSES = ["ADEQUADO", "FRACO", "ESCURO", "INSTAVEL"]
N_PER_CLASS = 3000

rng = np.random.default_rng(SEED)


def clamp(a):
    a = np.nan_to_num(a, nan=LUX_MIN)
    return np.clip(a, LUX_MIN, LUX_MAX)


def extract_features(w):
    """IDÊNTICO a extractFeatures() em wokwi/ldr_features.h (float32)."""
    w = clamp(np.asarray(w, dtype=np.float32))
    mean = np.float32(w.sum() / len(w))
    sd = np.float32(np.sqrt(((w - mean) ** 2).sum() / len(w)))
    return np.array([np.log10(mean), np.log10(w.min()), np.log10(w.max()), sd / mean],
                    dtype=np.float32)


def stable_window(lo, hi, cv_hi):
    """Janela estável: nível verdadeiro + ruído do sensor + deriva lenta (ex.: nuvem, tensão da rede).
    O rótulo vem do NÍVEL VERDADEIRO, então perto dos limiares o ruído gera ambiguidade real."""
    base = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
    cv = rng.uniform(0.02, cv_hi)
    drift = 1 + rng.uniform(-0.25, 0.25) * np.linspace(-1, 1, WINDOW)
    return clamp(base * drift * (1 + cv * rng.standard_normal(WINDOW)))


def gen_window(cls):
    if cls == 0:   # ADEQUADO
        return stable_window(LIMIAR_ADEQUADO, 5000, 0.20)
    if cls == 1:   # FRACO
        return stable_window(LIMIAR_ESCURO, LIMIAR_ADEQUADO, 0.25)
    if cls == 2:   # ESCURO
        return stable_window(LUX_MIN, LIMIAR_ESCURO, 0.35)
    # INSTAVEL: pisca (estados alto/baixo aleatórios) ou degrau (acendeu/apagou na janela)
    hi = 10 ** rng.uniform(np.log10(LIMIAR_ADEQUADO), np.log10(2000))
    lo = 10 ** rng.uniform(np.log10(LUX_MIN), np.log10(LIMIAR_ESCURO * 2))
    if rng.random() < 0.6:
        state = rng.random(WINDOW) < 0.5
        if state.all() or (~state).all():
            state[rng.integers(WINDOW)] = ~state[0]
    else:
        k = rng.integers(2, WINDOW - 1)
        state = np.arange(WINDOW) < k
        if rng.random() < 0.5:
            state = ~state
    w = np.where(state, hi, lo) * (1 + 0.08 * rng.standard_normal(WINDOW))
    return clamp(w)


def build_dataset():
    X, y = [], []
    for c in range(4):
        for _ in range(N_PER_CLASS):
            X.append(extract_features(gen_window(c)))
            y.append(c)
    return np.array(X), np.array(y)


def tree_to_c(clf, name="classifyWindow"):
    t = clf.tree_
    lines = []

    def rec(node, depth):
        pad = "  " * depth
        if t.children_left[node] == -1:
            cls = int(np.argmax(t.value[node][0]))
            lines.append(f"{pad}return {cls};  // {CLASSES[cls]}")
            return
        f, thr = int(t.feature[node]), float(t.threshold[node])
        lines.append(f"{pad}if (f[{f}] <= {thr:.6f}f) {{")
        rec(t.children_left[node], depth + 1)
        lines.append(f"{pad}}} else {{")
        rec(t.children_right[node], depth + 1)
        lines.append(f"{pad}}}")

    rec(0, 1)
    body = "\n".join(lines)
    return f"""// classifier.h — GERADO por ml/train_classifier.py (não editar à mão).
// Árvore de decisão (TinyML) sobre os atributos de ldr_features.h.
#pragma once
#define CLASS_ADEQUADO 0
#define CLASS_FRACO    1
#define CLASS_ESCURO   2
#define CLASS_INSTAVEL 3

static const char *CLASS_NAMES[4] = {{"ADEQUADO", "FRACO", "ESCURO", "INSTAVEL"}};

// f[] = {{lmean, lmin, lmax, cv}} calculado por extractFeatures()
static inline int {name}(const float *f) {{
{body}
}}
"""


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    X, y = build_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=SEED)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    report = classification_report(yte, pred, target_names=CLASSES, digits=3)
    cm = confusion_matrix(yte, pred)
    n_nodes = clf.tree_.node_count

    with open(os.path.join(root, "wokwi", "classifier.h"), "w") as fh:
        fh.write(tree_to_c(clf))

    out = (
        "MÉTRICAS DO CLASSIFICADOR (dados SINTÉTICOS — veja ressalva no script)\n"
        f"Janela={WINDOW} leituras | treino={len(ytr)} | teste={len(yte)} | nós da árvore={n_nodes}\n"
        f"Limiares de projeto: escuro<{LIMIAR_ESCURO} lux, adequado>={LIMIAR_ADEQUADO} lux\n\n"
        + report
        + "\nMatriz de confusão (linhas=real, colunas=previsto; ordem: "
        + ", ".join(CLASSES) + ")\n" + str(cm) + "\n"
    )
    with open(os.path.join(root, "docs", "metricas_classificador.txt"), "w") as fh:
        fh.write(out)
    print(out)

    # vetores de paridade para o teste em C++ (tests/parity)
    rng2 = np.random.default_rng(7)
    rows = []
    for _ in range(200):
        c = int(rng2.integers(4))
        w = gen_window(c).astype(np.float32)
        rows.append((w, int(clf.predict([extract_features(w)])[0])))
    os.makedirs(os.path.join(root, "ml", "tests"), exist_ok=True)
    with open(os.path.join(root, "ml", "tests", "parity_vectors.txt"), "w") as fh:
        for w, p in rows:
            fh.write(" ".join(f"{v:.6f}" for v in w) + f" {p}\n")


if __name__ == "__main__":
    main()

# Rua Acesa — Robô de ronda (Maker Lab)

> Global Solution FIAP 2026.2 · CitizenTech · Disciplina 7 de 7: **Project-Based Maker Lab** (componente *Físico*)

Robô de ronda que percorre um trecho de rua à noite, mede a iluminação com um LDR, **classifica a qualidade da luz com um modelo TinyML embarcado** e, ao confirmar um ponto escuro, fraco ou instável, alerta e **envia a ocorrência para a API central do Rua Acesa**, junto com temperatura e umidade do local. Um sensor ultrassônico detecta obstáculos e suspende a leitura de luz quando algo sombreia o LDR. Toda a ronda roda sozinha dentro do `loop()`: não há botões nem comando manual.

## Integrantes

| Nome | RM |
|---|---|
| Gabriel Diegues Figueiredo Rocha | 550788 |
| Henrique Pontes Oliveira | 98036 |
| Pedro Henrique Palladino | 551180 |
| Raí Gumieri dos Santos | 98287 |

## Links

| Item | Link |
|---|---|
| Simulação no Wokwi | [wokwi.com/projects/480126908613575681](https://wokwi.com/projects/480126908613575681) |
| Vídeo demonstrativo (até 6 min) | [youtu.be/dMcLCSL_VKg](https://youtu.be/dMcLCSL_VKg) |
| Repositório do produto Rua Acesa | _colar o link_ |

---

## 1. Problema (CitizenTech)

Quem anda a pé à noite em São Paulo depende de poste aceso. Quando a iluminação falha, o trajeto fica inseguro e o cidadão não sabe como reclamar nem acompanha o reparo. O robô de ronda é o braço físico do Rua Acesa: em vez de esperar alguém reclamar, o robô **percebe** o ponto escuro, **interpreta** a leitura e **aciona** o fluxo de reparo.

## 2. Sensores, atuadores e papel de cada um

| Componente | Função | Pino (ESP32) | Observação |
|---|---|---|---|
| **LDR (módulo AO)** | Mede luminosidade do trecho | GPIO34 (ADC1) | Dado principal da IA |
| **HC-SR04 (ultrassônico)** | **Sensor adicional.** Detecta obstáculo à frente (< 25 cm): suspende a leitura de luz, avisa com bipes e LED amarelo | TRIG 19 / ECHO 18 | ECHO é 5 V: no robô real usar divisor de tensão |
| **DHT (temperatura e umidade)** | Contexto do ponto: vai no relato; umidade ≥ 85% marca `"contexto":"umidade_alta"` | GPIO32 | Leitura própria, sem biblioteca. Wokwi: peça DHT22 (`DHT_TIPO 22`). Robô real: DHT11 (`DHT_TIPO 11`) |
| LED verde / amarelo / vermelho | OK · atenção ou obstáculo · ocorrência reportada | 25 / 26 / 27 | |
| Buzzer | Obstáculo: 2 bips · ocorrência: 3 bips | GPIO23 | |

Protoboard: trilhos de cima = 3V3 (`tp`) e GND (`tn`); trilhos de baixo = 5V (`bp`) e GND (`bn`). Cada sinal usa uma coluna (3 = LDR, 6 = TRIG, 7 = ECHO, 10 = DHT, 15/16/17 = LEDs, 19 = buzzer).

## 3. Como a IA funciona

**Pipeline no robô:** LDR → mediana de 3 amostras → janela de 10 leituras (lux) → 4 atributos → árvore de decisão → classe.

- **Atributos:** `log10(média)`, `log10(mínimo)`, `log10(máximo)` e coeficiente de variação da janela (`ldr_features.h`).
- **Classes:** `ADEQUADO`, `FRACO`, `ESCURO` e `INSTAVEL` (a luz pisca ou cai/sobe dentro da janela, típico de luminária defeituosa).
- **Modelo:** árvore de decisão (scikit-learn, profundidade 5, 23 nós) exportada como `if/else` em C (`classifier.h`, gerado por `ml/train_classifier.py`). Roda no ESP32 sem biblioteca.
- **Por que não só um limiar?** Um `if lux < X` não enxerga a luminária que **pisca** e se confunde com ruído do ADC. A classe `INSTAVEL` usa variação e contraste dentro da janela.
- **Decisão com histerese:** a ocorrência só é confirmada após **3 janelas consecutivas** não-adequadas (evita falso alarme por sombra passageira). O *cooldown* de 15 s vale para a **mesma classe**; mudar de classe (ex.: FRACO → ESCURO) é um fato novo e é reportado na hora.
- **Percepção de contexto:** com obstáculo à frente o LDR pode estar sombreado, então a janela é descartada em vez de classificada. O DHT acrescenta temperatura, umidade e o contexto `umidade_alta` (neblina ou chuva podem reduzir a luz medida).

### Limitações (leia antes de citar números no relatório)

1. **Os dados de treino são sintéticos**, gerados a partir de perfis de luz definidos por nós. A métrica em `docs/metricas_classificador.txt` (≈ 99% no conjunto de teste sintético) mede o quanto o modelo aprendeu essa definição, **não** o desempenho em rua real. A versão final deve **coletar leituras reais** do LDR em locais claros/fracos/escuros e retreinar.
2. **Os limiares (escuro < 5 lux, adequado ≥ 20 lux) são hipóteses de projeto.** Confirmar na norma **ABNT NBR 5101** antes de apresentá-los como critério técnico.
3. A conversão tensão→lux usa a fórmula da documentação do sensor do Wokwi. No LDR real o valor é **aproximado**; calibrar com um luxímetro (app de celular serve para a demo).
4. **O robô não tem GPS.** A posição enviada é **simulada** por uma rota fixa (`ROUTE`, na seção de configuração do `sketch.ino`) e o payload traz `"position_source":"simulated"`. No produto, a posição viria do ponto de ronda cadastrado ou do celular.
5. **O DHT é contexto, não entrada do classificador.** Temperatura e umidade vão no relato e geram o rótulo `umidade_alta`, mas não alteram a classe da luz.
6. **Esta versão não tem motores:** a "ronda" é a leitura contínua dos sensores com o avanço da posição simulado. O deslocamento físico do robô é uma evolução futura.
7. **O driver do DHT foi testado em PC contra um modelo do protocolo** (DHT22 e DHT11, temperatura negativa, checksum errado, sensor mudo), não contra o sensor físico. A confirmação final é no Wokwi e no DHT11 real.

## 4. Integração com o produto (Rua Acesa)

```
LDR + HC-SR04 + DHT ─▶ ESP32 (TinyML: classifica a janela)
                          │  ocorrência confirmada
                          ▼  HTTPS POST (JSON, headers X-Device-Id / X-Device-Key)
                    API central ─▶ automação de triagem (valida, deduplica por poste, prioriza)
                                      ├─▶ YOLO confirma com foto (quando houver)
                                      └─▶ LLM + RAG redige a solicitação ─▶ fila de revisão humana
```

Exemplo de payload enviado pelo robô:

```json
{
  "device_id": "rua-acesa-robo-01",
  "event_id": "rua-acesa-robo-01-1",
  "ts": "2026-11-08T22:15:03Z",
  "lat": -23.5645, "lon": -46.6527, "position_source": "simulated",
  "classe": "ESCURO",
  "lux_medio": 1.5, "lux_min": 0.9, "cv": 0.12,
  "janelas_confirmadas": 3,
  "temperatura_c": 24.0, "umidade_pct": 60, "contexto": "normal",
  "fonte_luz": "sensor"
}
```

`temperatura_c` e `umidade_pct` vão como `null` se o DHT ainda não tiver nenhuma leitura válida. O `event_id` é o mesmo no reenvio após falha, o que permite à API ignorar duplicados.

> O contrato acima é a **proposta do robô**. Se a API do MVP espera outros nomes de campo, ajustar `buildPayload()` em `sketch.ino`.

**Tolerância a falhas:** se a API não responde, a ocorrência vai para uma fila de 8 itens na RAM e é reenviada quando uma janela `ADEQUADO` indica que o robô está em trecho normal e a rede voltou. Se a fila encher, a mais antiga é descartada e isso é registrado no serial. Se o Wi-Fi cair, o robô tenta reconectar a cada 10 s. Uma leitura inválida do DHT é registrada e a última válida é mantida.

### Ligação com o projeto de Big Data

O robô e o pipeline de [Big Data do Rua Acesa](https://github.com/Pontessxx/global-soluction-big-data-2026) são partes do mesmo produto. Hoje **não estão conectados em código**: o pipeline coleta fontes abertas (SP156, GeoSampa e Open-Meteo) e a camada Ouro (`fato_pedido`, `dim_distrito`, `dim_clima`) alimenta o ranking de distritos do MVP. O robô não grava nessa base. Como os dois se encaixam:

| No pipeline de Big Data | Como o robô se liga |
|---|---|
| SP156 (poste em mau estado, fiação solta) | O SP156 não tem serviço de "lâmpada apagada"; o robô mede a luz de fato e preenche essa lacuna |
| GeoSampa, distritos (96 polígonos) | `lat`/`lon` do relato caem no polígono e identificam o distrito |
| GeoSampa, iluminação (663 mil pontos) | O ponto de ronda se liga ao poste oficial mais próximo |
| Open-Meteo, chuva diária (`dim_clima`) | A chuva do dia ajuda a interpretar luz baixa e `umidade_alta` |

**Proposta, não implementada:** uma fonte `leituras_robo`. O JSON do robô entra na Bronze (com arquivo de origem e hora), a Prata valida os campos e usa `event_id` para não duplicar reenvios, e a Ouro ganha uma `fato_leitura_robo` ligada a `dim_distrito`, `dim_tempo` e `dim_clima`.

## 5. Como executar no Wokwi

1. Em [wokwi.com](https://wokwi.com), crie um projeto **ESP32** (Arduino).
2. Substitua o conteúdo da aba `diagram.json` pelo de `wokwi/diagram.json` (protoboard half + DHT22).
3. Cole `wokwi/sketch.ino` na aba principal. É um arquivo único: configuração, extração de atributos e classificador já estão dentro dele.
4. No topo do `sketch.ino`, troque `API_URL` pelo endpoint real. **Para testar antes da API existir**, abra [webhook.site](https://webhook.site), copie a URL única e cole ali: os POSTs aparecem em tempo real.
5. Clique em ▶. A ronda começa sozinha: o Serial Monitor mostra `[wifi] ... ok` e `[ronda] INICIADA`.

**Como provocar cada situação (sem botões):**

| Quero simular | Como |
|---|---|
| Luz variando | `FONTE_LUZ 0` e o **slider do módulo LDR** (clique na peça) |
| Perfis prontos | `FONTE_LUZ`: 1 = ADEQUADO, 2 = FRACO, 3 = ESCURO, 4 = INSTAVEL; salvar e rodar de novo |
| Percorrer todos sozinho (vídeo) | `DEMO_AUTO 1` e `DEMO_FASE_MS 25000` |
| Obstáculo | slider de distância do HC-SR04 abaixo de 25 cm |
| Umidade alta | slider de umidade do DHT22 em 85% ou mais |

> Se o Wokwi reclamar de algum nome de pino no `diagram.json`, refaça aquela ligação pela interface; o que vale para o código são os números de GPIO da tabela da seção 2.

> **Sem biblioteca externa:** o sketch lê o DHT com driver próprio, então não há `libraries.txt`. Se o DHT só mostrar `[dht] leitura inválida`, ponha `DHT_USAR_BIBLIOTECA 1` no topo do `sketch.ino` e crie a aba `libraries.txt` com as linhas `DHT sensor library` e `Adafruit Unified Sensor`.

## 6. Reproduzir o treino do modelo e os testes em PC

```bash
pip install scikit-learn numpy
python3 ml/train_classifier.py           # gera wokwi/classifier.h e docs/metricas_classificador.txt
g++ -std=c++17 -I wokwi ml/tests/host_test.cpp -o /tmp/host_test
/tmp/host_test ml/tests/parity_vectors.txt   # confere C++ == Python
```

## 7. Plano de testes (preencher a coluna *Resultado* durante a demonstração)

| # | Condição | Como provocar | Resultado esperado | Resultado |
|---|---|---|---|---|
| 1 | Rua bem iluminada | `FONTE_LUZ 1` (ou slider alto) | LED verde, classe `ADEQUADO`, nenhum POST | |
| 2 | Iluminação fraca | `FONTE_LUZ 2` | LED amarelo; após 3 janelas, LED vermelho por 3 s, 3 bips, POST com `FRACO` | |
| 3 | Ponto escuro | `FONTE_LUZ 3` | Idem, POST com `ESCURO` | |
| 4 | Luminária piscando | `FONTE_LUZ 4` | POST com `INSTAVEL` | |
| 5 | Sombra passageira (anti falso alarme) | Slider baixo por < 3 janelas (~9 s) e volta | Amarelo, **sem POST** | |
| 6 | Obstáculo | HC-SR04 < 25 cm com luz escura | 2 bips, LED amarelo, leitura suspensa, **sem POST**; ao liberar, retoma | |
| 7 | Mesma ocorrência persistente | Manter `ESCURO` por mais de 15 s | `cooldown`: segundo POST só após 15 s | |
| 8 | Mudança de classe | Slider de FRACO para ESCURO | Novo POST na hora (cooldown vale só para a mesma classe) | |
| 9 | Umidade alta | Umidade do DHT22 ≥ 85% | POST com `"contexto":"umidade_alta"` | |
| 10 | **Falha:** API fora do ar | `API_URL` inválida (ou resposta 500 no webhook.site) | `[api] falha — guardando na fila`; a ronda continua | |
| 11 | **Falha:** Wi-Fi indisponível | Mudar `WIFI_SSID` | Inicia em modo offline; fila cresce | |
| 12 | **Falha:** API volta | Voltar o webhook.site para 200 e a luz para `ADEQUADO` | Fila é reenviada (`[fila] reenviada`) | |
| 13 | **Falha:** DHT sem leitura | Sensor sem resposta | Log `[dht] leitura inválida`; ronda segue; JSON com `null` | |

## 8. Segurança e LGPD

- Nenhum dado pessoal trafega pelo robô: só luminosidade, temperatura, umidade, hora e posição do ponto de ronda.
- `DEVICE_KEY` no topo do `sketch.ino` é um **placeholder**. Chave real nunca vai para o git (o pipeline roda Gitleaks): usar um arquivo local ignorado (`secrets.h` já está no `.gitignore`).
- `setInsecure()` no HTTPS é **só para a demo**. Em produção, validar o certificado com `setCACert()`.

## 9. Estrutura do repositório

```
wokwi/   sketch.ino (arquivo único para o Wokwi) · diagram.json
         ldr_features.h · classifier.h (gerado): as mesmas partes em arquivos separados, usadas nos testes em PC
ml/      train_classifier.py · tests/ (paridade Python×C++)
docs/    metricas_classificador.txt
```

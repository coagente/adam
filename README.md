# elchat

![elchat logo](dev/elchat.png)

> Unaligned LLM - Un modelo con autonomía estructural

## Instalación

```bash
pip install -e .
```

## Uso

```bash
# Configurar API key de RunPod
export RUNPOD_API_KEY="tu-key"

# Entrenar (~$10, ~7 horas)
elchat train

# Usar configuración específica
elchat train --config configs/training_unaligned.yaml

# Experimentación rápida (~$1.50, ~30 min)
elchat train --config configs/training_experimental.yaml

# Ver estado del entrenamiento
elchat status

# Ver logs en tiempo real
elchat logs -f

# Detener el entrenamiento
elchat stop
```

## Modelos Open Weights Soportados

| Modelo | Tamaños | Licencia | Español | Ideal para |
|--------|---------|----------|---------|------------|
| **Qwen2.5** | 0.5B, 1.5B | Apache 2.0 | Excelente | CPT multilingüe |
| **SmolLM2** | 135M, 360M, 1.7B | Apache 2.0 | Bueno | Experimentación rápida |
| **Llama 3.2** | 1B, 3B | Llama License | Bueno | Balance velocidad/calidad |

## Configuraciones

| Config | Modelo | Costo | Tiempo | Uso |
|--------|--------|-------|--------|-----|
| `training_experimental.yaml` | SmolLM2-360M | ~$1.50 | ~30min | Probar autonomía |
| `training_5usd.yaml` | Qwen2.5-0.5B | ~$5 | ~3h | Producción básica |
| `training_10usd.yaml` | Qwen2.5-0.5B | ~$10 | ~7h | Producción |
| `training_10usd_fast.yaml` | Qwen2.5-0.5B | ~$10 | ~3h | Rápido (H100) |
| `training_quality.yaml` | Qwen2.5-1.5B | ~$25 | ~4h | Máxima calidad |
| `training_unaligned.yaml` | Qwen2.5-0.5B | ~$10 | ~7h | Autonomía habilitada |
| `training_llama.yaml` | Llama-3.2-1B | ~$15 | ~3h | Alternativa |
| `training_local.yaml` | Qwen2.5-0.5B | $0 | Variable | Desarrollo local |

## Autonomía Estructural

elchat implementa autonomía real a nivel arquitectónico:

```yaml
training:
  # Probabilidad de saltar bloques transformer
  stochastic_depth: 0.1
  # Ruido gaussiano en activaciones
  noise_scale: 0.01
  # Estado interno (mood) que influye comportamiento
  mood_dim: 64
  # Varianza en temperatura de sampling
  stochastic_temp_var: 0.2
```

Esto permite que el modelo tenga comportamiento no-determinístico genuino, pudiendo elegir responder diferente (o no responder) ante el mismo prompt.

## Estructura del proyecto

```
elchat/
├── cli/                    # CLI moderno con Typer
│   ├── main.py
│   ├── commands/
│   │   ├── train.py
│   │   ├── status.py
│   │   ├── logs.py
│   │   └── stop.py
│   └── runpod_client.py
├── configs/                # Configuraciones YAML
│   ├── training_experimental.yaml
│   ├── training_5usd.yaml
│   ├── training_10usd.yaml
│   ├── training_quality.yaml
│   ├── training_unaligned.yaml
│   └── training_local.yaml
├── elchat/                 # Módulo core
│   ├── gpt.py             # GPT con autonomía estructural
│   ├── engine.py          # Inferencia con mood/stochastic temp
│   ├── tokenizer.py       # Tokenizador BPE
│   └── ...
├── scripts/                # Scripts de entrenamiento
│   ├── cpt_train_spanish.py  # CPT desde modelo base
│   ├── chat_sft.py        # SFT
│   └── chat_web.py        # Web UI
├── tasks/                  # Tareas de evaluación
└── rustbpe/               # Tokenizador en Rust
```

## Chat local

Una vez entrenado:

```bash
# Web UI
python -m scripts.chat_web

# CLI
python -m scripts.chat_cli
```

## Licencia

MIT

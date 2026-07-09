# Atenuación Continua en Dominio de Gradiente para la Estimación Multiescala de Reflectancia

Proyecto de investigación desarrollado para el curso **IEE3787 — Procesamiento Multiescala de Imágenes** (Pontificia Universidad Católica de Chile).

**Autor:** Jorge Medina
**Supervisor:** Prof. IEE3787
**Repositorio:** https://github.com/JORGLUIS/multiscale-reflectance-gradient-domain
**Enfoque:** Descomposición intrínseca ($I = R \times S$) mediante una atenuación continua en dominio de gradiente que generaliza Retinex-Horn, implementada sobre pirámides Starlet (à trous B3-spline), MMT (mediana) y Wavelet (SWT db2). El punto de partida fue una regla de atribución local por magnitud con coherencia inter-escala y modulación de color, auditada contra el ground-truth (ver Sección IV-A del informe).

---

## Características del Proyecto
* **Transformadas Multiescala Avanzadas:** Soporte completo de la transformada **Starlet à trous** y **Multiscale Median Transform (MMT)** para descomposición sin ringing de Gibbs en bordes.
* **Módulos de Coherencia Físicos:**
  * **Coherencia de Color:** Modula localmente el umbral de reflectancia utilizando el gradiente de la cromaticidad normalizada para evitar fugas de sombras.
  * **Coherencia Inter-Escala:** Valida la consistencia espacial de los coeficientes de detalle entre niveles adyacentes de la transformada.
* **Baselines de Comparación:** Implementación y evaluación exhaustiva contra:
  * Filtrado Homomórfico (filtro pasa-altos logarítmico lineal).
  * Single-Scale Retinex (SSR) y Multi-Scale Retinex (MSR) con conservación exacta de energía.
  * Retinex-Horn (basado en Poisson utilizando resolvedores multigrid sparse con `pyamg`).
* **Protocolo Cuantitativo Estricto:** Evaluación sobre los benchmarks canónicos **MIT Intrinsic Images** y **MPI-Sintel** utilizando la métrica invariante a escala **LMSE** (Local Mean Squared Error), validada contra el código de referencia de Grosse et al. (ICCV 2009).

---

## Instalación y Requisitos

Este proyecto requiere Python 3.10+ y un gestor de dependencias standard (`pip`).

1. Clonar el repositorio y acceder al directorio del proyecto:
   ```bash
   git clone https://github.com/JORGLUIS/multiscale-reflectance-gradient-domain.git
   cd multiscale-reflectance-gradient-domain
   ```
2. Instalar las dependencias listadas en `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   > **Nota:** La biblioteca `pyamg` se requiere para resolver la ecuación de Poisson en Retinex-Horn.

---

## Descarga de Datos

Los conjuntos de datos no se guardan en el repositorio Git. Descárgalos y configúralos ejecutando:
```bash
python scripts/download_data.py
```
Este script creará la carpeta `data/` y descargará:
1. El dataset **MIT Intrinsic Images** (objetos reales con 11 iluminaciones c/u).
2. El dataset **MPI-Sintel Intrinsic** (escenas sintéticas densas de videojuegos).

---

## Demostración Rápida (Una Línea)

Para correr la demostración interactiva que carga un objeto, aplica las descomposiciones, calcula el LMSE y guarda las estimaciones visuales, ejecuta:
```bash
python demo.py
```
Este script imprimirá una tabla comparativa en la terminal y guardará las imágenes resultantes en `results/demo_outputs/`.

---

## Reproducción de Experimentos

Para reproducir cualquier experimento, usando el id del archivo YAML correspondiente en `configs/`:
```bash
python scripts/run_experiment.py --config configs/<experimento_id>.yaml
```
Por ejemplo, para reproducir el modelo propuesto MMT completo en Sintel:
```bash
python scripts/run_experiment.py --config configs/f3_sintel_mmt_all.yaml
```
Los resultados cuantitativos se guardan en archivos CSV en `results/tables/`.

---

## Pruebas Unitarias

Para correr toda la suite de pruebas automatizadas de sanidad (reconstrucción perfecta, conservación física e identidad de la métrica LMSE):
```bash
pytest -v
```

---

## Estructura del Código

```text
├── configs/          # Configuraciones YAML de experimentos (.yaml)
├── data/             # Carpeta de datasets descargados (MIT y Sintel)
├── notebooks/        # Jupyter Notebook de visualización interactiva
│   └── demo.ipynb
├── results/          # Tablas CSV de resultados y salidas visuales de demostración
├── scripts/          # Scripts operativos CLI (download_data.py, run_experiment.py)
├── src/              # Lógica pura del pipeline
│   ├── baselines/    # Baselines (homomorphic, ssr, msr, horn)
│   ├── data/         # Loaders y normalizadores de MIT y Sintel
│   ├── decompose/    # Reglas de atribución y pipeline multiescala
│   ├── metrics/      # Implementación del LMSE scale-invariant
│   ├── transforms/   # Wavelet Starlet y Multiscale Median Transform (MMT)
│   └── utils/        # resolvedor de Poisson, espacio de color y color mapping
├── tests/            # Suite de pruebas unitarias pytest (F0 a F3)
├── Informe/          # Informe final (PDF)
└── README.md         # Esta guía de usuario
```

---

## Licencia

Este código ha sido desarrollado únicamente con fines académicos para el curso IEE3787 (Procesamiento Multiescala de Imágenes) de la Pontificia Universidad Católica de Chile.

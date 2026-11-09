# Estado de las 28 Issues

Oleadas (Sección 0, regla 11): wave-1 = MVP end-to-end; wave-2 = precisión; wave-3 = opcional.
Cada línea: `[x] COMPLETADA` / `[ ] PENDIENTE` · issue · oleada · título · rama.


## Milestone 1: Infraestructura, CI/CD, Fixtures Sintéticas y Puente de Evaluación

- [x] COMPLETADA · #01 · wave-1 · Inicialización de paquete Python, .gitignore anti-fugas y configuración tipada · `feat/issue-01-repo-scaffold`
- [x] COMPLETADA · #02 · wave-1 · GitHub Actions CI/CD, Security Gate anti-fuga de datos y automatización de las 28 Issues · `feat/issue-02-github-ci-security`
- [x] COMPLETADA · #03 · wave-1 · Generador determinista de partido sintético en formato FIFA/EPTS + SportsCode XML · `feat/issue-03-synthetic-match-generator`
- [x] COMPLETADA · #04 · wave-1 · Puente nativo con possession_metrics.py y validate_submission.py · `feat/issue-04-official-eval-bridge`

## Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV

- [x] COMPLETADA · #05 · wave-1 · Parser en streaming de ATD (.txt/.xml) a metros reales con caché .npy · `feat/issue-05-atd-streaming-loader`
- [x] COMPLETADA · #06 · wave-1 · Parser de Smart Tagging XML a máscaras y eventos por frame · `feat/issue-06-smart-tagging-loader`
- [ ] PENDIENTE · #07 · wave-1 · Cargador vectorizado de matrices de homografía $H$ e inversas $H^{-1}$ · `feat/issue-07-homography-loader`
- [ ] PENDIENTE · #08 · wave-1 · Filtrado de suplentes en banda y fusión de tracks duplicados ($< 0.6\text{ m}$) · `feat/issue-08-track-dedup-and-bounds`
- [ ] PENDIENTE · #09 · wave-1 · Cinemática Savitzky-Golay $(\vec{v}, \vec{a})$ y normalización de orientación de ataque · `feat/issue-09-savgol-kinematics-orientation`
- [ ] PENDIENTE · #10 · wave-3 · Sincronización de cámara de TV por bloques continuos e inyección de balón · `feat/issue-10-broadcast-block-sync`

## Milestone 3: Visión Guiada por Atención y Geometría 3D de Balón Aéreo

- [ ] PENDIENTE · #11 · wave-3 · Generador de Attention Crops ($640\times640$) con homografía e interfaz de detector · `feat/issue-11-attention-crops-yolo`
- [ ] PENDIENTE · #12 · wave-3 · Detector de balón aéreo (is_aerial) por velocidad aparente y curvatura · `feat/issue-12-aerial-ball-filter`

## Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva

- [ ] PENDIENTE · #13 · wave-1 · Señales cinemáticas continuas de balón muerto (D) con y sin balón visible · `feat/issue-13-dead-ball-kinematic-signals`
- [ ] PENDIENTE · #14 · wave-1 · Anclaje quirúrgico del frame de reinicio $t^*$ en los 59 SET PIECES y post-GOALS · `feat/issue-14-setpiece-exact-anchoring`
- [ ] PENDIENTE · #15 · wave-2 · Resolución determinista del equipo lanzador en los 26 Throw-in con Team = N/A · `feat/issue-15-throwin-team-resolver`
- [ ] PENDIENTE · #16 · wave-1 · Distancias por equipo, esfera de influencia y detector de toques (touch_score) · `feat/issue-16-ball-carrier-and-touches`
- [ ] PENDIENTE · #17 · wave-2 · Física Colectiva I: Porteros dinámicos y derivada de área (Espectral + Convex Hull Numba) · `feat/issue-17-collective-physics-hull-gk`
- [ ] PENDIENTE · #18 · wave-2 · Física Colectiva II: Coherencia vectorial $R \in [0,1]$ y memoria del último balón visto · `feat/issue-18-phase-coherence-and-ball-memory`

## Milestone 5: Degradador de Validación, Auto-Supervisión y Modelos Especialistas

- [ ] PENDIENTE · #19 · wave-2 · Degradador realista de partidos limpios (huecos de 5–60 s + rotura de IDs) · `feat/issue-19-realistic-atd-degrader`
- [ ] PENDIENTE · #20 · wave-2 · Modelo Especialista A (con balón visible): entrenamiento y calibración de probabilidades · `feat/issue-20-specialist-a-ball-visible`
- [ ] PENDIENTE · #21 · wave-2 · Generador de Pseudo-Labeling con *Block Masking* (5–45 s) sobre el propio partido · `feat/issue-21-self-supervised-block-masking`
- [ ] PENDIENTE · #22 · wave-2 · Modelo Especialista B (sin balón) y fusor de emisiones log_probs[N, 3] · `feat/issue-22-specialist-b-ball-absent`

## Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega

- [ ] PENDIENTE · #23 · wave-1 · Constructor de la matriz de transición dependiente del tiempo $A_t \in \mathbb{R}^{N\times 3\times 3}$ · `feat/issue-23-dynamic-transition-matrix`
- [ ] PENDIENTE · #24 · wave-1 · Decodificador Viterbi / CRF lineal con @numba.njit por mitad independiente · `feat/issue-24-numba-viterbi-decoder`
- [ ] PENDIENTE · #25 · wave-1 · Post-procesador directo de spellF1: poda de micro-posesiones e inserción de puente D · `feat/issue-25-spell-pruning-and-d-bridge`
- [ ] PENDIENTE · #26 · wave-1 · Consenso MBR de fronteras y contrato estricto de las 199.319 filas · `feat/issue-26-mbr-consensus-and-contract`
- [ ] PENDIENTE · #27 · wave-2 · Cuadro de mando automatizado de los 6 KPIs de calidad · `feat/issue-27-six-kpi-dashboard`
- [ ] PENDIENTE · #28 · wave-1 · CLI reproducible generate_submission.py, TECHNICAL_REPORT.md y test E2E de determinismo · `feat/issue-28-cli-submission-and-report`

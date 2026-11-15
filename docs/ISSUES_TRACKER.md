# Estado de las 28 Issues

Oleadas (Sección 0, regla 11): wave-1 = MVP end-to-end; wave-2 = precisión; wave-3 = opcional.
Cada línea: `[x] COMPLETADA` / `[~] OLEADA 1` / `[~] OLEADA 2` / `[ ] PENDIENTE` · issue · oleada · título · rama.


## Milestone 1: Infraestructura, CI/CD, Fixtures Sintéticas y Puente de Evaluación

- [x] COMPLETADA · #01 · wave-1 · Inicialización de paquete Python, .gitignore anti-fugas y configuración tipada · `feat/issue-01-repo-scaffold`
- [x] COMPLETADA · #02 · wave-1 · GitHub Actions CI/CD, Security Gate anti-fuga de datos y automatización de las 28 Issues · `feat/issue-02-github-ci-security`
- [x] COMPLETADA · #03 · wave-1 · Generador determinista de partido sintético en formato FIFA/EPTS + SportsCode XML · `feat/issue-03-synthetic-match-generator`
- [x] COMPLETADA · #04 · wave-1 · Puente nativo con possession_metrics.py y validate_submission.py · `feat/issue-04-official-eval-bridge`

## Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV

- [x] COMPLETADA · #05 · wave-1 · Parser en streaming de ATD (.txt/.xml) a metros reales con caché .npy · `feat/issue-05-atd-streaming-loader`
- [x] COMPLETADA · #06 · wave-1 · Parser de Smart Tagging XML a máscaras y eventos por frame · `feat/issue-06-smart-tagging-loader`
- [x] COMPLETADA · #07 · wave-1 · Cargador vectorizado de matrices de homografía $H$ e inversas $H^{-1}$ · `feat/issue-07-homography-loader`
- [x] COMPLETADA · #08 · wave-1 · Filtrado de suplentes en banda y fusión de tracks duplicados ($< 0.6\text{ m}$) · `feat/issue-08-track-dedup-and-bounds`
- [x] COMPLETADA · #09 · wave-1 · Cinemática Savitzky-Golay $(\vec{v}, \vec{a})$ y normalización de orientación de ataque · `feat/issue-09-savgol-kinematics-orientation`
- [x] COMPLETADA · #10 · wave-3 · Sincronización de cámara de TV por bloques continuos e inyección de balón · `feat/issue-10-broadcast-block-sync`

## Milestone 3: Visión Guiada por Atención y Geometría 3D de Balón Aéreo

- [x] COMPLETADA · #11 · wave-3 · Generador de Attention Crops ($640\times640$) con homografía e interfaz de detector · `feat/issue-11-attention-crops-yolo`
- [x] COMPLETADA · #12 · wave-3 · Detector de balón aéreo (is_aerial) por velocidad aparente y curvatura · `feat/issue-12-aerial-ball-filter`

## Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva

- [x] COMPLETADA · #13 · wave-1 · Señales cinemáticas continuas de balón muerto (D) con y sin balón visible · `feat/issue-13-dead-ball-kinematic-signals`
- [x] COMPLETADA · #14 · wave-1 · Anclaje quirúrgico del frame de reinicio $t^*$ en los 59 SET PIECES y post-GOALS · `feat/issue-14-setpiece-exact-anchoring`
- [x] COMPLETADA · #15 · wave-2 · Resolución determinista del equipo lanzador en los 26 Throw-in con Team = N/A · `feat/issue-15-throwin-team-resolver`
- [x] COMPLETADA · #16 · wave-1 · Distancias por equipo, esfera de influencia y detector de toques (touch_score) · `feat/issue-16-ball-carrier-and-touches`
- [x] COMPLETADA · #17 · wave-2 · Física Colectiva I: Porteros dinámicos y derivada de área (Espectral + Convex Hull Numba) · `feat/issue-17-collective-physics-hull-gk`
- [x] COMPLETADA · #18 · wave-2 · Física Colectiva II: Coherencia vectorial $R \in [0,1]$ y memoria del último balón visto · `feat/issue-18-phase-coherence-and-ball-memory`

## Milestone 5: Degradador de Validación, Auto-Supervisión y Modelos Especialistas

- [x] COMPLETADA · #19 · wave-2 · Degradador realista de partidos limpios (huecos de 5–60 s + rotura de IDs) · `feat/issue-19-realistic-atd-degrader`
- [x] COMPLETADA · #20 · wave-2 · Modelo Especialista A (con balón visible): entrenamiento y calibración de probabilidades · `feat/issue-20-specialist-a-ball-visible`
- [x] COMPLETADA · #21 · wave-2 · Generador de Pseudo-Labeling con *Block Masking* (5–45 s) sobre el propio partido · `feat/issue-21-self-supervised-block-masking`
- [x] COMPLETADA · #22 · wave-2 · Modelo Especialista B (sin balón) y fusor de emisiones log_probs[N, 3] · `feat/issue-22-specialist-b-ball-absent`

## Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega

- [x] COMPLETADA · #23 · wave-1 · Constructor de la matriz de transición dependiente del tiempo $A_t \in \mathbb{R}^{N\times 3\times 3}$ · `feat/issue-23-dynamic-transition-matrix`
- [x] COMPLETADA · #24 · wave-1 · Decodificador Viterbi / CRF lineal con @numba.njit por mitad independiente · `feat/issue-24-numba-viterbi-decoder`
- [x] COMPLETADA · #25 · wave-1 · Post-procesador directo de spellF1: poda de micro-posesiones e inserción de puente D · `feat/issue-25-spell-pruning-and-d-bridge`
- [x] COMPLETADA · #26 · wave-1 · Consenso MBR de fronteras y contrato estricto de las 199.319 filas · `feat/issue-26-mbr-consensus-and-contract`
- [x] COMPLETADA · #27 · wave-2 · Cuadro de mando automatizado de los 6 KPIs de calidad · `feat/issue-27-six-kpi-dashboard`
- [x] COMPLETADA · #28 · wave-1/2 · CLI reproducible generate_submission.py, TECHNICAL_REPORT.md y test E2E de determinismo · `feat/issue-28-cli-submission-and-report`

## Milestone 7: Mejoras post-Optuna (oclusiones, rutas, MBR, duración)

- [x] COMPLETADA · #29 · wave-4 · Overrides de ruta heurística desde el barrido Optuna externo · `feat/issue-29-heuristic-route-overrides`
- [x] COMPLETADA · #30 · wave-4 · Giros de posesión con balón oculto: ventanas pasado/futuro para el Especialista B (adoptado) y apertura de transiciones guiada por B (medida, desactivada) · `feat/issue-30-occlusion-turnovers`
- [x] COMPLETADA · #31 · wave-4 · Overrides de ruta `specialists` · `feat/issue-31-specialists-route-overrides`
- [x] COMPLETADA · #32 · wave-4 · Consenso MBR sobre variantes de la ruta especialistas · `feat/issue-32-mbr-specialists`
- [x] COMPLETADA · #33 · wave-4 · Regenerar `reports/external_evidence.json` · `feat/issue-33-external-evidence-refresh`
- [x] COMPLETADA · #34 · wave-4 · Decodificador semi-Markov con distribución de duración (medido, desactivado) · `feat/issue-34-duration-aware-decoder`
- [x] COMPLETADA · #35 · wave-4 · Entrenar los especialistas con más partidos externos (PFF adoptado, SkillCorner desactivado) · `feat/issue-35-more-training-matches`
- [x] COMPLETADA · #36 · wave-4 · Detector de balón multi-frame (movimiento t-1,t,t+1, desactivado) y vídeo de auditoría con HUD · `feat/issue-36-multiframe-ball-and-visual-hud`

## Milestone 8: Datos reales (ingesta, Smart Tagging, sincronización TV)

- [x] COMPLETADA · #37 · wave-5 · Comprobación de ingesta real BAR-ATH con informe solo agregado · `feat/issue-37-real-data-sanity`
- [x] COMPLETADA · #38 · wave-4 · Degradador calibrado a las estadísticas de huecos del ATD real y reselección de modelo (medido; el modelo por defecto se mantiene, `specialists_calib.json` seleccionable) · `feat/issue-38-calibrated-degrader`
- [x] COMPLETADA · #39 · wave-5 · Sincronización TV en datos reales: cobertura y decisión de activación (medida, desactivada) · `feat/issue-39-broadcast-real`
- [x] COMPLETADA · #40 · wave-5 · Smart Tagging real: fase de juego con el equipo poseedor (fases de defensa invertidas), anclajes t* y saques de banda · `feat/issue-40-smart-tagging-real`

## Milestone 9: Análisis inteligente sobre datos facilitados e invarianza a caída de jugadores

- [x] COMPLETADA · #42 · wave-6 · Invarianza a caída de jugadores: diagnóstico por estratos, canal `s` nativo (m/s verificado), forma robusta al tamaño muestral (medido, no adoptado: desactivado por config) · `feat/issue-42-player-dropout-invariance`
- [x] COMPLETADA · #43 · wave-6 · Prior de balón vivo y transiciones desde Smart Tagging con balón oculto (simulador de tags; medido, no adoptado: desactivado por config) · `feat/issue-43-smart-tagging-alive-prior`
- [x] COMPLETADA · #44 · wave-6 · Re-optimización acotada del decodificador bajo el degradador calibrado y ensemble de modelos (barrido en external-benchmarks #5; mejor +0.0091 en B'(33) < +0.01, no adoptado: variante `--model-variant ensemble` y bloque `decoder_candidates.issue_44` desactivados) · `feat/issue-44-decoder-reopt`
- [x] COMPLETADA · #45 · wave-6 · Documentación y sincronización de todos los repos de la organización · `feat/issue-45-docs-and-org-repos-sync`
- [x] COMPLETADA · #46 · wave-6 · Cierre e integración del Milestone 9: docs finales, entrega regenerada con `--specialists load` (SHA256 doble), develop a main · `feat/issue-46-milestone9-release`
- [x] COMPLETADA · #47 · wave-6 · Auditoría completa contra el repositorio oficial del reto (`docs/OFFICIAL_AUDIT.md`) · `feat/issue-47-official-audit`
- [x] COMPLETADA · #48 · wave-6 · Colocación realista de huecos (afinidad balón oculto / balón muerto): degradador, estimador sin etiquetas validado, curva y reselección (adoptado: entrenamiento con afinidad 33) · `feat/issue-48-gap-placement`
- [x] COMPLETADA · #49 · wave-6 · Smart Tagging: reglas de equipo en reinicios (gol/2ª parte/voto con tracking), ventana de equipo más ancha, prior de posición de t* y banderas Long Ball (medido en B'/A con tags simulados, no adoptado: desactivado por config) · `feat/issue-49-restart-team-rules`
- [x] COMPLETADA · #50 · wave-6 · Anclajes de set pieces: acotar el fallback tag-window y fijar los Kick Off de inicio de periodo (Milestone 9) · `feat/issue-50-setpiece-anchor-fallback`
- [x] COMPLETADA · #51 · wave-6 · Aislar el front-end por play_blocks (sin fuga a través del descanso ni del calentamiento) · `feat/issue-51-play-blocks-isolation`
- [x] COMPLETADA · #52 · wave-6 · Reproducibilidad: `--specialists auto` carga siempre `models/specialists.json`; ruta real en diagnostics · `feat/issue-52-reproducible-specialists`
- [x] COMPLETADA · #53 · wave-6 · Docs: cita CC BY 4.0 de IDSSE (DOI) en README y enlaces de acceso público a IDSSE/PFF en Opus Operandi.md (issues oficiales #8 y #16) · `feat/issue-53-idsse-pff-citation`
- [x] COMPLETADA · #54 · wave-6 · Nombre de equipo exacto "Opus Operandi" (con espacio) en config, entregables (`Opus Operandi.csv`, `docs/Opus Operandi.md`) y docs · `feat/issue-54-team-name-spacing`

## Milestone 10: Experimentos post-congelación (rama aparte, adoptar solo si >= +0.010)

- [x] COMPLETADA · #55 · wave-5 · 56 partidos PFF en el entrenamiento (no adoptado) · `feat/issue-55-pff-all-matches`
- [x] COMPLETADA · #56 · wave-5 · Contexto multiescala y profundidad de porteros en el Especialista B (no adoptado) · `feat/issue-56-multiscale-gk-context`
- [x] COMPLETADA · #57 · wave-5 · Especialista C: puerta aprendida de cambios H<->A (no adoptado, desactivado) · `feat/issue-57-learned-touch-gate`
- [x] COMPLETADA · #58 · wave-5 · Chequeo sin etiquetas de detectores de balón en el vídeo táctico real (solo agregados; visión desactivada) · `feat/issue-58-real-video-ball-check`
- [x] COMPLETADA · #59 · wave-5 · Sesgo de latencia de las fronteras (diagnóstico; sin desplazamiento) · `feat/issue-59-latency-bias`
- [x] COMPLETADA · #60 · wave-5 · Bagging de semillas de los especialistas (no adoptado) · `feat/issue-60-seed-bagging`
- [x] COMPLETADA · #61 · wave-5 · Ajuste de fronteras al toque / reinicio (no adoptado, desactivado) · `feat/issue-61-boundary-snap-touch`
- [x] COMPLETADA · #62 · wave-5 · Decodificación por spellF1 esperado (MBR sobre el posterior; no adoptado, desactivado) · `feat/issue-62-expected-spellf1-decoding`
- [x] COMPLETADA · #63 · wave-5 · Detector de balón roboflow/sports + supervision en el vídeo real (no adoptado, desactivado) · `feat/issue-63-supervision-ball-detector`
- [x] COMPLETADA · #64 · wave-5 · Micro-geometría 22v22 de presión y portador implícito (no adoptado, desactivado) · `feat/issue-64-carrier-swarm-features`
- [x] COMPLETADA · #65 · wave-5 · Prior de fase de Smart Tagging en H vs A con balón oculto (no adoptado, desactivado) · `feat/issue-65-hidden-phase-prior`
- [x] COMPLETADA · #66 · wave-5 · Filtro por tracklets sobre sv_ball640 (simulado; no adoptado, desactivado) · `feat/issue-66-tracklet-gated-vision`
- [x] COMPLETADA · #67 · wave-5 · Techo oráculo del balón oculto, curva de precisión, descomposición del error y evidencia blanda (diagnóstico; desactivado) · `feat/issue-67-oracle-ceiling`
- [x] COMPLETADA · #68 · wave-5 · Detector de balón auto-supervisado en el vídeo táctico, GPU (no adoptado, desactivado) · `feat/issue-68-selfsup-ball-detector`
- [x] COMPLETADA · #69 · wave-5 · Integración del detector auto-supervisado: decisión de no adoptar (precisión ~15 % a 0.3 m frente a ~0.2 m necesarios; sin cambios en la ruta) · `feat/issue-69-detector-decision`
- [x] COMPLETADA · #70 · wave-5 · Investigación de otras fuentes de valor (notas y sondas agregadas; nada adoptado) · `feat/issue-70-value-research`
- [x] COMPLETADA · #71 · wave-5 · Selección de mundo sin etiquetas (validación adversarial, razón de densidades; sin variante adoptada) · `feat/issue-71-world-selection`
- [x] COMPLETADA · #72 · wave-5 · Modelo secuencial TCN (emisiones sin balón / detector de cambios); regla cumplida en externos (E100 B'(33) +0.0106), ADOPTADO: ensemble de 2 semillas, E100 (B'(33) +0.0130), activado por defecto; fallback `sequence_model.enabled: false` · `feat/issue-72-sequence-emissions`
- [x] COMPLETADA · #73 · wave-5 · Emisiones del TCN también en frames con balón visible (`visible_weight`, `emission_scale`); regla cumplida en externos, ADOPTADO V50s (visible_weight 0.5, escala 0.8; B'(33) +0.0568); fallback `visible_weight: 0.0, emission_scale: 0.5` (= #72) · `feat/issue-73-sequence-visible-emissions`
- [x] COMPLETADA · #74 · wave-5 · Reentrenamiento del TCN con features carrier/swarm (#64) y pérdida ponderada en fronteras (R64 / RB / R64B); NO ADOPTADO (mejor R64B: B'(33) +0.0089 < +0.010, B'(1) +0.0178, A +0.0067); producción sin cambios (#72/#73) · `feat/issue-74-sequence-retrain-boundaries`

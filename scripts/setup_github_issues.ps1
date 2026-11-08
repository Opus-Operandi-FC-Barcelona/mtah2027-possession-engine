param([string]$Repo = "Opus-Operandi-FC-Barcelona/mtah2027-possession-engine")
$ErrorActionPreference = "Stop"
# Creates the 6 milestones, the wave labels and the 28 atomic issues. Idempotent: existing
# milestones and issues (matched by title) are left untouched.
$existingMilestones = gh api "repos/$Repo/milestones?state=all&per_page=100" | ConvertFrom-Json
$existingIssues = gh issue list --repo $Repo --state all --limit 200 --json title | ConvertFrom-Json
foreach ($label in @(@("wave-1","0e8a16","MVP end-to-end"),@("wave-2","fbca04","Precision"),@("wave-3","c5def5","Optional"))) {
    gh label create $label[0] --repo $Repo --color $label[1] --description $label[2] --force | Out-Null
}
if (-not ($existingMilestones | Where-Object { $_.title -eq 'Milestone 1: Infraestructura, CI/CD, Fixtures Sintéticas y Puente de Evaluación' })) {
    gh api "repos/$Repo/milestones" -f ("title=" + 'Milestone 1: Infraestructura, CI/CD, Fixtures Sintéticas y Puente de Evaluación') | Out-Null
    Write-Host 'milestone: Milestone 1: Infraestructura, CI/CD, Fixtures Sintéticas y Puente de Evaluación'
}
if (-not ($existingMilestones | Where-Object { $_.title -eq 'Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV' })) {
    gh api "repos/$Repo/milestones" -f ("title=" + 'Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV') | Out-Null
    Write-Host 'milestone: Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV'
}
if (-not ($existingMilestones | Where-Object { $_.title -eq 'Milestone 3: Visión Guiada por Atención y Geometría 3D de Balón Aéreo' })) {
    gh api "repos/$Repo/milestones" -f ("title=" + 'Milestone 3: Visión Guiada por Atención y Geometría 3D de Balón Aéreo') | Out-Null
    Write-Host 'milestone: Milestone 3: Visión Guiada por Atención y Geometría 3D de Balón Aéreo'
}
if (-not ($existingMilestones | Where-Object { $_.title -eq 'Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva' })) {
    gh api "repos/$Repo/milestones" -f ("title=" + 'Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva') | Out-Null
    Write-Host 'milestone: Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva'
}
if (-not ($existingMilestones | Where-Object { $_.title -eq 'Milestone 5: Degradador de Validación, Auto-Supervisión y Modelos Especialistas' })) {
    gh api "repos/$Repo/milestones" -f ("title=" + 'Milestone 5: Degradador de Validación, Auto-Supervisión y Modelos Especialistas') | Out-Null
    Write-Host 'milestone: Milestone 5: Degradador de Validación, Auto-Supervisión y Modelos Especialistas'
}
if (-not ($existingMilestones | Where-Object { $_.title -eq 'Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega' })) {
    gh api "repos/$Repo/milestones" -f ("title=" + 'Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega') | Out-Null
    Write-Host 'milestone: Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega'
}
$body = @'
**Branch:** `feat/issue-01-repo-scaffold`

**Files:** `pyproject.toml`, `requirements.txt`, `.gitignore`, `configs/default.yaml`, `src/mtah2027/config.py`, `tests/test_01_config.py`

**Specification and acceptance:** Definir constantes inmutables (`N_VIDEO_FRAMES=199319`, `FPS=25`, `PITCH=(105.3, 68.0)`, `HALF1=(11185, 90651)`, `HALF2_DEAD=(110808, 113741)`, `HALF2_PLAY=(113742, 190197)`), fijador global de semillas `set_global_seeds(0)` e importación automática de `games.py`. **Test:** `pytest tests/test_01_config.py`.

**Milestone:** Milestone 1: Infraestructura, CI/CD, Fixtures Sintéticas y Puente de Evaluación

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#01 Inicialización de paquete Python, .gitignore anti-fugas y configuración tipada' })) {
    gh issue create --repo $Repo --title '#01 Inicialización de paquete Python, .gitignore anti-fugas y configuración tipada' --body $body --milestone 'Milestone 1: Infraestructura, CI/CD, Fixtures Sintéticas y Puente de Evaluación' --label wave-1 | Out-Null
    Write-Host 'issue: #01 Inicialización de paquete Python, .gitignore anti-fugas y configuración tipada'
}
$body = @'
**Branch:** `feat/issue-02-github-ci-security`

**Files:** `.github/workflows/ci.yml`, `.github/ISSUE_TEMPLATE/task.md`, `scripts/setup_github_issues.ps1`, `docs/ISSUES_TRACKER.md`, `tests/test_02_security_and_ci.py`

**Specification and acceptance:** Crear el workflow CI (sección 4), el script que registra las 28 Issues con `gh issue create` (o actualiza `docs/ISSUES_TRACKER.md`) y un test que verifica que `.gitignore` bloquea cualquier archivo en `data/`, `cache/`, `*.mp4`, `*_atd.txt` y `*.npy`. **Test:** `pytest tests/test_02_security_and_ci.py`.

**Milestone:** Milestone 1: Infraestructura, CI/CD, Fixtures Sintéticas y Puente de Evaluación

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#02 GitHub Actions CI/CD, Security Gate anti-fuga de datos y automatización de las 28 Issues' })) {
    gh issue create --repo $Repo --title '#02 GitHub Actions CI/CD, Security Gate anti-fuga de datos y automatización de las 28 Issues' --body $body --milestone 'Milestone 1: Infraestructura, CI/CD, Fixtures Sintéticas y Puente de Evaluación' --label wave-1 | Out-Null
    Write-Host 'issue: #02 GitHub Actions CI/CD, Security Gate anti-fuga de datos y automatización de las 28 Issues'
}
$body = @'
**Branch:** `feat/issue-03-synthetic-match-generator`

**Files:** `scripts/generate_synthetic_fixtures.py`, `tests/conftest.py`, `tests/test_03_synthetic_fixtures.py`

**Specification and acceptance:** Generar en memoria/directorio temporal un mini-partido sintético con `atd.txt` (`<frame_id>:<46 slots>:<ball>`), `atd.xml`, `homography.csv`, `smart_tagging.xml` (con `SET PIECES`, `Throw-in` con `Team=N/A` y `GOALS`) y `broadcast_atd` desfasado con cortes. **Test:** `pytest tests/test_03_synthetic_fixtures.py`.

**Milestone:** Milestone 1: Infraestructura, CI/CD, Fixtures Sintéticas y Puente de Evaluación

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#03 Generador determinista de partido sintético en formato FIFA/EPTS + SportsCode XML' })) {
    gh issue create --repo $Repo --title '#03 Generador determinista de partido sintético en formato FIFA/EPTS + SportsCode XML' --body $body --milestone 'Milestone 1: Infraestructura, CI/CD, Fixtures Sintéticas y Puente de Evaluación' --label wave-1 | Out-Null
    Write-Host 'issue: #03 Generador determinista de partido sintético en formato FIFA/EPTS + SportsCode XML'
}
$body = @'
**Branch:** `feat/issue-04-official-eval-bridge`

**Files:** `src/mtah2027/validation/eval_bridge.py`, `tests/test_04_eval_bridge.py`

**Specification and acceptance:** Envolver `evaluate()`, `spell_scores()`, `macro_f1()` y `validate_submission()` del repo oficial para poder evaluar arrays en memoria en milisegundos durante validación y tests. **Test:** `pytest tests/test_04_eval_bridge.py`.

**Milestone:** Milestone 1: Infraestructura, CI/CD, Fixtures Sintéticas y Puente de Evaluación

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#04 Puente nativo con possession_metrics.py y validate_submission.py' })) {
    gh issue create --repo $Repo --title '#04 Puente nativo con possession_metrics.py y validate_submission.py' --body $body --milestone 'Milestone 1: Infraestructura, CI/CD, Fixtures Sintéticas y Puente de Evaluación' --label wave-1 | Out-Null
    Write-Host 'issue: #04 Puente nativo con possession_metrics.py y validate_submission.py'
}
$body = @'
**Branch:** `feat/issue-05-atd-streaming-loader`

**Files:** `src/mtah2027/io/atd_loader.py`, `tests/test_05_atd_loader.py`

**Specification and acceptance:** Leer `atd.xml` (asignando `0=HOME`, `1=AWAY` a cada uno de los 46 slots) y parsear `atd.txt` en una sola pasada convirtiendo coordenadas `[0,1]` a metros (`105.3 × 68.0`), indexando por `video_frame = frame_id - 1` y guardando/cargando en `cache/*.npy`. **Test:** `pytest tests/test_05_atd_loader.py`.

**Milestone:** Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#05 Parser en streaming de ATD (.txt/.xml) a metros reales con caché .npy' })) {
    gh issue create --repo $Repo --title '#05 Parser en streaming de ATD (.txt/.xml) a metros reales con caché .npy' --body $body --milestone 'Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV' --label wave-1 | Out-Null
    Write-Host 'issue: #05 Parser en streaming de ATD (.txt/.xml) a metros reales con caché .npy'
}
$body = @'
**Branch:** `feat/issue-06-smart-tagging-loader`

**Files:** `src/mtah2027/io/smart_tagging_loader.py`, `tests/test_06_smart_tagging_loader.py`

**Specification and acceptance:** Parsear `<instance>` de `smart_tagging.xml` con `round(seconds * 25)`, normalizar `code` y `tags` sin sensibilidad a mayúsculas, mapear `Barcelona -> H`, `Atletic Club -> A`, `N/A -> -1`, y generar tensores por frame de fase activa y lista estructurada de `SET PIECES` y `GOALS`. **Test:** `pytest tests/test_06_smart_tagging_loader.py`.

**Milestone:** Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#06 Parser de Smart Tagging XML a máscaras y eventos por frame' })) {
    gh issue create --repo $Repo --title '#06 Parser de Smart Tagging XML a máscaras y eventos por frame' --body $body --milestone 'Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV' --label wave-1 | Out-Null
    Write-Host 'issue: #06 Parser de Smart Tagging XML a máscaras y eventos por frame'
}
$body = @'
**Branch:** `feat/issue-07-homography-loader`

**Files:** `src/mtah2027/io/homography_loader.py`, `tests/test_07_homography_loader.py`

**Specification and acceptance:** Cargar `homography.csv` como `H[N, 3, 3]`, precalcular inversas vectorizadas `H_inv[N, 3, 3]` donde `valid == 1`, e implementar `project_pitch_to_pixels(H, x_m, y_m)` y `project_pixels_to_pitch(H_inv, u_px, v_px)` convirtiendo internamente entre metros y $[0,1]^2$. **Test:** `pytest tests/test_07_homography_loader.py`.

**Milestone:** Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#07 Cargador vectorizado de matrices de homografía $H$ e inversas $H^{-1}$' })) {
    gh issue create --repo $Repo --title '#07 Cargador vectorizado de matrices de homografía $H$ e inversas $H^{-1}$' --body $body --milestone 'Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV' --label wave-1 | Out-Null
    Write-Host 'issue: #07 Cargador vectorizado de matrices de homografía $H$ e inversas $H^{-1}$'
}
$body = @'
**Branch:** `feat/issue-08-track-dedup-and-bounds`

**Files:** `src/mtah2027/preprocessing/clean_tracks.py`, `tests/test_08_clean_tracks.py`

**Specification and acceptance:** Enmascarar como `NaN` cualquier slot de jugador fuera de $[-2.0\text{ m}, 107.3\text{ m}] \times [-2.0\text{ m}, 70.0\text{ m}]$ y, cuando dos slots activos del mismo equipo estén a $< 0.6\text{ m}$ en el mismo frame, desactivar el duplicado conservando el de trayectoria más larga. **Test:** `pytest tests/test_08_clean_tracks.py`.

**Milestone:** Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#08 Filtrado de suplentes en banda y fusión de tracks duplicados ($< 0.6\text{ m}$)' })) {
    gh issue create --repo $Repo --title '#08 Filtrado de suplentes en banda y fusión de tracks duplicados ($< 0.6\text{ m}$)' --body $body --milestone 'Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV' --label wave-1 | Out-Null
    Write-Host 'issue: #08 Filtrado de suplentes en banda y fusión de tracks duplicados ($< 0.6\text{ m}$)'
}
$body = @'
**Branch:** `feat/issue-09-savgol-kinematics-orientation`

**Files:** `src/mtah2027/preprocessing/kinematics_and_orientation.py`, `tests/test_09_kinematics_orientation.py`

**Specification and acceptance:** (1) Interpolar huecos microscópicos ($\le 5$ frames = $0.2\text{ s}$) y calcular $\vec{v}, \vec{a}$ de balón y jugadores con filtro Savitzky-Golay (`window=7, poly=2`). (2) Detectar el lado de cada equipo en cada mitad y reflejar $(x \to 105.3 - x, y \to 68.0 - y, v_x \to -v_x, v_y \to -v_y)$ para que `H` ataque siempre hacia $+x$ en ambas mitades. **Test:** `pytest tests/test_09_kinematics_orientation.py`.

**Milestone:** Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#09 Cinemática Savitzky-Golay $(\vec{v}, \vec{a})$ y normalización de orientación de ataque' })) {
    gh issue create --repo $Repo --title '#09 Cinemática Savitzky-Golay $(\vec{v}, \vec{a})$ y normalización de orientación de ataque' --body $body --milestone 'Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV' --label wave-1 | Out-Null
    Write-Host 'issue: #09 Cinemática Savitzky-Golay $(\vec{v}, \vec{a})$ y normalización de orientación de ataque'
}
$body = @'
**Branch:** `feat/issue-10-broadcast-block-sync`

**Files:** `src/mtah2027/preprocessing/broadcast_sync.py`, `tests/test_10_broadcast_sync.py`

**Specification and acceptance:** Segmentar el ATD de TV en tramos continuos con `valid == 1` y `reprojection_error < 5.0` ($\ge 75$ frames). Calcular la huella 1D invariante a recorte (velocidad media visible + mediana de distancia al rival más cercano), hallar $\Delta t_{\text{block}}$ por correlación cruzada z-normalizada e inyectar `ball_xy` de TV donde el táctico es `NaN` (si correlación $\ge 0.65$). **Test:** `pytest tests/test_10_broadcast_sync.py`.

**Milestone:** Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#10 Sincronización de cámara de TV por bloques continuos e inyección de balón' })) {
    gh issue create --repo $Repo --title '#10 Sincronización de cámara de TV por bloques continuos e inyección de balón' --body $body --milestone 'Milestone 2: Ingesta en Streaming, Limpieza de Slots y Sincronización de TV' --label wave-3 | Out-Null
    Write-Host 'issue: #10 Sincronización de cámara de TV por bloques continuos e inyección de balón'
}
$body = @'
**Branch:** `feat/issue-11-attention-crops-yolo`

**Files:** `src/mtah2027/vision/attention_crops.py`, `tests/test_11_attention_crops.py`

**Specification and acceptance:** Para cada frame sin balón, calcular el centroide de presión (jugadores con mayor velocidad convergente + interpolación entre bordes del hueco), proyectarlo a píxeles $(u, v)$ con $H$, recortar ventana $640\times640$ limitada a $1920\times1080$ y reproyectar detecciones al campo con $H^{-1}$. Funciona con detector real o en modo *dry-run/tracking-only*. **Test:** `pytest tests/test_11_attention_crops.py`.

**Milestone:** Milestone 3: Visión Guiada por Atención y Geometría 3D de Balón Aéreo

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#11 Generador de Attention Crops ($640\times640$) con homografía e interfaz de detector' })) {
    gh issue create --repo $Repo --title '#11 Generador de Attention Crops ($640\times640$) con homografía e interfaz de detector' --body $body --milestone 'Milestone 3: Visión Guiada por Atención y Geometría 3D de Balón Aéreo' --label wave-3 | Out-Null
    Write-Host 'issue: #11 Generador de Attention Crops ($640\times640$) con homografía e interfaz de detector'
}
$body = @'
**Branch:** `feat/issue-12-aerial-ball-filter`

**Files:** `src/mtah2027/vision/aerial_filter.py`, `tests/test_12_aerial_filter.py`

**Specification and acceptance:** Marcar `is_aerial = True` cuando el balón proyectado supera $35\text{ m/s}$, presenta saltos de reproyección o describe una trayectoria larga ($> 18\text{ m}$) sin desaceleración de rozamiento con el césped. Exportar máscara booleana `is_aerial[N]`. **Test:** `pytest tests/test_12_aerial_filter.py`.

**Milestone:** Milestone 3: Visión Guiada por Atención y Geometría 3D de Balón Aéreo

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#12 Detector de balón aéreo (is_aerial) por velocidad aparente y curvatura' })) {
    gh issue create --repo $Repo --title '#12 Detector de balón aéreo (is_aerial) por velocidad aparente y curvatura' --body $body --milestone 'Milestone 3: Visión Guiada por Atención y Geometría 3D de Balón Aéreo' --label wave-3 | Out-Null
    Write-Host 'issue: #12 Detector de balón aéreo (is_aerial) por velocidad aparente y curvatura'
}
$body = @'
**Branch:** `feat/issue-13-dead-ball-kinematic-signals`

**Files:** `src/mtah2027/features/dead_ball_signals.py`, `tests/test_13_dead_ball_signals.py`

**Specification and acceptance:** Calcular por frame: (a) indicador de balón fuera de $[0, 105.3]\times[0, 68.0]$, (b) duración acumulada de balón inmóvil ($v_{\text{ball}} < 0.4\text{ m/s}$), (c) velocidad media y percentil 80 de los 22 jugadores (detecta paradas aunque el balón sea visible o esté quieto), (d) geometría de saque de centro/puerta/córner. **Test:** `pytest tests/test_13_dead_ball_signals.py`.

**Milestone:** Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#13 Señales cinemáticas continuas de balón muerto (D) con y sin balón visible' })) {
    gh issue create --repo $Repo --title '#13 Señales cinemáticas continuas de balón muerto (D) con y sin balón visible' --body $body --milestone 'Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva' --label wave-1 | Out-Null
    Write-Host 'issue: #13 Señales cinemáticas continuas de balón muerto (D) con y sin balón visible'
}
$body = @'
**Branch:** `feat/issue-14-setpiece-exact-anchoring`

**Files:** `src/mtah2027/features/setpiece_anchor.py`, `tests/test_14_setpiece_anchor.py`

**Specification and acceptance:** Dentro de cada ventana `[start_frame, end_frame]` de `SET PIECES`, localizar el frame exacto $t^*$ donde el balón pasa de reposo ($<1.0\text{ m/s}$) a lanzamiento ($>4.0\text{ m/s}$) o donde la velocidad colectiva da un salto positivo. Marcar como `D` los frames previos a $t^*$ y fijar `D` entre cada `GOALS` y el $t^*$ del siguiente `Kick Off (After Goal)`. **Test:** `pytest tests/test_14_setpiece_anchor.py`.

**Milestone:** Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#14 Anclaje quirúrgico del frame de reinicio $t^*$ en los 59 SET PIECES y post-GOALS' })) {
    gh issue create --repo $Repo --title '#14 Anclaje quirúrgico del frame de reinicio $t^*$ en los 59 SET PIECES y post-GOALS' --body $body --milestone 'Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva' --label wave-1 | Out-Null
    Write-Host 'issue: #14 Anclaje quirúrgico del frame de reinicio $t^*$ en los 59 SET PIECES y post-GOALS'
}
$body = @'
**Branch:** `feat/issue-15-throwin-team-resolver`

**Files:** `src/mtah2027/features/throwin_resolver.py`, `tests/test_15_throwin_resolver.py`

**Specification and acceptance:** Para cada `SET PIECES` de tipo `Throw-in` (`Team = N/A`), identificar en `[start_frame, t*]` al jugador situado a $< 1.5\text{ m}$ de la banda ($y \approx 0$ o $y \approx 68$) más aislado de rivales y confirmar con el equipo del primer jugador que controla en $[t^*, t^* + 35]$. **Test:** `pytest tests/test_15_throwin_resolver.py`.

**Milestone:** Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#15 Resolución determinista del equipo lanzador en los 26 Throw-in con Team = N/A' })) {
    gh issue create --repo $Repo --title '#15 Resolución determinista del equipo lanzador en los 26 Throw-in con Team = N/A' --body $body --milestone 'Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva' --label wave-2 | Out-Null
    Write-Host 'issue: #15 Resolución determinista del equipo lanzador en los 26 Throw-in con Team = N/A'
}
$body = @'
**Branch:** `feat/issue-16-ball-carrier-and-touches`

**Files:** `src/mtah2027/features/ball_kinematics.py`, `tests/test_16_ball_kinematics.py`

**Specification and acceptance:** Calcular de forma 100% vectorizada: distancias $d_H^{(1..3)}, d_A^{(1..3)}$ a los 3 jugadores más cercanos de cada equipo, diferencia $\Delta d = d_A^{(1)} - d_H^{(1)}$, velocidad relativa al balón, peso de control de Link & Hoernig y `touch_score[N]` (pico de $\|\vec{a}_{\text{ball}}\| > 12\text{ m/s}^2$ o giro $> 35^\circ$ con jugador a $< 1.8\text{ m}$ y `~is_aerial`). **Test:** `pytest tests/test_16_ball_kinematics.py`.

**Milestone:** Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#16 Distancias por equipo, esfera de influencia y detector de toques (touch_score)' })) {
    gh issue create --repo $Repo --title '#16 Distancias por equipo, esfera de influencia y detector de toques (touch_score)' --body $body --milestone 'Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva' --label wave-1 | Out-Null
    Write-Host 'issue: #16 Distancias por equipo, esfera de influencia y detector de toques (touch_score)'
}
$body = @'
**Branch:** `feat/issue-17-collective-physics-hull-gk`

**Files:** `src/mtah2027/features/collective_physics.py`, `tests/test_17_collective_physics.py`

**Specification and acceptance:** (1) Identificar en cada frame al portero de `H` (mínimo $x$) y de `A` (máximo $x$) y extraer su posición y velocidad. (2) Excluir al portero y calcular para los jugadores de campo de cada equipo el área espectral $\pi\sqrt{\lambda_1\lambda_2}$, el área de la envolvente convexa compilada con `@numba.njit` (algoritmo Monotone Chain) y sus derivadas a 2 segundos ($\Delta A_H, \Delta A_A$). **Test:** `pytest tests/test_17_collective_physics.py`.

**Milestone:** Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#17 Física Colectiva I: Porteros dinámicos y derivada de área (Espectral + Convex Hull Numba)' })) {
    gh issue create --repo $Repo --title '#17 Física Colectiva I: Porteros dinámicos y derivada de área (Espectral + Convex Hull Numba)' --body $body --milestone 'Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva' --label wave-2 | Out-Null
    Write-Host 'issue: #17 Física Colectiva I: Porteros dinámicos y derivada de área (Espectral + Convex Hull Numba)'
}
$body = @'
**Branch:** `feat/issue-18-phase-coherence-and-ball-memory`

**Files:** `src/mtah2027/features/ball_memory.py`, `tests/test_18_ball_memory.py`

**Specification and acceptance:** (1) Calcular la coherencia de fase de velocidades $R_{\text{team}} = \|\frac{1}{N}\sum \vec{v}_i / (\|\vec{v}_i\|+10^{-3})\|$ (inmune a la discontinuidad en $\pm\pi$) y el avance medio de líneas $\bar{v}_{x, H}, \bar{v}_{x, A}$. (2) Propagar hacia delante y hacia atrás la última/próxima coordenada conocida del balón ($\Delta t_{\text{lost}}, x_{\text{last}}, y_{\text{last}}$, último poseedor claro y densidad de presión alrededor). **Test:** `pytest tests/test_18_ball_memory.py`.

**Milestone:** Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#18 Física Colectiva II: Coherencia vectorial $R \in [0,1]$ y memoria del último balón visto' })) {
    gh issue create --repo $Repo --title '#18 Física Colectiva II: Coherencia vectorial $R \in [0,1]$ y memoria del último balón visto' --body $body --milestone 'Milestone 4: Balón Muerto (D), Anclaje de Set Pieces y Física Colectiva' --label wave-2 | Out-Null
    Write-Host 'issue: #18 Física Colectiva II: Coherencia vectorial $R \in [0,1]$ y memoria del último balón visto'
}
$body = @'
**Branch:** `feat/issue-19-realistic-atd-degrader`

**Files:** `src/mtah2027/models/external_degrader.py`, `tests/test_19_external_degrader.py`

**Specification and acceptance:** Dado un partido con balón completo, enmascarar el 41% de los frames en bloques continuos de duración muestreada de la distribución empírica del ATD (5–20 s: 33%, 20–60 s: 48%, >60 s: 19%), con probabilidad proporcional a la densidad de jugadores en un radio de $5\text{ m}$, y permutar slots en los cortes. **Test:** `pytest tests/test_19_external_degrader.py`.

**Milestone:** Milestone 5: Degradador de Validación, Auto-Supervisión y Modelos Especialistas

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#19 Degradador realista de partidos limpios (huecos de 5–60 s + rotura de IDs)' })) {
    gh issue create --repo $Repo --title '#19 Degradador realista de partidos limpios (huecos de 5–60 s + rotura de IDs)' --body $body --milestone 'Milestone 5: Degradador de Validación, Auto-Supervisión y Modelos Especialistas' --label wave-2 | Out-Null
    Write-Host 'issue: #19 Degradador realista de partidos limpios (huecos de 5–60 s + rotura de IDs)'
}
$body = @'
**Branch:** `feat/issue-20-specialist-a-ball-visible`

**Files:** `src/mtah2027/models/specialist_ball_present.py`, `tests/test_20_specialist_ball_present.py`

**Specification and acceptance:** Clasificador `LightGBM` (con fallback a `HistGradientBoostingClassifier` de scikit-learn) que combina las señales de los módulos #13–#18 cuando el balón es visible y devuelve probabilidades calibradas `P_vis(H, A, D)` y una máscara de anclas de ultra-alta confianza ($P > 0.98$ sostenido $\ge 2\text{ s}$). **Test:** `pytest tests/test_20_specialist_ball_present.py`.

**Milestone:** Milestone 5: Degradador de Validación, Auto-Supervisión y Modelos Especialistas

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#20 Modelo Especialista A (con balón visible): entrenamiento y calibración de probabilidades' })) {
    gh issue create --repo $Repo --title '#20 Modelo Especialista A (con balón visible): entrenamiento y calibración de probabilidades' --body $body --milestone 'Milestone 5: Degradador de Validación, Auto-Supervisión y Modelos Especialistas' --label wave-2 | Out-Null
    Write-Host 'issue: #20 Modelo Especialista A (con balón visible): entrenamiento y calibración de probabilidades'
}
$body = @'
**Branch:** `feat/issue-21-self-supervised-block-masking`

**Files:** `src/mtah2027/models/pseudo_labeling.py`, `tests/test_21_pseudo_labeling.py`

**Specification and acceptance:** Tomar los tramos del propio partido donde el Especialista A tiene confianza $> 0.98$, aplicar *block masking* de 5 a 45 s borrando las variables directas del balón y recalculando `ball_memory` (#18) desde el inicio de cada máscara artificial para crear el dataset de entrenamiento auto-supervisado del partido. **Test:** `pytest tests/test_21_pseudo_labeling.py`.

**Milestone:** Milestone 5: Degradador de Validación, Auto-Supervisión y Modelos Especialistas

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#21 Generador de Pseudo-Labeling con *Block Masking* (5–45 s) sobre el propio partido' })) {
    gh issue create --repo $Repo --title '#21 Generador de Pseudo-Labeling con *Block Masking* (5–45 s) sobre el propio partido' --body $body --milestone 'Milestone 5: Degradador de Validación, Auto-Supervisión y Modelos Especialistas' --label wave-2 | Out-Null
    Write-Host 'issue: #21 Generador de Pseudo-Labeling con *Block Masking* (5–45 s) sobre el propio partido'
}
$body = @'
**Branch:** `feat/issue-22-specialist-b-ball-absent`

**Files:** `src/mtah2027/models/specialist_ball_absent.py`, `tests/test_22_specialist_ball_absent.py`

**Specification and acceptance:** Entrenar el Especialista B exclusivamente con Física Colectiva (#17), Coherencia y Memoria (#18), Señales de `D` (#13–#15) y priors suaves de Smart Tagging (#6) sobre el dataset auto-supervisado (#21). Fusionar Especialista A (en frames con balón) y Especialista B (en frames sin balón) en `log_probs[N, 3]` cuantizados a `1e-6`. **Test:** `pytest tests/test_22_specialist_ball_absent.py`.

**Milestone:** Milestone 5: Degradador de Validación, Auto-Supervisión y Modelos Especialistas

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#22 Modelo Especialista B (sin balón) y fusor de emisiones log_probs[N, 3]' })) {
    gh issue create --repo $Repo --title '#22 Modelo Especialista B (sin balón) y fusor de emisiones log_probs[N, 3]' --body $body --milestone 'Milestone 5: Degradador de Validación, Auto-Supervisión y Modelos Especialistas' --label wave-2 | Out-Null
    Write-Host 'issue: #22 Modelo Especialista B (sin balón) y fusor de emisiones log_probs[N, 3]'
}
$body = @'
**Branch:** `feat/issue-23-dynamic-transition-matrix`

**Files:** `src/mtah2027/decoding/transition_matrix.py`, `tests/test_23_transition_matrix.py`

**Specification and acceptance:** Construir $\log A_t(i, j)$ para cada frame $t$: en reposo $\log A_t(H, A) = -11.5$; en ventana de $\pm 5$ frames alrededor de un pico de `touch_score` con ambos equipos a $< 2.0\text{ m}$ del balón no aéreo, abrir a $-1.4$; modular $\log A_t(\cdot, D)$ con salida de campo/parada y abrir $\log A_t(D, \text{Team})$ en los frames de reinicio $t^*$ (#14). **Test:** `pytest tests/test_23_transition_matrix.py`.

**Milestone:** Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#23 Constructor de la matriz de transición dependiente del tiempo $A_t \in \mathbb{R}^{N\times 3\times 3}$' })) {
    gh issue create --repo $Repo --title '#23 Constructor de la matriz de transición dependiente del tiempo $A_t \in \mathbb{R}^{N\times 3\times 3}$' --body $body --milestone 'Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega' --label wave-1 | Out-Null
    Write-Host 'issue: #23 Constructor de la matriz de transición dependiente del tiempo $A_t \in \mathbb{R}^{N\times 3\times 3}$'
}
$body = @'
**Branch:** `feat/issue-24-numba-viterbi-decoder`

**Files:** `src/mtah2027/decoding/dynamic_viterbi.py`, `tests/test_24_dynamic_viterbi.py`

**Specification and acceptance:** Implementar `viterbi_dynamic(log_probs, log_trans)` compilado con `@numba.njit` (y fallback idéntico en NumPy) y `decode_match_halves()` que decodifica de forma totalmente separada `11185..90651` y `113742..190197` en $< 0.2\text{ s}$. **Test:** `pytest tests/test_24_dynamic_viterbi.py`.

**Milestone:** Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#24 Decodificador Viterbi / CRF lineal con @numba.njit por mitad independiente' })) {
    gh issue create --repo $Repo --title '#24 Decodificador Viterbi / CRF lineal con @numba.njit por mitad independiente' --body $body --milestone 'Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega' --label wave-1 | Out-Null
    Write-Host 'issue: #24 Decodificador Viterbi / CRF lineal con @numba.njit por mitad independiente'
}
$body = @'
**Branch:** `feat/issue-25-spell-pruning-and-d-bridge`

**Files:** `src/mtah2027/postprocessing/spell_optimizer.py`, `tests/test_25_spell_optimizer.py`

**Specification and acceptance:** (1) Absorber micro-posesiones $< 1.2\text{ s}$ (30 frames) de tipo `H -> A -> H` o `A -> H -> A` que ocurran sin balón visible o sin doble pico de `touch_score`. (2) En posesiones de un mismo equipo $> 25\text{ s}$ que contengan un valle de $\ge 4\text{ s}$ con alta señal de balón muerto + banda/`SET PIECES`, insertar el puente `D` (`H -> D -> H`). **Test:** `pytest tests/test_25_spell_optimizer.py`.

**Milestone:** Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#25 Post-procesador directo de spellF1: poda de micro-posesiones e inserción de puente D' })) {
    gh issue create --repo $Repo --title '#25 Post-procesador directo de spellF1: poda de micro-posesiones e inserción de puente D' --body $body --milestone 'Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega' --label wave-1 | Out-Null
    Write-Host 'issue: #25 Post-procesador directo de spellF1: poda de micro-posesiones e inserción de puente D'
}
$body = @'
**Branch:** `feat/issue-26-mbr-consensus-and-contract`

**Files:** `src/mtah2027/decoding/mbr_consensus.py`, `src/mtah2027/postprocessing/submission_contract.py`, `tests/test_26_mbr_and_contract.py`

**Specification and acceptance:** (1) `mbr_consensus`: emparejar segmentos de múltiples corridas/modelos y tomar la mediana de `start_frame` y `end_frame`. (2) `enforce_submission_contract`: rellenar hacia delante cualquier `X` interno en `11185..90651` y `110808..190197`, forzar `D` en `110808..113741`, forzar `X` en `0..11184`, `90652..110807` y `190198..199318`, y escribir `frame,label` validado. **Test:** `pytest tests/test_26_mbr_and_contract.py`.

**Milestone:** Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#26 Consenso MBR de fronteras y contrato estricto de las 199.319 filas' })) {
    gh issue create --repo $Repo --title '#26 Consenso MBR de fronteras y contrato estricto de las 199.319 filas' --body $body --milestone 'Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega' --label wave-1 | Out-Null
    Write-Host 'issue: #26 Consenso MBR de fronteras y contrato estricto de las 199.319 filas'
}
$body = @'
**Branch:** `feat/issue-27-six-kpi-dashboard`

**Files:** `src/mtah2027/validation/kpi_dashboard.py`, `scripts/run_kpi_dashboard.py`, `tests/test_27_kpi_dashboard.py`

**Specification and acceptance:** Implementar los 6 KPIs con salida en tabla y JSON: (1) Sanidad macroscópica **como alarma, no objetivo** (rangos amplios en `configs/default.yaml`: `150..600` posesiones, `6..30 s` de duración media, `10..45%` `D` sobre frames evaluados, `35..75%` `H` sobre frames vivos), (2) Test de ceguera artificial en los primeros 20 min (`spellF1 >= 0.75`), (3) Evaluación en partido degradado (`spellF1 >= 0.65`, latencia `< 8` frames), (4) Smoke test en los 5 min públicos con bootstrap, (5) Exportación de 20 transiciones para auditoría visual, (6) Verificación de `validate_submission.py` y `SHA256`. **Test:** `pytest tests/test_27_kpi_dashboard.py`.

**Milestone:** Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#27 Cuadro de mando automatizado de los 6 KPIs de calidad' })) {
    gh issue create --repo $Repo --title '#27 Cuadro de mando automatizado de los 6 KPIs de calidad' --body $body --milestone 'Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega' --label wave-2 | Out-Null
    Write-Host 'issue: #27 Cuadro de mando automatizado de los 6 KPIs de calidad'
}
$body = @'
**Branch:** `feat/issue-28-cli-submission-and-report`

**Files:** `scripts/generate_submission.py`, `docs/TECHNICAL_REPORT.md`, `README.md`, `tests/test_28_e2e_reproducibility.py`

**Specification and acceptance:** Conectar de principio a fin las Issues #1–#27 en `scripts/generate_submission.py` (con soporte para `--data-dir`, `--out` y `--synthetic`), redactar `docs/TECHNICAL_REPORT.md` cubriendo todos los puntos de `SUBMISSION.md`, y verificar en `test_28_e2e_reproducibility.py` que dos ejecuciones seguidas producen exactamente el mismo hash `SHA256`. **Test:** `pytest tests/test_28_e2e_reproducibility.py`.

**Milestone:** Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega

_Source: claude.md section 3._
'@
if (-not ($existingIssues | Where-Object { $_.title -eq '#28 CLI reproducible generate_submission.py, TECHNICAL_REPORT.md y test E2E de determinismo' })) {
    gh issue create --repo $Repo --title '#28 CLI reproducible generate_submission.py, TECHNICAL_REPORT.md y test E2E de determinismo' --body $body --milestone 'Milestone 6: Viterbi Dinámico A_t, Optimización Directa de spellF1, KPIs y Entrega' --label wave-1 | Out-Null
    Write-Host 'issue: #28 CLI reproducible generate_submission.py, TECHNICAL_REPORT.md y test E2E de determinismo'
}

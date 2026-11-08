param([string]$OrgName = "Opus-Operandi-FC-Barcelona")
Write-Host "Aprovisionando Organización de GitHub: $OrgName"

# 0. Portada de la organización (.github)
gh repo create "$OrgName/.github" --public --description "Official Organization Profile — Opus Operandi | More than a Hack 2027 (FC Barcelona & Metrica Sports)"

# 1. Repositorio Principal Entregable (Fase 1)
gh repo create "$OrgName/mtah2027-possession-engine" --private --description "Phase 1 Official Deliverable: Deterministic Frame-Level Ball Possession Engine (BAR-ATH)"

# 2. Repositorio de Benchmarks Externos (IDSSE / Metrica Sample Data)
gh repo create "$OrgName/mtah2027-external-benchmarks" --private --description "External Tracking Benchmarks (IDSSE Bundesliga & Metrica Open Data) + Degrader & Optuna Tuning"

# 3. Sandbox Privado del Equipo (NUNCA hacer público)
gh repo create "$OrgName/mtah2027-sandbox-private" --private --description "STRICTLY PRIVATE: Internal Team Notebooks, KPI Reports & Visual Transition Audits"

# 4. Repositorio de Innovación (Fase 2 - Talent Arena Pitch)
gh repo create "$OrgName/mtah2027-phase2-innovation" --private --description "Phase 2 Innovation Challenge: Interactive Tactical Intelligence App & Final Pitch"

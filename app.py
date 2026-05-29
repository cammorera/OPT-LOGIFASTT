"""
app.py — Streamlit App: Optimización Cross-Docking LogiFast CR
==============================================================
Ejecutar con:
    streamlit run app.py
"""

import io
import textwrap
import time

import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

from modelo import (
    BIG_M, T_CHANGEOVER, T_TRANSFER, T_UNIT,
    InstanceData, SolverResult,
    build_and_solve, parse_ts5,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LogiFast CR — Cross Docking Optimizer",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

  .main-title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #FF6B2B 0%, #FF9F1C 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .subtitle {
    font-size: 1rem; color: #888; margin-bottom: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #FF6B2B33;
    border-radius: 12px; padding: 1.2rem 1.5rem;
    text-align: center; margin-bottom: 1rem;
  }
  .metric-value {
    font-size: 2.4rem; font-weight: 800; color: #FF6B2B;
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-label { font-size: 0.78rem; color: #aaa; text-transform: uppercase; letter-spacing: 0.1em; }
  .order-badge {
    display: inline-block;
    background: #FF6B2B22; border: 1px solid #FF6B2B66;
    color: #FF9F1C; border-radius: 6px;
    padding: 0.25rem 0.75rem; margin: 0.2rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;
  }
  .section-header {
    font-size: 1.1rem; font-weight: 600;
    border-left: 4px solid #FF6B2B;
    padding-left: 0.75rem; margin: 1.5rem 0 0.75rem;
  }
  .info-box {
    background: #0d1117; border: 1px solid #30363d;
    border-radius: 8px; padding: 1rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    color: #8b949e; white-space: pre-wrap;
  }
  .stButton > button {
    background: linear-gradient(135deg, #FF6B2B, #FF9F1C) !important;
    color: white !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.6rem 2rem !important; font-family: 'Syne', sans-serif !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }
  .sidebar-section { margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚛 LogiFast CR")
    st.markdown("**Optimizador Cross-Docking**")
    st.divider()

    st.markdown("### 📂 Datos de Entrada")
    upload_mode = st.radio(
        "Fuente de datos",
        ["Archivo TS5", "Ejemplo TS5 (por defecto)"],
        index=1,
    )

    ts5_content: str | None = None

    if upload_mode == "Archivo TS5":
        uploaded = st.file_uploader("Sube tu archivo TS5 (.txt)", type=["txt"])
        if uploaded:
            ts5_content = uploaded.read().decode("utf-8")
            st.success("✅ Archivo cargado")
    else:
        ts5_content = (
            "i\t5\t\to\t3\t\tn\t8\t\t"
            "r\t1\t1\t170\t"
            "r\t2\t1\t6\tr\t2\t2\t6\tr\t2\t3\t19\tr\t2\t4\t50\t"
            "r\t2\t5\t38\tr\t2\t6\t6\tr\t2\t7\t19\tr\t2\t8\t56\t"
            "r\t3\t1\t49\tr\t3\t2\t31\tr\t3\t3\t60\tr\t3\t6\t12\t"
            "r\t3\t7\t37\tr\t3\t8\t31\t"
            "r\t4\t5\t143\tr\t4\t7\t47\t"
            "r\t5\t4\t58\tr\t5\t5\t36\tr\t5\t7\t72\tr\t5\t8\t14\t"
            "s\t1\t1\t75\ts\t1\t2\t12\ts\t1\t3\t59\ts\t1\t6\t9\t"
            "s\t1\t7\t98\ts\t1\t8\t40\t"
            "s\t2\t1\t150\ts\t2\t5\t217\t"
            "s\t3\t2\t25\ts\t3\t3\t20\ts\t3\t4\t108\ts\t3\t6\t9\t"
            "s\t3\t7\t77\ts\t3\t8\t61"
        )
        st.info("Usando datos del caso TS5")

    st.divider()
    st.markdown("### ⚙️ Parámetros del Solver")
    time_limit = st.slider("Tiempo límite (seg)", 30, 300, 120, 10)
    mip_gap    = st.slider("Gap MIP (%)", 0.5, 10.0, 1.0, 0.5) / 100.0

    st.divider()
    st.markdown("### 📐 Parámetros Operativos")
    st.markdown(f"""
    | Parámetro | Valor |
    |-----------|-------|
    | Carga/descarga | {T_UNIT} min/ud |
    | Traslado interno | {T_TRANSFER} min/lote |
    | Cambio camión | {T_CHANGEOVER} min |
    """)

    solve_btn = st.button("🔍 Resolver Optimización", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Encabezado principal
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">Cross-Docking Optimizer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">LogiFast CR · Programación Entera Mixta (MIP) · '
    'Minimización de Makespan</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Vista previa de datos
# ─────────────────────────────────────────────────────────────────────────────
if ts5_content:
    try:
        inst: InstanceData = parse_ts5(ts5_content)
    except Exception as e:
        st.error(f"Error al parsear el archivo: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📊 Datos", "🧮 Modelo MIP", "🚀 Resultados"])

    # ── Tab 1: Datos ──────────────────────────────────────────────────────────
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Camiones Entrada (i)", inst.n_in)
        col2.metric("Camiones Salida (o)", inst.n_out)
        col3.metric("Tipos de Producto (n)", inst.n_prod)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<p class="section-header">Camiones de Entrada</p>',
                        unsafe_allow_html=True)
            rows_r = []
            for i in inst.inbound_trucks:
                for k in inst.products:
                    qty = inst.r.get((i, k), 0)
                    if qty > 0:
                        rows_r.append({"Camión": f"R{i}", "Producto": f"P{k}", "Cantidad": qty})
            df_r = pd.DataFrame(rows_r)
            if not df_r.empty:
                pivot_r = df_r.pivot_table(
                    index="Camión", columns="Producto",
                    values="Cantidad", fill_value=0, aggfunc="sum"
                )
                st.dataframe(pivot_r, use_container_width=True)
                # Totales
                totales_in = {i: inst.unload_time(i) for i in inst.inbound_trucks}
                df_tot_in = pd.DataFrame([
                    {"Camión": f"R{i}", "Total Unidades": int(t / T_UNIT),
                     "Tiempo Descarga (min)": t}
                    for i, t in totales_in.items()
                ])
                st.dataframe(df_tot_in, use_container_width=True, hide_index=True)

        with c2:
            st.markdown('<p class="section-header">Camiones de Salida</p>',
                        unsafe_allow_html=True)
            rows_s = []
            for j in inst.outbound_trucks:
                for k in inst.products:
                    qty = inst.s.get((j, k), 0)
                    if qty > 0:
                        rows_s.append({"Camión": f"S{j}", "Producto": f"P{k}", "Cantidad": qty})
            df_s = pd.DataFrame(rows_s)
            if not df_s.empty:
                pivot_s = df_s.pivot_table(
                    index="Camión", columns="Producto",
                    values="Cantidad", fill_value=0, aggfunc="sum"
                )
                st.dataframe(pivot_s, use_container_width=True)
                totales_out = {j: inst.load_time(j) for j in inst.outbound_trucks}
                df_tot_out = pd.DataFrame([
                    {"Camión": f"S{j}", "Total Unidades": int(t / T_UNIT),
                     "Tiempo Carga (min)": t}
                    for j, t in totales_out.items()
                ])
                st.dataframe(df_tot_out, use_container_width=True, hide_index=True)

        # Heatmap demanda vs oferta
        st.markdown('<p class="section-header">Mapa de Producto: Oferta vs Demanda</p>',
                    unsafe_allow_html=True)
        supply = {k: sum(inst.r.get((i, k), 0) for i in inst.inbound_trucks)
                  for k in inst.products}
        demand = {k: sum(inst.s.get((j, k), 0) for j in inst.outbound_trucks)
                  for k in inst.products}
        df_bal = pd.DataFrame({
            "Producto": [f"P{k}" for k in inst.products],
            "Oferta (entrada)": [supply[k] for k in inst.products],
            "Demanda (salida)": [demand[k] for k in inst.products],
        })
        fig_bar = go.Figure()
        fig_bar.add_bar(x=df_bal["Producto"], y=df_bal["Oferta (entrada)"],
                        name="Oferta", marker_color="#FF6B2B")
        fig_bar.add_bar(x=df_bal["Producto"], y=df_bal["Demanda (salida)"],
                        name="Demanda", marker_color="#1E90FF")
        fig_bar.update_layout(
            barmode="group", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20), legend_orientation="h",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tab 2: Descripción del modelo ─────────────────────────────────────────
    with tab2:
        st.markdown("""
### Formulación del Problema

**Objetivo:** Minimizar el *makespan* — el tiempo en que el último camión de salida
termina de cargarse y puede abandonar el muelle.

---
#### Variables de Decisión

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `MS` | Continua ≥ 0 | Makespan (tiempo total) |
| `T_in[i]` | Continua ≥ 0 | Tiempo de inicio de descarga del camión de entrada *i* |
| `T_out[j]` | Continua ≥ 0 | Tiempo en que el camión de salida *j* termina de cargar |
| `α[i₁,i₂]` | Binaria | 1 si camión entrada *i₁* va **antes** que *i₂* |
| `β[j₁,j₂]` | Binaria | 1 si camión salida *j₁* va **antes** que *j₂* |
| `x[i,j,k]` | Entera ≥ 0 | Unidades del producto *k* del camión entrada *i* al salida *j* |
| `z[i,j,k]` | Binaria | 1 = transferencia directa; 0 = pasa por almacenamiento temporal |

---
#### Función Objetivo
```
min  MS
```

---
#### Restricciones

| # | Tipo | Descripción |
|---|------|-------------|
| 1 | `MS ≥ T_out[j]` ∀j | Makespan mayorado por fin de cada camión salida |
| 2 | `Σⱼ x[i,j,k] = r[i,k]` ∀i,k | Conservación de inventario en camiones de **entrada** |
| 3 | `Σᵢ x[i,j,k] = s[j,k]` ∀j,k | Conservación de inventario en camiones de **salida** |
| 4 | `x[i,j,k] ≤ M·z[i,j,k]` | Relación flujo–transferencia directa |
| 5–7 | Secuencia lineal entrada | `T_in[i₂] ≥ T_in[i₁] + descarga(i₁) + cambio` si α[i₁,i₂]=1 |
| 6 | Antisimetría α | `α[i₁,i₂] + α[i₂,i₁] = 1` |
| 8 | No auto-precedencia entrada | Implícita (α no definida para i₁=i₂) |
| 9–11 | Secuencia lineal salida | `T_out[j₂] ≥ T_out[j₁] + cambio` si β[j₁,j₂]=1 |
| 10 | Antisimetría β | `β[j₁,j₂] + β[j₂,j₁] = 1` |
| 12 | No auto-precedencia salida | Implícita |
| 13 | Enlace entrada–salida | `T_out[j] ≥ T_in[i] + descarga(i) + traslado + carga(j)` si z[i,j,k]=1 |

---
#### Tamaño del Modelo para TS5
""")
        n_i, n_j, n_k = inst.n_in, inst.n_out, inst.n_prod
        n_cont   = 1 + n_i + n_j
        n_bin    = n_i*(n_i-1) + n_j*(n_j-1) + n_i*n_j*n_k
        n_int    = n_i * n_j * n_k
        n_const  = (n_j                  # C1
                  + n_i*n_k              # C2
                  + n_j*n_k              # C3
                  + n_i*n_j*n_k         # C4
                  + n_i*(n_i-1)         # C5
                  + n_i*(n_i-1)//2      # C6
                  + n_j*(n_j-1)         # C9
                  + n_j*(n_j-1)//2      # C10
                  + n_i*n_j*n_k)        # C13

        cols = st.columns(4)
        cols[0].metric("Variables continuas", n_cont)
        cols[1].metric("Variables binarias", n_bin)
        cols[2].metric("Variables enteras", n_int)
        cols[3].metric("Restricciones", n_const)

    # ── Tab 3: Resultados ─────────────────────────────────────────────────────
    with tab3:
        if not solve_btn:
            st.info("👈 Presiona **Resolver Optimización** en el panel lateral para iniciar.")
        else:
            with st.spinner("🔄 Resolviendo modelo MIP con CBC..."):
                t0 = time.time()
                result: SolverResult = build_and_solve(
                    inst, time_limit=time_limit, mip_gap=mip_gap
                )
                elapsed = time.time() - t0

            if result.makespan is None:
                st.error(f"❌ El solver no encontró solución. Estado: {result.status}")
                st.text(result.log)
            else:
                # ── KPIs ──────────────────────────────────────────────────
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{result.makespan:.0f}</div>
                  <div class="metric-label">Makespan (min)</div>
                </div>""", unsafe_allow_html=True)
                c2.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{result.makespan/60:.1f}</div>
                  <div class="metric-label">Makespan (horas)</div>
                </div>""", unsafe_allow_html=True)
                c3.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{result.status}</div>
                  <div class="metric-label">Estado</div>
                </div>""", unsafe_allow_html=True)
                c4.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{elapsed:.1f}s</div>
                  <div class="metric-label">Tiempo CPU</div>
                </div>""", unsafe_allow_html=True)

                # ── Orden de camiones ─────────────────────────────────────
                st.markdown('<p class="section-header">Orden Óptimo de Camiones</p>',
                            unsafe_allow_html=True)
                col_in, col_out = st.columns(2)

                with col_in:
                    st.markdown("**Camiones de Entrada** (orden de atención en muelle)")
                    badges_in = " ".join(
                        f'<span class="order-badge">#{pos+1} → R{i}</span>'
                        for pos, i in enumerate(result.order_in)
                    )
                    st.markdown(badges_in, unsafe_allow_html=True)

                    df_tin = pd.DataFrame([
                        {
                            "Camión": f"R{i}",
                            "Inicio Descarga (min)": round(result.T_in[i], 1),
                            "Fin Descarga (min)": round(result.T_in[i] + inst.unload_time(i), 1),
                            "Duración (min)": inst.unload_time(i),
                        }
                        for i in result.order_in
                    ])
                    st.dataframe(df_tin, use_container_width=True, hide_index=True)

                with col_out:
                    st.markdown("**Camiones de Salida** (orden de atención en muelle)")
                    badges_out = " ".join(
                        f'<span class="order-badge">#{pos+1} → S{j}</span>'
                        for pos, j in enumerate(result.order_out)
                    )
                    st.markdown(badges_out, unsafe_allow_html=True)

                    df_tout = pd.DataFrame([
                        {
                            "Camión": f"S{j}",
                            "Inicio Carga (min)": round(result.T_out[j] - inst.load_time(j), 1),
                            "Fin Carga (min)": round(result.T_out[j], 1),
                            "Duración (min)": inst.load_time(j),
                        }
                        for j in result.order_out
                    ])
                    st.dataframe(df_tout, use_container_width=True, hide_index=True)

                # ── Diagrama de Gantt ─────────────────────────────────────
                st.markdown('<p class="section-header">Diagrama de Gantt</p>',
                            unsafe_allow_html=True)

                gantt_rows = []
                for i in result.order_in:
                    start = result.T_in[i]
                    end   = start + inst.unload_time(i)
                    gantt_rows.append(dict(
                        Task=f"R{i} (Entrada)",
                        Start=start, Finish=end,
                        Resource="Entrada"
                    ))
                for j in result.order_out:
                    end   = result.T_out[j]
                    start = end - inst.load_time(j)
                    gantt_rows.append(dict(
                        Task=f"S{j} (Salida)",
                        Start=start, Finish=end,
                        Resource="Salida"
                    ))

                df_gantt = pd.DataFrame(gantt_rows)
                # Convertir a datetime base para plotly (usa t=0 como epoch)
                from datetime import datetime, timedelta
                base = datetime(2024, 1, 1, 0, 0, 0)
                df_gantt["Start_dt"] = df_gantt["Start"].apply(
                    lambda x: base + timedelta(minutes=x)
                )
                df_gantt["Finish_dt"] = df_gantt["Finish"].apply(
                    lambda x: base + timedelta(minutes=x)
                )

                colors = {"Entrada": "#FF6B2B", "Salida": "#1E90FF"}
                fig_gantt = ff.create_gantt(
                    [{"Task": r["Task"],
                      "Start": r["Start_dt"].strftime("%Y-%m-%d %H:%M:%S"),
                      "Finish": r["Finish_dt"].strftime("%Y-%m-%d %H:%M:%S"),
                      "Resource": r["Resource"]}
                     for _, r in df_gantt.iterrows()],
                    colors=colors,
                    index_col="Resource",
                    show_colorbar=True,
                    group_tasks=True,
                    title="",
                )
                fig_gantt.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Minutos desde inicio",
                )
                # Reemplazar labels del eje X por minutos reales
                tick_vals = list(range(0, int(result.makespan) + 30, 30))
                tick_dates = [(base + timedelta(minutes=v)).strftime("%Y-%m-%d %H:%M:%S")
                              for v in tick_vals]
                fig_gantt.update_xaxes(
                    tickvals=tick_dates,
                    ticktext=[str(v) for v in tick_vals],
                )
                st.plotly_chart(fig_gantt, use_container_width=True)

                # ── Flujos de productos ───────────────────────────────────
                st.markdown('<p class="section-header">Flujos de Producto: Entrada → Salida</p>',
                            unsafe_allow_html=True)

                flow_rows = []
                for i in inst.inbound_trucks:
                    for j in inst.outbound_trucks:
                        for k in inst.products:
                            qty = result.x.get((i, j, k), 0)
                            if qty and qty > 0.5:
                                is_direct = result.z.get((i, j, k), 0) == 1
                                flow_rows.append({
                                    "De": f"R{i}",
                                    "A": f"S{j}",
                                    "Producto": f"P{k}",
                                    "Cantidad": int(round(qty)),
                                    "Ruta": "🟢 Directo" if is_direct else "🟡 Almacén Temp.",
                                })

                if flow_rows:
                    df_flows = pd.DataFrame(flow_rows)
                    st.dataframe(
                        df_flows.sort_values(["De", "A", "Producto"]),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Resumen directo vs temporal
                    n_direct = df_flows["Ruta"].str.startswith("🟢").sum()
                    n_temp   = df_flows["Ruta"].str.startswith("🟡").sum()
                    total_direct_units = df_flows[df_flows["Ruta"].str.startswith("🟢")]["Cantidad"].sum()
                    total_temp_units   = df_flows[df_flows["Ruta"].str.startswith("🟡")]["Cantidad"].sum()

                    cx, cy = st.columns(2)
                    cx.metric("Transferencias Directas", f"{n_direct} lotes ({total_direct_units} uds)")
                    cy.metric("Vía Almacén Temporal", f"{n_temp} lotes ({total_temp_units} uds)")

                    # Sankey de flujos
                    st.markdown('<p class="section-header">Diagrama de Flujo (Sankey)</p>',
                                unsafe_allow_html=True)
                    # Nodos: camiones entrada + camiones salida
                    in_labels  = [f"R{i}" for i in inst.inbound_trucks]
                    out_labels = [f"S{j}" for j in inst.outbound_trucks]
                    all_labels = in_labels + out_labels
                    label_idx  = {lbl: idx for idx, lbl in enumerate(all_labels)}

                    # Agregar flujos por par (R_i, S_j)
                    pair_flow: dict[tuple, int] = {}
                    for row in flow_rows:
                        key = (row["De"], row["A"])
                        pair_flow[key] = pair_flow.get(key, 0) + row["Cantidad"]

                    s_src = [label_idx[k[0]] for k in pair_flow]
                    s_tgt = [label_idx[k[1]] for k in pair_flow]
                    s_val = list(pair_flow.values())

                    node_colors = (
                        ["#FF6B2B"] * len(in_labels) +
                        ["#1E90FF"] * len(out_labels)
                    )
                    fig_sankey = go.Figure(go.Sankey(
                        node=dict(
                            pad=15, thickness=20, line=dict(color="black", width=0.5),
                            label=all_labels, color=node_colors,
                        ),
                        link=dict(
                            source=s_src, target=s_tgt, value=s_val,
                            color=["rgba(255,107,43,0.3)"] * len(s_src)
                        ),
                    ))
                    fig_sankey.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=20, r=20, t=20, b=20),
                    )
                    st.plotly_chart(fig_sankey, use_container_width=True)

                # ── Log del solver ─────────────────────────────────────────
                with st.expander("🔎 Log del Solver"):
                    st.markdown(
                        f'<div class="info-box">{result.log}</div>',
                        unsafe_allow_html=True,
                    )

else:
    st.warning("Selecciona o sube un archivo TS5 en el panel lateral para comenzar.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#555; font-size:0.78rem; font-family: JetBrains Mono'>"
    "LogiFast CR · Optimización Cross-Docking · UCR Ingeniería Industrial · I Semestre 2026"
    "</center>",
    unsafe_allow_html=True,
)

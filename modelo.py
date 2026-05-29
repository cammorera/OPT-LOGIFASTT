"""
modelo.py — Modelo MIP para Cross-Docking (LogiFast CR)
=========================================================
Minimiza el makespan: tiempo en que el último camión de salida
termina de cargar y puede abandonar el muelle.

Variables de decisión
---------------------
Continuas:
  T_in[i]   : tiempo de llegada al muelle del camión de entrada i  (>= 0)
  T_out[j]  : tiempo en que el camión de salida j termina de cargar (>= 0)
  MS        : makespan (tiempo total de operación)

Enteras binarias:
  alpha[i1,i2] : 1 si camión de entrada i1 va ANTES que i2 en el muelle
  beta[j1,j2]  : 1 si camión de salida  j1 va ANTES que j2 en el muelle
  z[i,j,k]     : 1 si producto k va DIRECTO del camión i al camión j
                 (0 => pasa por almacenamiento temporal)

No binarias enteras:
  x[i,j,k] : unidades del producto k que van del camión de entrada i
              al camión de salida j  (entero >= 0)

Restricciones
-------------
 1. Makespan >= T_out[j] para todo j
 2. Conservación en camiones de entrada: sum_j x[i,j,k] = r[i,k]
 3. Conservación en camiones de salida:  sum_i x[i,j,k] = s[j,k]
 4. Relación x-z: x[i,j,k] <= M * z[i,j,k]
 5-7. Secuencia válida camiones de entrada (orden lineal)
 8. Ningún camión de entrada precede a sí mismo
 9-11. Secuencia válida camiones de salida
12. Ningún camión de salida precede a sí mismo
13. T_out[j] >= T_in[i] + unload_i + transfer + load_j  si z[i,j,k]=1
    (el camión de salida no puede terminar antes de que los productos
     del camión de entrada i estén disponibles)

Parámetros operativos
---------------------
  - 1 min  por unidad de carga/descarga
  - 5 min  de traslado interno por lote
  - 10 min de cambio entre camiones en el muelle
"""

from __future__ import annotations
import io
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pulp
import numpy as np


# ---------------------------------------------------------------------------
# Parámetros operativos (constantes del problema)
# ---------------------------------------------------------------------------
T_UNIT      = 1    # minutos por unidad cargada/descargada
T_TRANSFER  = 5    # minutos de traslado interno por lote
T_CHANGEOVER= 10   # minutos de cambio entre camiones en el muelle
BIG_M       = 1e6  # constante Big-M


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------
@dataclass
class InstanceData:
    """Datos leídos del archivo TS5."""
    n_in:   int                           # número de camiones de entrada
    n_out:  int                           # número de camiones de salida
    n_prod: int                           # número de productos
    r: Dict[Tuple[int,int], int] = field(default_factory=dict)  # r[i,k]
    s: Dict[Tuple[int,int], int] = field(default_factory=dict)  # s[j,k]

    @property
    def inbound_trucks(self) -> List[int]:
        return list(range(1, self.n_in + 1))

    @property
    def outbound_trucks(self) -> List[int]:
        return list(range(1, self.n_out + 1))

    @property
    def products(self) -> List[int]:
        return list(range(1, self.n_prod + 1))

    def unload_time(self, i: int) -> float:
        """Tiempo total de descarga del camión de entrada i."""
        total = sum(self.r.get((i, k), 0) for k in self.products)
        return total * T_UNIT

    def load_time(self, j: int) -> float:
        """Tiempo total de carga del camión de salida j."""
        total = sum(self.s.get((j, k), 0) for k in self.products)
        return total * T_UNIT


# ---------------------------------------------------------------------------
# Parser del archivo TS5
# ---------------------------------------------------------------------------
def parse_ts5(content: str) -> InstanceData:
    """
    Parsea el contenido del archivo TS5 (texto) y devuelve un InstanceData.
    El formato puede tener los tokens separados por espacios o tabuladores,
    todo en una sola línea o en múltiples líneas.
    """
    tokens = content.split()
    it = iter(tokens)

    n_in = n_out = n_prod = None
    r: Dict[Tuple[int,int], int] = {}
    s: Dict[Tuple[int,int], int] = {}

    while True:
        try:
            tok = next(it)
        except StopIteration:
            break

        if tok == 'i':
            n_in = int(next(it))
        elif tok == 'o':
            n_out = int(next(it))
        elif tok == 'n':
            n_prod = int(next(it))
        elif tok == 'r':
            truck = int(next(it))
            prod  = int(next(it))
            qty   = int(next(it))
            r[(truck, prod)] = r.get((truck, prod), 0) + qty
        elif tok == 's':
            truck = int(next(it))
            prod  = int(next(it))
            qty   = int(next(it))
            s[(truck, prod)] = s.get((truck, prod), 0) + qty

    if None in (n_in, n_out, n_prod):
        raise ValueError("El archivo TS5 no contiene i, o, o n válidos.")

    return InstanceData(n_in=n_in, n_out=n_out, n_prod=n_prod, r=r, s=s)


# ---------------------------------------------------------------------------
# Construcción y resolución del modelo MIP
# ---------------------------------------------------------------------------
@dataclass
class SolverResult:
    status:        str
    makespan:      float | None
    T_in:          Dict[int, float]
    T_out:         Dict[int, float]
    order_in:      List[int]   # orden de camiones de entrada
    order_out:     List[int]   # orden de camiones de salida
    x:             Dict[Tuple[int,int,int], float]   # flujos
    z:             Dict[Tuple[int,int,int], int]      # directo vs temp
    log:           str


def build_and_solve(data: InstanceData,
                    time_limit: int = 120,
                    mip_gap: float = 0.01) -> SolverResult:
    """
    Construye el modelo MIP con PuLP y lo resuelve con CBC.

    Devuelve un SolverResult con todos los valores de las variables
    y metadatos del solver.
    """
    I  = data.inbound_trucks
    J  = data.outbound_trucks
    K  = data.products
    r  = data.r
    s  = data.s

    model = pulp.LpProblem("CrossDocking_LogiFast", pulp.LpMinimize)

    # ------------------------------------------------------------------
    # Variables de decisión
    # ------------------------------------------------------------------

    # Makespan (continua, >= 0)
    MS = pulp.LpVariable("Makespan", lowBound=0)

    # Tiempos de llegada/inicio de descarga de camiones de entrada (continua)
    T_in  = {i: pulp.LpVariable(f"T_in_{i}",  lowBound=0) for i in I}

    # Tiempos en que el camión de salida j termina de cargarse (continua)
    T_out = {j: pulp.LpVariable(f"T_out_{j}", lowBound=0) for j in J}

    # Orden entre camiones de entrada: alpha[i1,i2]=1 => i1 antes de i2
    alpha = {
        (i1, i2): pulp.LpVariable(f"alpha_{i1}_{i2}", cat="Binary")
        for i1 in I for i2 in I if i1 != i2
    }

    # Orden entre camiones de salida: beta[j1,j2]=1 => j1 antes de j2
    beta = {
        (j1, j2): pulp.LpVariable(f"beta_{j1}_{j2}", cat="Binary")
        for j1 in J for j2 in J if j1 != j2
    }

    # Unidades de producto k que van de camión entrada i a camión salida j
    x = {
        (i, j, k): pulp.LpVariable(f"x_{i}_{j}_{k}", lowBound=0, cat="Integer")
        for i in I for j in J for k in K
    }

    # z[i,j,k]=1 => transferencia directa (sin almacenamiento temporal)
    z = {
        (i, j, k): pulp.LpVariable(f"z_{i}_{j}_{k}", cat="Binary")
        for i in I for j in J for k in K
    }

    # ------------------------------------------------------------------
    # Función objetivo: minimizar makespan
    # ------------------------------------------------------------------
    model += MS, "Minimizar_Makespan"

    # ------------------------------------------------------------------
    # Restricción 1: Makespan >= T_out[j] para todo j
    # ------------------------------------------------------------------
    for j in J:
        model += MS >= T_out[j], f"C1_makespan_j{j}"

    # ------------------------------------------------------------------
    # Restricción 2: Conservación en camiones de ENTRADA
    #   sum_j x[i,j,k] = r[i,k]   para todo i, k
    # ------------------------------------------------------------------
    for i in I:
        for k in K:
            rhs = r.get((i, k), 0)
            model += (
                pulp.lpSum(x[i, j, k] for j in J) == rhs,
                f"C2_in_i{i}_k{k}"
            )

    # ------------------------------------------------------------------
    # Restricción 3: Conservación en camiones de SALIDA
    #   sum_i x[i,j,k] = s[j,k]   para todo j, k
    # ------------------------------------------------------------------
    for j in J:
        for k in K:
            rhs = s.get((j, k), 0)
            model += (
                pulp.lpSum(x[i, j, k] for i in I) == rhs,
                f"C3_out_j{j}_k{k}"
            )

    # ------------------------------------------------------------------
    # Restricción 4: Relación x-z
    #   x[i,j,k] <= M * z[i,j,k]
    # ------------------------------------------------------------------
    for i in I:
        for j in J:
            for k in K:
                ub = min(
                    sum(r.get((i, kk), 0) for kk in K),
                    sum(s.get((j, kk), 0) for kk in K),
                    r.get((i, k), 0) + s.get((j, k), 0)
                )
                model += (
                    x[i, j, k] <= ub * z[i, j, k],
                    f"C4_xz_i{i}_j{j}_k{k}"
                )

    # ------------------------------------------------------------------
    # Restricciones 5-7: Secuencia válida camiones de ENTRADA
    # Si alpha[i1,i2]=1 => i1 va antes que i2
    # T_in[i2] >= T_in[i1] + unload(i1) + T_CHANGEOVER  si alpha[i1,i2]=1
    # Usando Big-M:
    #   T_in[i2] >= T_in[i1] + unload(i1) + T_CHANGEOVER - M*(1-alpha[i1,i2])
    # ------------------------------------------------------------------
    for i1 in I:
        for i2 in I:
            if i1 == i2:
                continue
            dur_i1 = data.unload_time(i1)
            model += (
                T_in[i2] >= T_in[i1] + dur_i1 + T_CHANGEOVER
                           - BIG_M * (1 - alpha[i1, i2]),
                f"C5_seq_in_{i1}_{i2}"
            )

    # ------------------------------------------------------------------
    # Restricción 6-7: alpha forma un orden total (antisimetría)
    #   alpha[i1,i2] + alpha[i2,i1] = 1   para i1 != i2
    # ------------------------------------------------------------------
    for i1 in I:
        for i2 in I:
            if i1 < i2:
                model += (
                    alpha[i1, i2] + alpha[i2, i1] == 1,
                    f"C6_antisym_in_{i1}_{i2}"
                )

    # ------------------------------------------------------------------
    # Restricción 8: Ningún camión de entrada precede a sí mismo
    #   (garantizado por var no definida para i1=i2, implícito)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Restricciones 9-11: Secuencia válida camiones de SALIDA
    # ------------------------------------------------------------------
    for j1 in J:
        for j2 in J:
            if j1 == j2:
                continue
            dur_j1 = data.load_time(j1)
            # T_out[j2] >= T_out[j1] + T_CHANGEOVER si beta[j1,j2]=1
            # pero T_out es fin de carga; inicio de carga de j2 >= fin de j1 + changeover
            # Simplificamos: T_out[j2] >= T_out[j1] + T_CHANGEOVER - M*(1-beta[j1,j2])
            model += (
                T_out[j2] >= T_out[j1] + T_CHANGEOVER
                            - BIG_M * (1 - beta[j1, j2]),
                f"C9_seq_out_{j1}_{j2}"
            )

    # ------------------------------------------------------------------
    # Restricción 10-11: beta forma un orden total
    # ------------------------------------------------------------------
    for j1 in J:
        for j2 in J:
            if j1 < j2:
                model += (
                    beta[j1, j2] + beta[j2, j1] == 1,
                    f"C10_antisym_out_{j1}_{j2}"
                )

    # ------------------------------------------------------------------
    # Restricción 12: No outbound truck precedes itself — implícito
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Restricción 13: Conectar T_out[j] con T_in[i] cuando hay transferencia
    # Si z[i,j,k]=1:
    #   T_out[j] >= T_in[i] + unload(i) + T_TRANSFER + load(j)
    # Con Big-M:
    #   T_out[j] >= T_in[i] + unload(i) + T_TRANSFER + load(j)
    #              - M*(1 - z[i,j,k])
    # Usamos una restricción por par (i,j) en lugar de por (i,j,k)
    # para evitar explosión de restricciones redundantes.
    # ------------------------------------------------------------------
    for i in I:
        for j in J:
            for k in K:
                if r.get((i, k), 0) == 0 or s.get((j, k), 0) == 0:
                    # No hay flujo posible → z=0 forzado
                    model += z[i, j, k] == 0, f"C4b_noprod_i{i}_j{j}_k{k}"
                    continue
                dur_i = data.unload_time(i)
                dur_j = data.load_time(j)
                model += (
                    T_out[j] >= T_in[i] + dur_i + T_TRANSFER + dur_j
                               - BIG_M * (1 - z[i, j, k]),
                    f"C13_connect_i{i}_j{j}_k{k}"
                )

    # ------------------------------------------------------------------
    # Resolver
    # ------------------------------------------------------------------
    log_stream = io.StringIO()
    solver = pulp.PULP_CBC_CMD(
        timeLimit=time_limit,
        gapRel=mip_gap,
        msg=True,
        logPath=None
    )
    status_code = model.solve(solver)
    status_str  = pulp.LpStatus[model.status]

    if model.status not in (1, -2):  # 1=Optimal, -2=Not Solved (feasible)
        return SolverResult(
            status=status_str,
            makespan=None,
            T_in={}, T_out={}, order_in=[], order_out=[],
            x={}, z={},
            log=f"Solver terminó con estado: {status_str}"
        )

    # ------------------------------------------------------------------
    # Extraer resultados
    # ------------------------------------------------------------------
    ms_val = pulp.value(MS)

    tin_val  = {i: pulp.value(T_in[i])  for i in I}
    tout_val = {j: pulp.value(T_out[j]) for j in J}

    # Orden de camiones de entrada según T_in
    order_in  = sorted(I,  key=lambda i: tin_val[i])
    # Orden de camiones de salida según T_out
    order_out = sorted(J,  key=lambda j: tout_val[j])

    x_val = {
        (i, j, k): pulp.value(x[i, j, k]) or 0.0
        for i in I for j in J for k in K
    }
    z_val = {
        (i, j, k): int(round(pulp.value(z[i, j, k]) or 0))
        for i in I for j in J for k in K
    }

    log_txt = (
        f"Estado solver : {status_str}\n"
        f"Makespan      : {ms_val:.1f} minutos\n"
        f"Orden entrada : {order_in}\n"
        f"Orden salida  : {order_out}\n"
    )

    return SolverResult(
        status=status_str,
        makespan=ms_val,
        T_in=tin_val,
        T_out=tout_val,
        order_in=order_in,
        order_out=order_out,
        x=x_val,
        z=z_val,
        log=log_txt,
    )


# ---------------------------------------------------------------------------
# Ejecución directa (prueba con TS5 hardcoded)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ts5_sample = """
i 5 o 3 n 8
r 1 1 170
r 2 1 6  r 2 2 6  r 2 3 19 r 2 4 50 r 2 5 38 r 2 6 6  r 2 7 19 r 2 8 56
r 3 1 49 r 3 2 31 r 3 3 60 r 3 6 12 r 3 7 37 r 3 8 31
r 4 5 143 r 4 7 47
r 5 4 58  r 5 5 36 r 5 7 72 r 5 8 14
s 1 1 75  s 1 2 12 s 1 3 59 s 1 6 9  s 1 7 98 s 1 8 40
s 2 1 150 s 2 5 217
s 3 2 25  s 3 3 20 s 3 4 108 s 3 6 9 s 3 7 77 s 3 8 61
"""
    inst = parse_ts5(ts5_sample)
    print(f"Instancia: {inst.n_in} camiones entrada, "
          f"{inst.n_out} camiones salida, {inst.n_prod} productos")

    res = build_and_solve(inst, time_limit=120)
    print(res.log)

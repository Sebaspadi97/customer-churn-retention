"""
¿Qué clientes están por irse? · SP Lab · Paso 1: datos SINTÉTICOS
Genera 20.000 clientes de "TiendaLab", una tienda en línea FICTICIA (mercado, hogar y cuidado personal) en Colombia.
Ninguna fila corresponde a una persona o empresa real. No incluye datos de pagos ni medios de pago.

Foto del cliente: 30 jun 2026. Etiqueta: abandono = no compra entre el 1 jul y el 30 sep 2026 (90 días).
Las reglas que generan el abandono están en `logit` (abajo): el modelo debe "descubrirlas" sin verlas.
Uso: python 01_generar_datos.py <archivo_salida.csv>
"""
import sys
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 20_000

ciudades = ["Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga", "Otras ciudades"]
p_ciudad = [0.38, 0.17, 0.12, 0.09, 0.06, 0.18]
retraso_base = {"Bogotá": 0.07, "Medellín": 0.08, "Cali": 0.10, "Barranquilla": 0.12, "Bucaramanga": 0.10, "Otras ciudades": 0.18}

ciudad = rng.choice(ciudades, N, p=p_ciudad)
antiguedad = np.clip(rng.gamma(2.0, 9.0, N), 1, 72).round().astype(int)                  # meses
canal_app = rng.random(N) < np.clip(0.35 + 0.004 * antiguedad, 0, 0.75)                   # app vs web
lealtad = rng.random(N) < np.clip(0.10 + 0.006 * antiguedad + 0.10 * canal_app, 0, 0.6)  # programa de puntos

# Frecuencia: más alta con lealtad y app
lam = np.exp(1.2 + 0.35 * lealtad + 0.25 * canal_app + 0.004 * antiguedad + rng.normal(0, 0.55, N))
pedidos_12m = np.clip(rng.poisson(lam), 1, 60)
# Recencia: inversa a la frecuencia
dias_ultima = np.clip(rng.exponential(365 / (pedidos_12m + 1.5), N) + rng.integers(0, 6, N), 0, 365).round().astype(int)
ticket = np.clip(rng.lognormal(np.log(85_000), 0.45, N), 15_000, 900_000).round(-2)       # COP
categorias = np.clip(1 + rng.binomial(7, np.clip(0.08 + 0.02 * pedidos_12m, 0, 0.85)), 1, 8)
pct_descuento = rng.beta(2, 5, N)                                                          # % de pedidos con cupón
pct_tarde = np.clip(rng.beta(1.5, 12, N) + np.vectorize(retraso_base.get)(ciudad) - 0.08, 0, 1)
devoluciones = rng.poisson(0.06 * pedidos_12m + 0.1 * pct_tarde * 10)
tickets_soporte = rng.poisson(0.25 + 2.2 * pct_tarde + 0.15 * devoluciones)
calif = np.clip(np.round(4.6 - 4.5 * pct_tarde - 0.25 * tickets_soporte + rng.normal(0, 0.6, N)), 1, 5).astype(int)
apertura_correos = rng.beta(2, 4, N) * (0.6 + 0.4 * lealtad)

# ---------- Regla oculta de abandono (no la ve el modelo) ----------
logit = (
    -2.35
    + 0.012 * dias_ultima                                   # recencia
    - 0.55 * np.log1p(pedidos_12m)                          # frecuencia
    + 3.2 * pct_tarde                                       # entregas tarde
    + 0.30 * tickets_soporte                                # fricción con soporte
    + 0.9 * (calif <= 2)                                    # mala experiencia reciente
    - 0.55 * lealtad
    - 0.30 * canal_app
    - 0.12 * (categorias - 1)                               # cliente que compra de todo
    + 1.4 * pct_descuento * (ticket < 70_000)               # cazadores de descuento de ticket bajo
    + 0.75 * (antiguedad <= 3)                              # clientes nuevos
    - 0.9 * apertura_correos
    + rng.normal(0, 0.6, N)                                 # lo que no se observa
)
abandono = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

df = pd.DataFrame({
    "id_cliente": [f"TL{i:05d}" for i in range(1, N + 1)],
    "ciudad": ciudad,
    "antiguedad_meses": antiguedad,
    "canal_principal": np.where(canal_app, "App", "Web"),
    "programa_lealtad": lealtad.astype(int),
    "pedidos_12m": pedidos_12m,
    "dias_desde_ultima_compra": dias_ultima,
    "ticket_promedio_cop": ticket.astype(int),
    "categorias_distintas": categorias,
    "pct_pedidos_con_descuento": pct_descuento.round(3),
    "pct_entregas_tarde": pct_tarde.round(3),
    "devoluciones_12m": devoluciones,
    "tickets_soporte_90d": tickets_soporte,
    "calificacion_ultima_compra": calif,
    "pct_apertura_correos": apertura_correos.round(3),
    "abandono_90d": abandono,
})
df.to_csv(sys.argv[1], index=False, encoding="utf-8-sig")
print(df.shape, "tasa de abandono:", round(df.abandono_90d.mean() * 100, 1), "%")
print(df.describe().T.round(2).to_string())

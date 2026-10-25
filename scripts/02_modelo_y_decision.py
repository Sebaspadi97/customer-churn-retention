"""
¿Qué clientes están por irse? · SP Lab · Paso 2: modelo + decisión de retención
Uso: python 02_modelo_y_decision.py <clientes.csv> <carpeta_salida>
Supuestos de negocio (editables, todos ficticios):
  - Margen bruto: 25 % del valor de compra.
  - Valor en riesgo de un cliente = margen de sus compras de los últimos 12 meses (se pierde si abandona).
  - Costo por contacto de retención: COP 5.000 (costo esperado del cupón redimido + gestión). Ver sensibilidad.
  - Efectividad de la campaña: retiene al 25 % de los clientes que sí iban a abandonar.
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

SRC, OUT = sys.argv[1], sys.argv[2]
for sub in ["tablas", "graficos"]:
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
MARGEN, COSTO, EFECTIVIDAD = 0.25, 5_000, 0.25

df = pd.read_csv(SRC, encoding="utf-8-sig")
y = df.abandono_90d
X = df.drop(columns=["id_cliente", "abandono_90d"])
CAT = ["ciudad", "canal_principal"]
NUM = [c for c in X.columns if c not in CAT]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

pre_lin = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
pre_tree = ColumnTransformer([("num", "passthrough", NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
MODELOS = {
    "Referencia (tasa promedio)": Pipeline([("pre", pre_tree), ("m", DummyClassifier(strategy="prior"))]),
    "Regresión logística": Pipeline([("pre", pre_lin), ("m", LogisticRegression(max_iter=2000))]),
    "Random Forest": Pipeline([("pre", pre_tree), ("m", RandomForestClassifier(n_estimators=400, min_samples_leaf=20, n_jobs=-1, random_state=42))]),
    "HistGradientBoosting": Pipeline([("pre", pre_tree), ("m", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0, random_state=42))]),
}

# ---------- 1. Comparación con validación cruzada (solo entrenamiento) ----------
cv = StratifiedKFold(5, shuffle=True, random_state=42)
filas = []
for nom, m in MODELOS.items():
    r = cross_validate(m, X_tr, y_tr, cv=cv, scoring=["roc_auc", "average_precision"], return_train_score=True)
    filas.append({"modelo": nom, "auc_cv": r["test_roc_auc"].mean(), "auc_cv_de": r["test_roc_auc"].std(),
                  "pr_auc_cv": r["test_average_precision"].mean(), "auc_train": r["train_roc_auc"].mean()})
comp = pd.DataFrame(filas).sort_values("pr_auc_cv", ascending=False).round(4)
# Regla de selección: mejor PR-AUC en CV; si otro modelo más simple está a menos de 1 DE de AUC, se prefiere el simple.
SIMPLICIDAD = ["Regresión logística", "HistGradientBoosting", "Random Forest"]
mejor = comp.iloc[0]
empatados = comp[(comp.modelo != "Referencia (tasa promedio)") & (comp.auc_cv >= mejor.auc_cv - mejor.auc_cv_de)].modelo.tolist()
ganador = [m for m in SIMPLICIDAD if m in empatados][0]
print(comp.to_string(index=False)); print("Ganador:", ganador)

# ---------- 2. Evaluación en prueba ----------
evals = []
probs = {}
for nom, m in MODELOS.items():
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]; probs[nom] = p
    orden = np.argsort(-p); top10 = orden[: len(p) // 10]
    evals.append({"modelo": nom, "auc": roc_auc_score(y_te, p), "pr_auc": average_precision_score(y_te, p),
                  "brier": brier_score_loss(y_te, p),
                  "precision_top10": y_te.values[top10].mean(),
                  "captura_top10": y_te.values[top10].sum() / y_te.sum(),
                  "lift_top10": y_te.values[top10].mean() / y_te.mean()})
ev = pd.DataFrame(evals).set_index("modelo").loc[comp.modelo].round(4)
print(ev.to_string())
p = probs[ganador]
mod = MODELOS[ganador]

# ---------- 3. ¿Qué pesa más? (importancia por permutación en prueba) ----------
pi = permutation_importance(mod, X_te, y_te, scoring="average_precision", n_repeats=10, random_state=42, n_jobs=-1)
imp = pd.DataFrame({"variable": X_te.columns, "importancia": pi.importances_mean, "de": pi.importances_std}).sort_values("importancia", ascending=False)
NOMBRES = {"dias_desde_ultima_compra": "Días desde la última compra", "pct_entregas_tarde": "% de entregas tarde",
           "pedidos_12m": "Pedidos en 12 meses", "tickets_soporte_90d": "Casos de soporte (90 días)",
           "calificacion_ultima_compra": "Calificación de la última compra", "pct_apertura_correos": "% de apertura de correos",
           "antiguedad_meses": "Antigüedad (meses)", "programa_lealtad": "Programa de lealtad",
           "pct_pedidos_con_descuento": "% de pedidos con descuento", "categorias_distintas": "Categorías compradas",
           "ticket_promedio_cop": "Ticket promedio", "canal_principal": "Canal (app o web)", "ciudad": "Ciudad",
           "devoluciones_12m": "Devoluciones en 12 meses"}
imp["nombre"] = imp.variable.map(NOMBRES)

# ---------- 4. Decisión: a quién contactar ----------
te = X_te.copy(); te["abandono_real"] = y_te.values; te["prob_abandono"] = p
te["valor_en_riesgo_cop"] = te.ticket_promedio_cop * te.pedidos_12m * MARGEN
te["valor_esperado_contacto_cop"] = te.prob_abandono * EFECTIVIDAD * te.valor_en_riesgo_cop - COSTO

def curva(orden_idx):
    """Ganancia real acumulada en prueba si se contacta a los primeros k clientes del orden dado."""
    ben = (te.abandono_real.values * EFECTIVIDAD * te.valor_en_riesgo_cop.values)[orden_idx]
    return np.cumsum(ben - COSTO)

n = len(te)
o_valor = np.argsort(-te.valor_esperado_contacto_cop.values)
o_prob = np.argsort(-te.prob_abandono.values)
o_azar = np.random.default_rng(7).permutation(n)
g_valor, g_prob, g_azar = curva(o_valor), curva(o_prob), curva(o_azar)
k_opt = int(np.argmax(g_valor)) + 1
k_regla = int((te.valor_esperado_contacto_cop > 0).sum())
resumen = pd.DataFrame([
    ("Modelo elegido", ganador),
    ("Clientes en prueba", n),
    ("Tasa de abandono real en prueba (%)", round(te.abandono_real.mean() * 100, 1)),
    ("Clientes a contactar según la regla (valor esperado > 0)", k_regla),
    ("% de la base a contactar (regla)", round(k_regla / n * 100, 1)),
    ("Ganancia neta real con la regla (COP)", int(g_valor[k_regla - 1])),
    ("Óptimo en retrospectiva: clientes", k_opt),
    ("Óptimo en retrospectiva: ganancia neta (COP)", int(g_valor[k_opt - 1])),
    ("Mismo número de contactos por probabilidad (COP)", int(g_prob[k_regla - 1])),
    ("Mismo número de contactos al azar (COP)", int(g_azar[k_regla - 1])),
    ("Contactar a todos (COP)", int(g_valor[-1])),
    ("Abandonos capturados por la regla (%)", round(te.abandono_real.values[o_valor[:k_regla]].sum() / te.abandono_real.sum() * 100, 1)),
], columns=["indicador", "valor"])
print(resumen.to_string(index=False))

# Sensibilidad de la decisión a los supuestos
sens = []
for costo in [2_000, 5_000, 10_000, 15_000]:
    for ef in [0.15, 0.25, 0.35]:
        ve = te.prob_abandono * ef * te.valor_en_riesgo_cop - costo
        sel = (ve > 0).values
        gan = (te.abandono_real.values * ef * te.valor_en_riesgo_cop.values - costo)[sel].sum()
        sens.append({"costo_contacto_cop": costo, "efectividad": ef, "clientes_a_contactar": int(sel.sum()),
                     "pct_base": round(sel.mean() * 100, 1), "ganancia_neta_cop": int(gan),
                     "abandonos_capturados_pct": round(te.abandono_real.values[sel].sum() / te.abandono_real.sum() * 100, 1)})
sens = pd.DataFrame(sens)
print(sens.to_string(index=False))

# Acción sugerida por cliente contactado (regla transparente sobre el motivo dominante)
obj = te.iloc[o_valor[:k_regla]].copy()
def accion(r):
    if r.pct_entregas_tarde >= 0.20 or r.tickets_soporte_90d >= 2 or r.calificacion_ultima_compra <= 2:
        return "Recuperar el servicio (disculpa + envío gratis)"
    if r.antiguedad_meses <= 3:
        return "Acompañar al cliente nuevo (segunda compra guiada)"
    if r.pct_pedidos_con_descuento >= 0.35 and r.ticket_promedio_cop < 70_000:
        return "Cupón focalizado en su categoría"
    if r.dias_desde_ultima_compra >= 90:
        return "Reactivar (recordatorio + novedad)"
    return "Invitar al programa de lealtad"
obj["accion_sugerida"] = obj.apply(accion, axis=1)
acc = obj.groupby("accion_sugerida").agg(clientes=("prob_abandono", "size"), prob_media=("prob_abandono", "mean"),
                                          valor_en_riesgo_cop=("valor_en_riesgo_cop", "sum")).sort_values("clientes", ascending=False)
acc["pct"] = acc.clientes / acc.clientes.sum() * 100
print(acc.round(2).to_string())

# Calibración
fr, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
cal = pd.DataFrame({"prob_predicha_media": mp, "tasa_real": fr}).round(3)

with pd.ExcelWriter(os.path.join(OUT, "tablas", "resultados_abandono_clientes.xlsx")) as xw:
    pd.DataFrame({"supuesto": ["Margen bruto", "Costo por contacto (COP)", "Efectividad de la campaña", "Ventana de abandono", "Datos"],
                  "valor": [MARGEN, COSTO, EFECTIVIDAD, "90 días (1 jul – 30 sep 2026)", "Sintéticos: 20.000 clientes de TiendaLab (ficticia)"]}).to_excel(xw, "00_supuestos", index=False)
    comp.to_excel(xw, "01_comparacion_cv", index=False)
    ev.to_excel(xw, "02_prueba")
    imp[["nombre", "variable", "importancia", "de"]].round(4).to_excel(xw, "03_importancia", index=False)
    resumen.to_excel(xw, "04_decision", index=False)
    acc.round(2).to_excel(xw, "05_acciones")
    sens.to_excel(xw, "05b_sensibilidad_supuestos", index=False)
    cal.to_excel(xw, "06_calibracion", index=False)
    pd.DataFrame({"pct_contactado": np.arange(1, n + 1) / n * 100, "ganancia_por_valor": g_valor,
                  "ganancia_por_probabilidad": g_prob, "ganancia_al_azar": g_azar}).iloc[::50].round(1).to_excel(xw, "07_curva_ganancia", index=False)
obj[["prob_abandono", "valor_en_riesgo_cop", "valor_esperado_contacto_cop", "accion_sugerida"]].round(3).head(300) \
    .to_csv(os.path.join(OUT, "tablas", "muestra_lista_de_contacto.csv"), encoding="utf-8-sig")

# ---------- 5. Gráficos (paleta SP Lab) ----------
FUENTES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fuentes")
for f in sorted(os.listdir(FUENTES)) if os.path.isdir(FUENTES) else []:
    if f.endswith(".ttf"):
        fm.fontManager.addfont(os.path.join(FUENTES, f))
_disp = {x.name for x in fm.fontManager.ttflist}
TITULO = "Montserrat" if "Montserrat" in _disp else "DejaVu Sans"   # títulos (marca SP Lab)
TEXTO = "Open Sans" if "Open Sans" in _disp else "DejaVu Sans"      # textos y etiquetas
plt.rcParams.update({"font.family": TEXTO, "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#F8FAF7", "axes.facecolor": "#F8FAF7"})
VERDE, GRIS, CARBON, DORADO, NEUTRO = "#245B45", "#66736C", "#202824", "#D6A64F", "#B9C2BC"
fmt = lambda v, d=0: f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
PIE = "Datos SINTÉTICOS de TiendaLab, tienda ficticia (20.000 clientes) · Análisis: Sebas Padilla Lab"
def enc(fig, t, s):
    fig.text(0.02, 0.965, t, fontsize=15, fontweight="bold", fontfamily=TITULO, color=CARBON, va="top")
    fig.text(0.02, 0.905, s, fontsize=10.5, color=GRIS, va="top")
    fig.text(0.02, 0.02, PIE, fontsize=8, color=GRIS)

# 5.1 Importancia
top = imp.head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6)); fig.subplots_adjust(left=0.33, right=0.93, top=0.82, bottom=0.09)
ax.barh(top.nombre, top.importancia, color=[VERDE if v in ("dias_desde_ultima_compra", "pct_entregas_tarde") else NEUTRO for v in top.variable], height=0.66)
for i, v in enumerate(top.importancia):
    ax.text(v + top.importancia.max() * 0.01, i, fmt(v, 3), va="center", fontsize=9.5, color=CARBON)
ax.set_xticks([]); ax.tick_params(axis="y", length=0, colors=CARBON)
for s in ["left", "bottom"]: ax.spines[s].set_visible(False)
enc(fig, "Días sin comprar y entregas tarde: las señales que más pesan", "Importancia por permutación: cuánto cae la precisión del modelo (PR-AUC) si se desordena cada variable")
fig.savefig(os.path.join(OUT, "graficos", "01_que_pesa_mas.png"), dpi=200); plt.close(fig)

# 5.2 Curva de captura
pct = np.arange(1, n + 1) / n * 100
cap = np.cumsum(te.abandono_real.values[o_prob]) / te.abandono_real.sum() * 100
fig, ax = plt.subplots(figsize=(9, 6)); fig.subplots_adjust(left=0.1, right=0.95, top=0.82, bottom=0.13)
ax.plot(pct, cap, color=VERDE, lw=2.4, label=f"Modelo ({ganador})")
ax.plot([0, 100], [0, 100], color=NEUTRO, lw=1.6, ls="--", label="Al azar")
c20 = cap[int(n * 0.2) - 1]
ax.scatter([20], [c20], color=DORADO, s=60, zorder=4)
ax.annotate(f"Contactando al 20 %\nse llega al {fmt(c20)} % de quienes se van", (20, c20), xytext=(32, c20 - 18),
            fontsize=9.5, color=CARBON, arrowprops=dict(arrowstyle="-", color=GRIS))
ax.set_xlabel("% de clientes contactados (de mayor a menor riesgo)", color=GRIS); ax.set_ylabel("% de los abandonos capturados", color=GRIS)
ax.set_xlim(0, 100); ax.set_ylim(0, 102); ax.tick_params(colors=GRIS)
ax.legend(frameon=False, loc="lower right")
enc(fig, f"Con el 20 % de los contactos se llega al {fmt(c20)} % de los clientes que se van", "Curva de captura en el conjunto de prueba (5.000 clientes)")
fig.savefig(os.path.join(OUT, "graficos", "02_curva_captura.png"), dpi=200); plt.close(fig)

# 5.3 Curva de ganancia
fig, ax = plt.subplots(figsize=(9, 6)); fig.subplots_adjust(left=0.13, right=0.95, top=0.82, bottom=0.13)
M = 1e6
ax.plot(pct, g_valor / M, color=VERDE, lw=2.4, label="Ordenando por valor esperado")
ax.plot(pct, g_prob / M, color=GRIS, lw=1.6, label="Ordenando solo por probabilidad")
ax.plot(pct, g_azar / M, color=NEUTRO, lw=1.6, ls="--", label="Al azar")
ax.axhline(0, color=CARBON, lw=0.8)
xr = k_regla / n * 100; yr = g_valor[k_regla - 1] / M
ax.scatter([xr], [yr], color=DORADO, s=70, zorder=5)
ax.annotate(f"Regla: contactar si el valor esperado > 0\n{fmt(k_regla)} clientes ({fmt(xr)} %) · COP {fmt(yr, 1)} millones",
            (xr, yr), xytext=(42, yr * 0.95), fontsize=9.5, color=CARBON, arrowprops=dict(arrowstyle="-", color=GRIS))
ax.set_xlabel("% de clientes contactados", color=GRIS); ax.set_ylabel("Ganancia neta de la campaña (millones COP)", color=GRIS)
ax.tick_params(colors=GRIS); ax.set_xlim(0, 100)
ax.legend(frameon=False, loc="lower left")
enc(fig, "Contactar a todos pierde plata; contactar a pocos y bien elegidos gana",
      f"Ganancia neta en prueba con supuestos ficticios: margen {int(MARGEN*100)} %, contacto COP {fmt(COSTO)}, efectividad {int(EFECTIVIDAD*100)} %")
fig.savefig(os.path.join(OUT, "graficos", "03_curva_ganancia.png"), dpi=200); plt.close(fig)
print(imp.head(8).to_string(), "\n", cal.to_string())

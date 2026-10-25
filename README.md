# ¿Qué clientes están por irse? · v1

**Sebas Padilla Lab · Pilar Machine Learning · Versión 1 (25 oct 2026)**

> **Pregunta:** ¿Qué clientes de una tienda en línea van a dejar de comprar en los próximos 90 días, a quiénes conviene contactar y con qué acción?

> ⚠️ **Datos sintéticos.** "TiendaLab" es una tienda en línea ficticia. Los 20.000 clientes se generaron con código (`scripts/01_generar_datos.py`). Ninguna fila corresponde a una persona o empresa real, y no hay datos de pagos. Los resultados son hallazgos del análisis, no impacto real.

## English summary
**Which customers are about to leave?** Churn model on **synthetic** data (20,000 customers of a fictional online store) turned into an economic retention decision. Logistic regression (AUC 0.82, tied with Random Forest and gradient boosting; chosen for simplicity and calibration). Contacting the top 20% riskiest customers reaches 58% of churners. Contacting everyone loses COP 10.2M on the test set, while contacting only customers with positive expected value earns COP 2.8M; ranking by expected value yields 2.5× more than ranking by churn probability alone. Includes a sensitivity table for cost and campaign effectiveness. Python · scikit-learn · pandas.

## En 30 segundos
- La tasa de abandono es del 16,8 %. El modelo elegido (regresión logística) ordena bien el riesgo: **AUC 0,82** y una **PR-AUC de 0,56**, frente a 0,17 de la referencia.
- **Contactando al 20 % de mayor riesgo se llega al 58 % de los clientes que se van.** En el 10 % de mayor riesgo, 2 de cada 3 sí abandonan (3,9 veces más que el promedio).
- **La señal más fuerte son los días sin comprar; la más accionable, las entregas tarde.** Le siguen la frecuencia de compra y los casos de soporte.
- **La campaña no debe ir a todos.** Con los supuestos base, contactar a toda la base **pierde COP 10,2 millones**, mientras que la regla "contactar solo si el valor esperado es mayor que 0" **gana COP 2,8 millones**. Esa regla selecciona al 15,5 % de los clientes de prueba y captura un tercio de los abandonos.
- **Ordenar por valor esperado rinde 2,5 veces más** que ordenar solo por probabilidad de abandono, con el mismo número de contactos (COP 2,8 millones frente a 1,1 millones). No basta con saber quién se va: importa cuánto vale.
- **Qué hacer con los contactados:** el 62 % necesita una recuperación del servicio (entregas tarde, soporte o mala calificación), y el 29 % una reactivación por inactividad.

## Datos (sintéticos)
- 20.000 clientes con su foto al 30 jun 2026. **Abandono** = no compra entre el 1 jul y el 30 sep 2026.
- 14 variables: ciudad, antigüedad, canal (app o web), programa de lealtad, pedidos en 12 meses, días desde la última compra, ticket promedio, categorías compradas, % de pedidos con descuento, % de entregas tarde, devoluciones, casos de soporte, calificación de la última compra y % de apertura de correos.
- El abandono se genera con reglas ocultas: recencia, frecuencia, entregas tarde, soporte, mala calificación, lealtad, canal, cazadores de descuento de ticket bajo, clientes nuevos y apertura de correos, más ruido. El modelo no ve esas reglas. Que las recupere en su ranking de importancia es una prueba de que el método funciona.

## Enfoque
1. **Partición:** 75 % entrenamiento y 25 % prueba (5.000 clientes), estratificada.
2. **Comparación con validación cruzada (5 folds):**

| Modelo | AUC CV | PR-AUC CV | AUC entrenamiento |
| --- | --- | --- | --- |
| Random Forest | 0,819 | 0,565 | 0,903 |
| Regresión logística | 0,819 | 0,563 | 0,821 |
| HistGradientBoosting | 0,818 | 0,561 | 0,867 |
| Referencia (tasa promedio) | 0,500 | 0,168 | 0,500 |

3. **Regla de selección:** gana la mejor PR-AUC, pero si un modelo más simple está a menos de una desviación estándar, se prefiere el simple. Los tres empatan, así que se eligió la **regresión logística**: es igual de precisa, no se sobreajusta (Random Forest pasa de 0,90 en entrenamiento a 0,82 en validación), es más fácil de explicar y es la mejor calibrada (Brier de 0,104). La buena calibración importa, porque la probabilidad se usa para calcular dinero.
4. **Importancia por permutación** en el conjunto de prueba.
5. **Decisión económica:**
   - Valor esperado de contactar a un cliente = probabilidad de abandono × efectividad × valor en riesgo − costo del contacto.
   - Valor en riesgo = margen de sus compras de los últimos 12 meses.

### Supuestos de negocio (ficticios y editables)
| Supuesto | Valor base |
| --- | --- |
| Margen bruto | 25 % |
| Costo por contacto (cupón esperado + gestión) | COP 5.000 |
| Efectividad (abandonos evitados entre quienes sí se iban) | 25 % |

**Sensibilidad (en prueba):**
- Con un costo de COP 2.000 y una efectividad del 25 %, conviene contactar a la mitad de la base (ganancia de COP 6,9 millones).
- Con COP 15.000 y una efectividad del 15 %, ninguna campaña es rentable.

La tabla completa está en `05b_sensibilidad_supuestos`. La decisión depende más del costo del contacto que del modelo.

## Decisión recomendada
1. Puntuar a toda la base cada mes y contactar solo a los clientes con valor esperado positivo.
2. Asignar la acción según el motivo dominante:

| Acción sugerida | % de contactados |
| --- | --- |
| Recuperar el servicio (disculpa + envío gratis) | 62 % |
| Reactivar (recordatorio + novedad) | 29 % |
| Invitar al programa de lealtad | 5 % |
| Acompañar al cliente nuevo (segunda compra guiada) | 2 % |
| Cupón focalizado en su categoría | 1 % |

3. Atacar la causa: si las entregas tarde explican la mayoría de los contactos, mejorar la logística puede valer más que cualquier cupón.
4. Medir con un **grupo de control** que no se contacte. La efectividad del 25 % es un supuesto, y solo un experimento la confirma.

## Contenido de la carpeta
| Archivo | Contenido |
| --- | --- |
| `datos/clientes_tiendalab_sinteticos.csv` | Los 20.000 clientes sintéticos |
| `tablas/resultados_abandono_clientes.xlsx` | Supuestos, comparación CV, prueba, importancia, decisión, acciones, sensibilidad, calibración, curva de ganancia |
| `tablas/muestra_lista_de_contacto.csv` | Primeros 300 clientes a contactar, con probabilidad, valor y acción sugerida |
| `graficos/01_que_pesa_mas.png` | Importancia de variables |
| `graficos/02_curva_captura.png` | % de abandonos capturados según % contactado |
| `graficos/03_curva_ganancia.png` | Ganancia neta según cuántos clientes se contactan |
| `scripts/01_generar_datos.py` · `scripts/02_modelo_y_decision.py` | Código reproducible (semilla 42) |
| `fuentes/` | Tipografías de la marca (Montserrat y Open Sans, licencia SIL Open Font License) usadas en los gráficos |

Para reproducir: `pip install -r requirements.txt`, luego `python scripts/01_generar_datos.py datos/clientes_tiendalab_sinteticos.csv` y `python scripts/02_modelo_y_decision.py datos/clientes_tiendalab_sinteticos.csv .`

## Limitaciones
- **Datos sintéticos:** el modelo aprende reglas que también fueron escritas en código. Con datos reales, las relaciones serán más ruidosas y el AUC probablemente más bajo.
- **Una sola foto en el tiempo:** en producción se validaría con cortes posteriores (validación temporal).
- **Predecir abandono no es predecir quién responde a la campaña.** El siguiente paso es un modelo de *uplift* con grupo de control.
- La importancia por permutación muestra asociación, no causa.

## Licencia
- **Código** (`scripts/`): © 2026 Sebastián Padilla Castaño. Todos los derechos reservados. Puedes leerlo y ejecutarlo para revisar el análisis, pero no reutilizarlo ni redistribuirlo sin autorización.
- **Datos, tablas y gráficos:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.es). Los datos sintéticos (`datos/`), las tablas y los gráficos son creación del autor y se comparten bajo CC BY 4.0.
- **Tipografías** (`fuentes/`): Montserrat y Open Sans, bajo SIL Open Font License 1.1 (ver `fuentes/OFL-*.txt`).
- Ver `LICENSE.md`.

---
Parte de **Sebas Padilla Lab** · De los datos a las decisiones.

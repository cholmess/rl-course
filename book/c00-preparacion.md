# Capítulo 0: Preparación y diagnóstico

*Calificación: diagnóstico 60 % (12,5/21), repaso 6/8 · Noviembre 2026. El experimento de CartPole se añadirá aquí cuando se califique.*

## Las ideas en breve

**RL frente a aprendizaje supervisado.** En supervisado hay una respuesta correcta por ejemplo. En RL no: hay una señal de recompensa, a menudo tardía. Tres cosas lo hacen difícil:
1. No te dicen qué acción era la correcta, solo cuánta recompensa obtuviste.
2. Tus acciones cambian los datos que ves después, así que hay que **explorar**.
3. La recompensa puede llegar mucho después de la acción que la causó (**asignación de crédito**).

**Política, episodio, retorno.** Una *política* es una regla que convierte una observación en una acción. Un *episodio* va desde `reset` hasta que termina: `terminated` (el entorno acabó, por ejemplo cae el palo) o `truncated` (se alcanzó el límite de pasos, 500 en CartPole). El *retorno* es la suma de recompensas.

**Descuento.** El retorno descontado es G = r₀ + γr₁ + γ²r₂ + … Si la recompensa es constante r para siempre, es una serie geométrica:

G = r / (1 − γ).

Con r = 1 y γ = 0,9: G = 10. Con r = 2 y γ = 0,5: G = 4. El descuento hace que la suma sea finita aunque nunca termine, y 1/(1 − γ) es el «horizonte efectivo» del agente.

**Valor esperado y varianza.**
- Var(X) = E[X²] − (E[X])². Para una variable 0/1 con probabilidad p: Var = p(1 − p).
- E[f(X)] ≠ f(E[X]) en general. Si X vale +2 o −2 con igual probabilidad, E[X] = 0, pero X² siempre vale 4, así que E[X²] = 4 y (E[X])² = 0. La diferencia, 4, es justo la varianza.
- El error de una media de N muestras baja como 1/√N: para reducirlo a la mitad hacen falta cuatro veces más muestras.

**Probabilidad condicional.** Cuando te dan información, el espacio de resultados se reduce. Dos monedas, sabiendo que *al menos una* es cara: quedan HH, HT, TH (TT queda descartado), y solo HH es «las dos caras», así que 1/3. Con dos dados y suma 4: (1,3), (2,2), (3,1), y P(primero = 1) = 1/3.

**Descenso de gradiente.** w ← w − α·f′(w). Para f(w) = (w − 3)² desde w = 0 con α = 0,1: f′(0) = −6, así que w = 0,6. Un paso no salta al mínimo: se acerca. El gradiente solo conoce el entorno inmediato; no da «el camino más corto al óptimo».

**Álgebra lineal que usarás pronto.** Si v = r + γPv, entonces v = (I − γP)⁻¹ r. Es exactamente cómo se evalúa una política en el módulo 2.

**NumPy.** `x.sum(axis=k)` elimina el eje k. Con forma (4, 7): `axis=1` → (4,), `axis=0` → (7,). Se cumple por todas partes: la forma pierde el eje que nombras.

**Bucle de entrenamiento en PyTorch.** Siempre en este orden:
1. `optimizer.zero_grad()`: borra los gradientes acumulados (PyTorch *suma* en `.grad`).
2. Paso hacia delante y cálculo de la pérdida.
3. `loss.backward()`: calcula los gradientes y los guarda en `.grad`. **No cambia los pesos.**
4. `optimizer.step()`: actualiza los pesos usando esos gradientes.

## Cómo se conecta
Los bandidos (módulo 1) son puro valor esperado y varianza. El descuento y la ecuación v = (I − γP)⁻¹ r son el módulo 2. El bucle de PyTorch vuelve en el módulo 7 y se usa en todo lo demás.

## Tus errores

| Pregunta | Qué respondiste | Por qué estaba mal | Estado |
|---|---|---|---|
| Var de una variable 0/1 con p = 0,3 | 0,09 | Es p², o sea (E[X])². Falta restarlo a E[X²] = 0,3: 0,3 − 0,09 = 0,21. | corregido |
| 2 monedas, al menos una cara, P(ambas) | 0,5 «por independencia» | Las monedas son independientes, pero la información elimina TT. | corregido |
| Retorno con r = 1, γ = 0,9 | infinito | Serie geométrica convergente: 1/(1 − 0,9) = 10. | corregido |
| Un paso de gradiente en (w − 3)² | 3 | Un paso da 0,6. El gradiente se acerca, no salta. | corregido |
| `loss.backward()` | «actualiza los pesos» | Solo calcula gradientes. Los pesos cambian con `step()`. | corregido |
| `zero_grad()` | «congela los pesos» | Borra los gradientes acumulados. | corregido |
| E[X²] con X = ±2 | 0 | Cuadrar antes de promediar da 4. Promediar y cuadrar no conmutan. | **abierto** |
| `sum(axis=1)` en forma (4, 7) | (7,) | El eje nombrado desaparece: (4,). Pruébalo con `np.ones((4,7)).sum(axis=1).shape`. | **abierto** |

Dos fallos «de ideas» más: pensar que el gradiente da el camino más corto al óptimo, y describir RL solo como «recompensas y penalizaciones» sin mencionar exploración ni crédito retardado.

Lo que sí dominas: álgebra lineal (4/4), derivadas (regla del producto y de la cadena), indexación con máscaras en NumPy, `argmax` y semillas reproducibles.

## Mini-test de repaso
1. Var(X) si X = 1 con prob. 0,6 y 0 si no.
2. Retorno total con recompensa 3 por paso para siempre y γ = 0,5.
3. X = ±3 con igual probabilidad: ¿E[X²] y (E[X])²?
4. Forma de `np.zeros((6, 2, 5)).mean(axis=1)`.
5. Orden correcto de las cuatro líneas del bucle de PyTorch.

*Respuestas:* 1) 0,24. 2) 6. 3) 9 y 0. 4) (6, 5). 5) zero_grad, pérdida, backward, step.

## Hacia tu problema en el trabajo
Ya sabes que el retorno descontado es lo que se optimiza, y que la varianza de las estimaciones decide cuántos datos necesitas. Con datos de logs de empresa (fase 5) la pregunta clave será cuántas muestras hacen falta para estimar bien una política nueva: la regla 1/√N empieza aquí.

## Mis notas
*(tuyas)*

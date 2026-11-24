# Module 0 write-up

## Predictions (escritas después de ejecutar el código)

Estas no son predicciones registradas de antemano. Son una reconstrucción honesta de lo
que habría esperado antes de ver los resultados, escrita después de ejecutar el código.

Mean return over 100 episodes (max 500):

| Policy | My prediction | Why |
|---|---|---|
| random | 25 | Sin información sobre el estado, las acciones aleatorias deberían dejar caer el poste rápidamente. |
| angle only | 120 | Empujar hacia donde se inclina el poste debería corregirlo durante un tiempo, aunque reaccionaría tarde por no considerar su inercia. |
| angle + velocity | 400 | Anticipar hacia dónde se mueve el poste debería estabilizarlo mucho mejor, aunque esperaba algún fallo debido al movimiento del carro. |

## Results

| policy | mean | std | min | max | % at 500 |
|---|---:|---:|---:|---:|---:|
| random | 25.0 | 15.4 | 9 | 93 | 0% |
| angle only | 41.0 | 8.6 | 25 | 58 | 0% |
| angle + velocity | 500.0 | 0.0 | 500 | 500 | 100% |

![results](results.png)

## Questions

1. No puedo medir limpiamente la precisión de estas predicciones porque las escribí después
   de ejecutar el experimento. Comparando la reconstrucción con los resultados, habría
   acertado la política aleatoria (25 esperado y 25.0 observado). Mi mayor error habría sido
   la política basada solo en el ángulo: esperaba 120 y obtuvo 41.0. Sobreestimé cuánto podía
   compensar una regla reactiva sin conocer la velocidad angular. También habría subestimado
   la política con velocidad: esperaba 400, pero llegó a 500 en todas las semillas.

2. La política basada solo en el ángulo empuja hacia el lado al que el poste ya está
   inclinado, pero ignora la velocidad con la que cae. Cuando el poste cruza la vertical, aún
   conserva momento angular; la política cambia de acción por el signo del ángulo, pero lo
   hace demasiado tarde para frenar ese movimiento. El carro y el poste oscilan de un lado
   al otro, las correcciones se retrasan y la amplitud aumenta hasta que el ángulo supera el
   límite del entorno. Añadir `theta_dot` aporta amortiguamiento: permite actuar antes de que
   la inclinación sea grande.

3. Elegí `k = 0.5` y uso la regla `theta + k * theta_dot > 0` para empujar a la derecha.
   Comparé 100 episodios para varios valores:

   | k | mean | std | min | max | % at 500 |
   |---:|---:|---:|---:|---:|---:|
   | 0.0 | 41.0 | 8.6 | 25 | 58 | 0% |
   | 0.1 | 500.0 | 0.0 | 500 | 500 | 100% |
   | 0.5 | 500.0 | 0.0 | 500 | 500 | 100% |
   | 1.0 | 493.1 | 30.9 | 320 | 500 | 95% |
   | 2.0 | 414.1 | 119.1 | 171 | 500 | 58% |

   La sorpresa es que un término de velocidad muy pequeño (`k = 0.1`) basta para resolver
   los 100 episodios, y que la regla tolera un rango amplio: incluso `k = 1` casi siempre
   llega a 500. Con `k = 0`, la acción llega tarde y el poste oscila hasta caer. Si `k` es
   demasiado grande, la regla reacciona en exceso a la velocidad angular y pierde precisión
   sobre la inclinación actual; el rendimiento se vuelve más variable y aparecen fallos.

4. Con el límite normal, `k = 0.5` no falló en 100 episodios. Al usar
   `gym.make("CartPole-v1", max_episode_steps=5000)`, sí falla en varias semillas porque el
   carro sale de la pista, no porque caiga el poste. Por ejemplo, con la semilla 0 terminó a
   los 582 pasos con la observación final
   `[x=-2.4023, x_dot=-0.0121, theta=0.0003, theta_dot=-0.2897]`. La posición `x` cruzó el
   límite de `-2.4`, mientras el ángulo seguía prácticamente en cero. Mi regla solo usa
   `theta` y `theta_dot`, así que puede mantener el poste vertical mientras deja que el carro
   derive hasta un borde. Para corregirlo añadiría realimentación débil de la posición y
   velocidad del carro, por ejemplo una puntuación del tipo
   `theta + k * theta_dot - c * x - d * x_dot`, y ajustaría `c` y `d` sin sacrificar el
   equilibrio del poste.

5. Este enfoque no escala porque exige diseñar y ajustar manualmente una regla usando una
   comprensión clara de la dinámica. En problemas con observaciones de alta dimensión,
   muchas acciones y efectos retardados, no es práctico decidir a mano qué variables usar ni
   cómo combinarlas. Una política aprendida puede extraer esas relaciones de la experiencia.

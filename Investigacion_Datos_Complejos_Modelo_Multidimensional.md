# Datos complejos en el Modelo Multidimensional

## 1. El Espacio Vectorial de las Imágenes

**Pregunta:** Si queremos que una red neuronal analice una imagen a color (RGB) de 256x256 píxeles, ¿cómo se representa esto matemáticamente en un arreglo multidimensional (Tensor)? ¿Cuál sería el valor de p (número de dimensiones) para una sola imagen?

**Respuesta:**

Una imagen a color se guarda como un tensor de 3 dimensiones: alto, ancho y canales de color.
Para una imagen de 256x256 en RGB, la forma sería `(256, 256, 3)`. Es como tener tres matrices de 256x256 encimadas, una para el rojo, otra para el verde y otra para el azul. Cada valor es la intensidad del color en ese píxel (de 0 a 255).
![img.png](img.png)
Si queremos meter la imagen como una fila de la matriz X, hay que "aplanarla", o sea, poner todos los valores en un solo vector. Entonces:

p = 256 x 256 x 3 = **196,608 dimensiones**

El problema es que al aplanarla se pierde la idea de qué píxel está junto a cuál, y además quedan demasiadas dimensiones para una sola imagen. Por eso se usan redes convolucionales (CNN), que trabajan con la imagen sin aplanarla y se fijan en zonas pequeñas de la imagen.

## 2. El Espacio Vectorial del Texto (Word Embeddings / TF-IDF)

**Pregunta:** Un algoritmo no puede calcular la distancia euclidiana entre la palabra "Feliz" y "Triste". Investiga brevemente cómo el Procesamiento de Lenguaje Natural convierte palabras u oraciones completas en vectores numéricos dentro del modelo multidimensional.

**Respuesta:**

Como las palabras no son números, primero hay que convertirlas en vectores. Hay varias formas de hacerlo:

- **One-hot / Bolsa de palabras:** se hace una lista con todas las palabras del vocabulario y cada palabra es un vector lleno de ceros con un 1 en su posición. Funciona, pero todas las palabras quedan igual de lejos entre sí, así que no dice nada del significado.
- **TF-IDF:** a cada palabra de un documento se le da un peso. Sube si la palabra se repite mucho en ese documento y baja si aparece en casi todos los documentos (como "el" o "de"). Así cada documento se vuelve un vector y se pueden comparar documentos entre sí.
- **Word Embeddings (Word2Vec, GloVe):** una red neuronal aprende un vector para cada palabra, normalmente de unas 100 a 300 dimensiones, viendo con qué otras palabras suele aparecer. Las palabras que se usan en contextos parecidos terminan cerca. El ejemplo clásico es que "rey - hombre + mujer" da un vector muy parecido a "reina".
- **Modelos como BERT:** hacen lo mismo pero toman en cuenta toda la oración, así que una misma palabra puede tener vectores distintos según el contexto. También sirven para sacar un vector de una oración completa.

Algo curioso es que "Feliz" y "Triste" pueden quedar cerca en un embedding, porque se usan en frases parecidas ("me siento feliz" / "me siento triste"). Además, en texto se suele usar la similitud coseno en lugar de la distancia euclidiana, porque importa más la dirección del vector que su tamaño.

## 3. La Maldición de la Dimensionalidad

**Pregunta:** De acuerdo a los conceptos de estadística y algoritmos basados en distancias (como se aborda en ISLP), ¿qué le ocurre geométricamente a la distancia entre los usuarios en la matriz X cuando el número de dimensiones (p) crece a miles o millones?

**Respuesta:**

Cuando p crece mucho, las distancias entre usuarios empiezan a parecerse todas. El usuario más cercano y el más lejano terminan casi a la misma distancia, así que ya no tiene mucho sentido hablar de "vecinos cercanos". Esto afecta a algoritmos como KNN o clustering, que dependen de las distancias.

Esto pasa por varias razones:

- **El espacio se vuelve muy grande y vacío.** Con cada dimensión nueva el volumen crece muchísimo, y los datos que tenemos quedan muy dispersos. Para llenar ese espacio necesitaríamos una cantidad enorme de datos.
- **Los "vecinos" ya no están cerca.** Como en ISLP con KNN, para encontrar aunque sea unos pocos vecinos hay que buscar en una zona tan grande que ya no es realmente local.
- **Las variables que no importan meten ruido.** Cada dimensión extra suma a la distancia, aunque no tenga nada que ver con el problema, y eso tapa las variables que sí son útiles.
- **Sobreajuste.** Si hay más variables que datos, el modelo se aprende los datos de memoria y no generaliza bien.

Para evitarlo se puede reducir la cantidad de dimensiones (por ejemplo con PCA), escoger solo las variables importantes, usar regularización y estandarizar los datos antes de medir distancias.

Ejemplo en Python para verlo:

```python
import numpy as np

for p in [2, 10, 100, 1000, 10000]:
    X = np.random.rand(500, p)
    d = np.linalg.norm(X[1:] - X[0], axis=1)
    print(p, (d.max() - d.min()) / d.min())
```

Mientras más grande es p, más se acerca el resultado a 0. Eso significa que la distancia al más cercano y al más lejano casi no cambia.

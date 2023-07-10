# Segmentación de fluido patológico en imágenes tomográficas con modelos neuronales

Este código proporciona la implementación de múltiples modelos neuronales probados para la segmentación semántica de imágenes tomográficas. La técnica OCT se ha popularizado en tareas de segmentación en *Computer Vision* aplicadas a imágenes biomédicas para detectar de forma no invasiva la presencia de fluido patológico en la retina.

Nuestro *dataset* consiste en 50 imágenes tomográficas (consideradas como un conjunto de datos limitado, ya que la mayoría de *datasets* populares cuentan con terabytes de imágenes), donde aproximadamente el 2% de los píxeles se corresponden con etiquetas de presencia de fluido patológico (por tanto tenemos un problema de desbalanceo de clases). No obstante, obtenemos resultados prometedores evaluando con las métricas F-Score e IoU utilizando *data augmentation* y aprendizaje adversario.

Las distintas implementaciones soportadas por este repositorio están escritas en el [Artículo](article.pdf) adjunto. En este README se explica cómo reproducir los resultados de entrenamiento y validación.


Se recomienda ejecutar todos los modelos utilizando la terminal con la siguiente sintaxis:


```shell
python3 test.py <model> <mode> -v -aug -adv  
```

- `model`: Especifica el modelo a utilizar. Las posibles elecciones son: `base`, `unet`, `linknet`, `pspnet`, `pan`, `attnunet`, `deformunet`.
- `mode`: Especifica el modo de ejecución (`kfold` o `train`). Utilizando `kfold` se acepta un argumento `--k`  para realizar `k` *fold cross validation* con el *dataset*. Las métricas finales se guardan en la carpeta  `--model_path`. Utilizando `train` se hará un único entrenamiento y validación con un *split* aleatorio (el 10% se reserva para validar).
- `-v`: Para mostrar el *trace* del etrenamiento.
- `-aug`: Para utilizar *data augmentation*.
- `-adv`: Para utilizar aprendizaje adversario. 


## Uso básico

A continuaciones proporcionamos comandos básicos para ejecutar el código y facilitar su comprensión:

- Ejecutar el modelo `base` sin *data augmentation* con *batch size* de 10 imágenes y guardar el resultado en  `../results/base`.

```shell
python3 test.py base kfold -v --batch_size=10 --model_path=../results/base/  
```

- Ejecutar el modelo `unet` con *data augmentation*:
```shell
python3 test.py unet kfold -v --batch_size=10 -aug  
```

- Ejecutar el modelo `linknet` con *data augmentation* y el *framework* de aprendizaje adversario:
```shell
python3 test.py unet train -v --batch_size=10 -aug  -adv
```

## Consulta del experimento

Los pesos de los modelos entrenados están disponibles en OneDrive a través del siguiente [enlace](https://udcgal-my.sharepoint.com/:f:/r/personal/ana_ezquerro_udc_es/Documents/GCED%20Q8/AIDA/p2?csf=1&web=1&e=HbyNlb). Para usarlos y poder comprobar las métricas obtenidas se recomienda consultar el Jupyter Notebook [eval.ipynb](eval.ipynb) adjunto.

Se puede consultar *in real time* los pesos de los modelos utilizados ejecutando a través de la terminal:

```shell
python3 counter.py
```

# Deep-tempest 2023

<img src="deep-tempest.png"/>

Este trabajo corresponde al Proyecto de Fin de Carrera en Ingeniería Eléctrica para la Facultad de Ingeniería de la Universidad de la República.

Integrantes:
- Santiago Fernández
- Emilio Martínez
- Gabriel Varela

Tutores:
- Federico La Rocca
- Pablo Musé

## Resumen

**Acá va el resumen ya corregido**

## Resultados
**Acá van tablas e imagenes de resultados de los metodos (reutilizar de los del overleaf)**

## Datos

Los datos utilizados [este enlace](https://finguy-my.sharepoint.com/:u:/g/personal/emilio_martinez_fing_edu_uy/EZ8KpQHJ7GZBvMRsBMtNj6gBkC3Fvivuz87-1fiQS6WKiw?e=LVjajm) dentro de un archivo ZIP (~7GB). Al descomprimirlo se pueden encontrar las imágenes sintéticas y capturadas realizadas para los experimentos, entrenamiento y evaluación durante el trabajo.

La estructura de los directorios es diferente para los datos sintéticos es diferente al de los capturados. 
**describir bien cómo es**

## Código y requerimientos

En cada una de los directorios se tiene una guía de cómo ejecutar las pruebas/entrenamiento/experimentos correspondientes. 

El código esta escrito en lenguaje Python versión 3.10, donde se utilizó ambientes de Anaconda. Para replicar el ambiente de trabajo crear uno nuevo con las bibliotecas del _requirements.txt_:

```shell
conda create --name deeptempest --file requirements.txt
```

Activarlo con:
```shell
conda activate deeptempest
```

## Referencias

**Referencia a códigos que reutilizamos**
- gr-tempest
- KAIR
    - DRUNet
    - PnP
- Maxima entropia (https://github.com/imadtoubal/Maximum-Entropy-Thresholding-Implementation-in-Python/blob/master/entropy_thresholding.ipynb)
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

**ACÁ VA el resumen ya corregido**

## Resultados
**ACÁ VAN tablas e imagenes de resultados de los metodos (reutilizar de los del overleaf)**

Más ejemplos se pueden ver en el directorio [deep-tempest_ejemplos](deep-tempest_ejemplos). 

También se pueden visualizar desde [este enlace](https://finguy-my.sharepoint.com/:f:/g/personal/emilio_martinez_fing_edu_uy/Eo_2mmNwq0lHguqmzjq7MyABb9pBbuDV3_EPOA9xGC-7vg?e=kevSbM) *(no estable)*. Aquí las imágenes están estructuradas con el siguiente orden:

1. Imagen original
2. Imagen espiada (_gr-tempest2.0_)
3. Imagen inferida por método _End-to-End_
4. Imagen inferida por método de _umbralización por máxima entropía_

## Datos

Los datos utilizados [este enlace](https://finguy-my.sharepoint.com/:u:/g/personal/emilio_martinez_fing_edu_uy/EZ8KpQHJ7GZBvMRsBMtNj6gBkC3Fvivuz87-1fiQS6WKiw?e=LVjajm) dentro de un archivo ZIP (~7GB). Al descomprimirlo se pueden encontrar las imágenes sintéticas y capturadas realizadas para los experimentos, entrenamiento y evaluación durante el trabajo.

La estructura de los directorios es diferente para los datos sintéticos es diferente al de los capturados. 
**ACA DESCRIBIR BIEN COMO ES**

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

**ACA PONER: Referencia a códigos que reutilizamos**
- gr-tempest
- KAIR
    - DRUNet
    - PnP
- Maxima entropia (https://github.com/imadtoubal/Maximum-Entropy-Thresholding-Implementation-in-Python/blob/master/entropy_thresholding.ipynb)

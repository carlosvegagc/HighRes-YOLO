# HighRes-Yolo
Model for using the YOLO core in its version 3 and 4 for object detection on large images. Works by including a previous and posterior layers for image slicing and recombination.

## General Info

---

Author: Carlos Vega García  
Year: 2022

## Includes

---
- Full modular approach with the implementation on Tensorflow.
- Prepared for custom object detection.
- Scripts for Anchor Calculation by K-means.
- Modular configuration with JSON files.
- Layers for slicing and combining the images.
- Global calculation of the mAP.
- Configurable Core with Tensorflow V4, V4-Lite, V3, V3-Lite.

## Acknowledgments

---

Thanks to the IUMA (Instituto Universitario de Microelectrónica Aplicada) from ULPGC (Universidad de Las Palmas de Gran Canaria) for supporting the development of this project.

Also it is important to thank some of the previous works that supports this project:

- You Only Look Twice: Rapid Multi-Scale Object Detection In Satellite Imagery (Adam Van Etten) [https://github.com/avanetten/yoltv4]
- YOLO V4 and V3 implementation over tensorflow by Python Lessons [https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3]

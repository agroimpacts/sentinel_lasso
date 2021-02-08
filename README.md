# Lasso Regression on Sentinel Data

Boka Luo

02/08/2021

This repo provides the protocol running lasso regression on calibrated [Sentinel-1 images from Google Earth Engine](https://developers.google.com/earth-engine/guides/sentinel1), including:

* mosaic images of the same dates
* resample images in time series
* apply iterative guided filter, which is observed to have less shifts than Lee filter, and can omit the gradient distortion phenomenon
* apply pixel-wise lasso regression
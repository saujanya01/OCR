#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:04:26 2019

@author: saujanya and naman(Team OCR)
"""

import cv2
import pytesseract
from wand.image import Image as wi
import numpy as np

path = str(input("Enter Image path : \n"))

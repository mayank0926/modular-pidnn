#!/bin/bash

pip install -r requirements.txt
gdown --id 1Y-0rTHlKJHRUXLoAHoZ2dxJBLiI1kPg4 && unzip Data.zip && rm Data.zip
gdown --id 1nDGmD2cDH0oiEcyjaSx2c7xmspUg73Li && unzip Models.zip && rm Models.zip

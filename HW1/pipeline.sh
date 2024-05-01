#!/bin/sh
cd ./ # Перейдем в текущую рабочую директорую, затем поочередно запустим со скриптами
python3 data_creation.py
python3 model_preprocessing.py
python3 model_preparation.py
python3 model_testing.py

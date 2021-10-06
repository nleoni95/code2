#!/bin/bash
#exécution du cleanup via le script cleanup.py

for d in ./*
do
	cd $d
	cp ../cleanup.py .
	python cleanup.py
	rm cleanup.py 
	cd ..
done

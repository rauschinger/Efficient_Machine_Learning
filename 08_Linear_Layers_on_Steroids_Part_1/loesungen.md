# Serie 8 Linear Layers on Steroids Part 1

## Aufgabenstellung

## Blocked Matrix Multiplications

![Alt-Text](https://github.com/rauschinger/Efficient_Machine_Learning/blob/main/08_Linear_Layers_on_Steroids_Part_1/aufgabenstellung.png)


TODO:

MatmulAtenBlocked.cpp Zeile 12

MatmulAtenBlocked.test.cpp Zeile 29

MatmulLibxsmm Zeile 5 (01:15), ab Zeile 81 (Offsets berechnen) 


## Small JITted GEMMs

![Alt-Text](https://github.com/rauschinger/Efficient_Machine_Learning/blob/main/08_Linear_Layers_on_Steroids_Part_1/aufgabenstellung_1.png)


Um die Version ohne OpenMP zu messen wird beim Ausführen im terminal der Befehl OMP_NUM_THREADS=0 gesetzt. Damit wird single core gemessen, da wir dann nur auf einem thread laufen



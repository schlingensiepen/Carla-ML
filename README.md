# CarlaML

Zielordner: ".../CARLA_x.x.xx/PythonAPI/examples"

tensorflow      2.4.1

Cuda Toolkit    11.0

cuDNN           8.0.4


- waypoints.py : Finden von Wegpunkten zum gezielten Spawnen des Fahrzeugs
- testcam.png : Beispielbild der Segmentation Camera
- opencvlanes.py / opencvlanesvideo.py : Zum Testen der Spurlinienerkennung anhand von Straßenfotos / -videos

- produce_data.py (in Bearbeitung): Zum Aufzeichnen der Trainingsdaten auf Basis des Carla-Skripts manual_control.py
- segcamtf.py (in Bearbeitung): Update zu tut2.py, Verwendung der Segmentation Camera


Diese ist ein Teilprojekt eines Studentenprojekt aus dem Sommersemester 2021: \
Erstellen eines selbstständigen Spielers für ein 3D Spiel

Im Rahmen des Projekts im 6. Semester hatten wir die Aufgabe, ein 3D-Spiel von einem künstlichen Spieler steuern zu lassen. Da wir uns das ganze Semester über coronabedingt im Lockdown befanden, fand das Projekt ausschließlich Online statt.

Studenten der Studiengänge Fahrzeugtechnik, Maschinenbau und Wirtschaftsingenieurwesen Dozent: Prof. Dr.-Ing. Jörn Schlingensiepen

Nach einer kurzen Einführung in die Funktionsweisen von Machine Learning, einem Python-Crashkurs und der Erläuterung von GitHub konnten wir schon mit den ersten Spielen starten. Dazu entschieden wir uns zunächst das Chrome Dino-Game, Flappy-Bird und den Klassiker Snake.

Anschließend war es Zeit für die nächste Herausforderung. Dazu entschieden wir uns für einen künstlichen Spieler für den Fahrsimulator “CARLA”, welcher ein Auto fährt, und einen künstlichen Spieler für das Spiel “Hide’n’Seek”, welcher den Hider und den Seeker gleichzeitig steuern kann, zu entwickeln.

Die Dokumentation unseres gesamten Projekts und Links zu den verwendeten Quellcodes, sowie unseren Quelltext haben wir hier zusammengestellt: \
https://github.com/schlingensiepen/AutoPlayer3D

Das Video unseres Abschlusspräsentation findet Ihr hier: \
https://youtu.be/qmr5ONcXWgQ

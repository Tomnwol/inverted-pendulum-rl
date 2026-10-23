# Comparaison des méthodes — retour moyen

À remplir à la main. Plus le nombre est proche de 0, mieux le pendule tient la verticale.

Pour REINFORCE, MC, TD et PPO, recopier le nombre après `éval` (50 épisodes, graine 123).
Pour la recherche aléatoire, recopier `meilleur J` aux mêmes budgets : le record gardé depuis le début, sur 20 épisodes (graine 100).

L'affichage tombe sur les budgets du tableau déjà atteints. Un paramètre `500` arrête l'exécution à 500 et n'affiche que les lignes `50` à `500`.

À itération égale, PPO a déjà fait dix passes sur le lot collecté. REINFORCE, MC et TD n'en font qu'une. Pour le chapitre 3, le budget est un nombre d'essais.

| Itérations | Recherche aléatoire (ch. 3) | REINFORCE (ch. 4) | Monte Carlo (ch. 5) | Différence temporelle (ch. 5) | PPO |
|---|---|---|---|---|---|
| 0 | -1302.8 | -1254.7 | -1254.7 | -1254.7 | -1254.7 |
| 5 | -1294.4 | -1284.9 | -1282.4 | -1400.3 | -1235.7 |
| 10 | -1294.4 | -1337.9 | -1322.6 | -1466.2 | -1225.0 |
| 15 | -1294.4 | -1371.6 | -1324.7 | -1438.3 | -1164.7 |
| 20 | -1294.4 | -1327.9 | -1327.4 | -1419.0 | -889.7 |
| 25 | -1263.1 | -1290.4 | -1325.3 | -1442.6 | -615.7 |
| 30 | -1203.7 | -1262.6 | -1316.8 | -1462.9 | -203.7 (TRAINED) |
| 35 | -1203.7 | -1250.1 | -1289.0 | -1466.3 | -170.8 |
| 40 | -1203.7 | -1229.4 | -1274.3 | -1466.4 | -158.9 |
| 45 | -1203.7 | -1233.9 | -1223.0 | -1466.4 | -156.6 |
| 50 | -1203.7 | -1270.3 | -1248.5 | -1466.4 | -155.1 |
| 100 | -1179.2 | -1234.1 | -1126.4 | -1456.6 | -150.0 |
| 150 | -1043.3 | -1224.3 | -1172.5| -1142.0 | -150.4 |
| 200 | -1043.3 | -1231.6 | -1083.0| -1287.2 | -151.2 |
| 300 | -1043.3 | -1143.0 | -1020.8| -1023.9 | -150.3 |
| 400 | -1038.2 |  -930.6 | -1121.7| -1057.2 | -151.9 |
| 500 | -1037.9 | -1083.3 | -1084.0| -959.7 | -148.8 |
| 1000 | -914.8 | -892.5 | -923.5|  -805.2| -152.4 |
| 2000 | -843.3 | -749.6 | -158.9 (TRAINED)| -805.2 | -150.1 |
| 3000 | -836.4 | -806.4 | -150.7| -805.0 | -147.4 |
| 4000 | -836.4 | -806.1 | -151.7| -804.9| -146.0 |
| 5000 | -792.7 | -1339.3| -147.5| -804.9| -149.3 |
| 10000 | -413.8 | -1287.1| -714.2| -812.8 | -153.9|
| 20000 | -317.4 | -1294.7| NOSIM | -815.7 | NOSIM |
| 30000 | -312.9 | NOSIM | NOSIM | -807.8 | NOSIM |
| 40000 | -307.9 | NOSIM | NOSIM | -809.6| NOSIM |

`train raw` est le même apprentissage que le chapitre 4 (avantage = G). La colonne REINFORCE suffit.

## Commandes

Depuis le dossier `pendule`. Chaque méthode repart de zéro. PPO écrase `pendule_acteur.pt`, la recherche aléatoire écrase `ch3_poids.npy`.

```bash
python ch3_recherche_aleatoire.py 500
python ch4_reinforce.py 500
python ch5_6_baseline_critique.py train mc 0 500
python ch5_6_baseline_critique.py train td 0 500
python ppo.py 500
```

Le `0` des commandes `train` est la graine. Le dernier nombre est le budget d'itérations.
Pour un autre budget, on change ce dernier nombre, par exemple `python ppo.py 200` ou `python ch4_reinforce.py 200`.
Sans argument, le chapitre 4 fait 5000 itérations, les modes `train` font 500, PPO fait 100, et la recherche aléatoire fait 14999 essais.

# Lumen Data Science 2025 - Detekcija Melanoma

Predstavljamo naše rješenje za natjecanje **Lumen Data Science 2025**! 

## Opis Zadatka

Melanom je jedan od najsmrtonosnijih oblika raka kože, ali rana detekcija značajno povećava šanse za uspješno liječenje. Tradicionalni modeli detekcije često su pristrani prema određenim tonovima kože, što može dovesti do nejednakosti u dijagnostici. Naš zadatak je:

- Razviti model strojnog učenja koji precizno detektira melanom na slikama kože.
- Osigurati da model radi jednako dobro na različitim tonovima kože.

## Ciljevi

1. **Točnost**: Postići visoku preciznost u detekciji melanoma.
2. **Pravednost**: Minimizirati pristranost prema boji kože.
3. **Generalizacija**: Osigurati da model dobro radi na raznolikim skupovima podataka.

## Podaci

Podaci su dostupni putem Kagglea. Preuzmite ih koristeći sljedeće upute:
```
mkdir ./data
cd ./data

kaggle datasets download sumaiyabinteshahid/isic-challenge-dataset-2020
unzip -q sumaiyabinteshahid/isic-challenge-dataset-2020
```
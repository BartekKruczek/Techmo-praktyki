# Techmo-praktyki

## To do:
- ~~dodać automatyczną optymalizację hiperparametrów w stylu grid search/ random search/ bayesian optimization~~
- ~~upewnić się na 110% z rozkładem płci i mówców w zbiorach treningowych, walidacyjnych i testowych~~
- ~~dodać więcej języka ang. do zdrowego korpusu~~
- ~~dodac zbieranie danych z treningu do wizualizacji -> tensorboard~~
- ~~kfold validacja -> pomysł porzucony, za duy nakład czasu obliczeniowego~~
- ~~augmentacja danych -> tylko dla zbioru treningowego~~
- ~~histogram długości plików~~
- ~~dodać labele child, female i male~~
- ~~wizualizacja płci~~
- ~~cos sie dzieje z ilosciami plikow, jak dodalem sprawdzanie plci to spadlo z 50k na 30k -> jest ok~~
- ~~zamienic mfcc na mel spectrogram~~
- ~~zaktualizowac sciezki do bazy danych -> nowy korpus ang.~~
- wziąć autoencoder, dotrenować go na naszych danych, a potem użyć go do ekstrakcji cech
- ~~zwiększyć liczbę klas~~
- ~~opcjonalnie zmniejszyć ilość daych - > jedna epoka trwa 1h~~
- ~~podrasować istniejący model konwolucyjny~~
- ~~dodać metryki takie jak F1, precision, recall~~
- aktualizacja wizualizacji z multi klasyfikacji
- ~~dodać ekstrakcje innych cech z sygnalu audio~~
- ~~zbudowac model, ktory by przyjmowac cechy w postaci wektorowej~~
- ~~naprawa wymiarowości wejścia do modelu~~
- ~~połączenie ze sobą cech wektorowych w jedną macierz~~
- naprawić sieć RNN -> nie chce się skutecznie uczyć
- ~~dodac ewaluacje per demografia -> kobieta, mężczyzna, dziecko (chore, zdrowe)~~

## Opis baz danych:
- [Czech_SLI](https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-1597)
- [Mobile device Parkinson](https://zenodo.org/records/2867216#.XeTbN59R2BZ)
- [Torgo dysarthria](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)
- [Saarbruecken Voice Database](https://stimmdatenbank.coli.uni-saarland.de/help_en.php4)

### Opcjonalnie, jak dotychczasowe dane nie wystarczą
- [TalkBank -> do pobrania](https://talkbank.org)
- [MEEI -> do pobrania](https://ocw.mit.edu/courses/6-542j-laboratory-on-the-physiology-acoustics-and-perception-of-speech-fall-2005/pages/lab-database/)

### Linki do błędów:
- [Kompatybilność tfio z czipami armx64](https://github.com/tensorflow/io/issues/1859)
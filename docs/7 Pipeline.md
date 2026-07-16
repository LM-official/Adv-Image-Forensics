# **Pipeline di esecuzione**
Questa relazione descrive, passo passo, l’esecuzione del progetto: dalla clonazione iniziale all’analisi finale dei risultati. La pipeline è composta da sei fasi automatiche eseguite in sequenza; la teoria delle operazioni impiegate è raccolta nell’[approfondimento teorico](5%20Approfondimento%20teorico.md), il modello forense è [TruFor](6%20TruFor.md) ed il [dataset](4%20Dataset.md) riguarda le elezioni europee 2024.

## **Configurazione dell’ambiente:**
I test sono stati svolti su un MacBook Air M5, con macOS Tahoe 26.4.1. La procedura di installazione (descritta anche nel [README](../README.md)) è la seguente:
1. **Clonazione del repository** del progetto. Nella cartella `TruFor/` è inclusa una copia del [repository ufficiale](https://github.com/grip-unina/TruFor/tree/main) di TruFor (GRIP-UNINA), trattata come dipendenza in sola lettura (la pipeline non vi scrive mai) ma con alcune piccole patch già applicate: il caricamento dei pesi con `weights_only=False` (richiesto dalle versioni recenti di PyTorch), il nuovo flag `--max-pixels` (ridimensiona le immagini troppo grandi prima dell’inferenza, vedi [Limitazioni tecniche](#limitazioni-tecniche)) ed il supporto alla scheda video Apple (`mps`) in alternativa a NVIDIA (`cuda`).
2. **Download dei [pesi](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip)** del modello (esclusi dal repository), da posizionare in `TruFor/TruFor_train_test/weights/trufor_ph3/best.pth.tar`.
3. **Creazione dell’ambiente virtuale** Python 3.14 (venv), utile per isolare e gestire le dipendenze software, elencate in [requirements.txt](../requirements.txt).
4. **Dataset** (anch’esso escluso dal repository): il file `crowdtangle.jsonl` e la cartella `crowdtangle_media/` vanno posizionati in `dataset/`.

## **Architettura:**
Le sei fasi sono sei script Python in [scripts/](../scripts/), da eseguire in ordine. [config.py](../scripts/config.py) è l’unica fonte di verità: raccoglie tutti i percorsi ed i parametri regolabili (si modifica quello, non gli script); [utils.py](../scripts/utils.py) contiene le funzioni condivise (CSV, `.npz`, pulizia del testo, etc.).

L’esecuzione ruota attorno ad una tabella, `scripts/temp/filtered_dataset.csv`: quasi ogni fase vi aggiunge le proprie colonne (unite per `filename`), mentre ogni report leggibile viene scritto accanto come file `.txt`/`.png` separato, così dati e report restano distinti.

| Colonne | Aggiunte da | Significato |
| :------ | :---------- | :---------- |
| `id` | `filter.py` | Identificatore del post. |
| `crowdtangle_score` | `filter.py` | Punteggio di engagement. |
| `filename` | `filter.py` | Nome del file (hash dell’URL): la chiave di unione di ogni fase. |
| `trufor_score` | `trufor.py` | Punteggio globale di manipolazione in [0, 1]. |
| 19 colonne (`aspect_ratio`, `brightness`, …) | `features.py` | Le feature visive, una colonna ciascuna. |
| `anomaly_txt` | `describe.py` | Descrizione testuale della regione alterata (didascalia + parole lette). |

TruFor riceve la cartella di input (`temp/images`) e quella di output (`temp/npz`) entrambe sotto `scripts/temp/`: tutti i file intermedi vivono lì, il dataset originale non viene mai toccato.

## **Esecuzione:**
```sh
source .venv/bin/activate
cd scripts

python filter.py
python trufor.py --generate-truth
python features.py
python mask.py
python describe.py
python anomaly_classification.py
```

`--generate-truth` serve solo alla prima esecuzione: genera il template della [ground truth](3%20Definizioni.md#ground-truth) e mette in pausa lo script finché l’utente non lo ha etichettato a mano; le esecuzioni successive riusano il file compilato (se presente).

## **Fase 1 — Filtraggio (filter.py):**
[filter.py](../scripts/filter.py) legge il dataset grezzo e conserva solo i post analizzabili secondo i [criteri di filtraggio](4%20Dataset.md#filtraggio-filterpy) (punteggio presente, soli media fotografici, file rintracciabile su disco tramite l’[hash dell’URL](4%20Dataset.md#nomenclatura-dei-file-multimediali), nessuna collisione tra hash). Produce `filtered_dataset.csv` (una riga per immagine, ordinata per nome file) ed i symlink alle immagini valide in `temp/images/`, la cartella di input di TruFor.

Analizza poi la [distribuzione](5%20Approfondimento%20teorico.md#distribuzione) del punteggio [CrowdTangle](3%20Definizioni.md#crowdtangle): [mediana](3%20Definizioni.md#mediana) e [media](3%20Definizioni.md#media-aritmetica), [asimmetria](3%20Definizioni.md#asimmetria-statistica-skewness), [Tukey fences](3%20Definizioni.md#intervalli-tra-quartili-iqr---tukey-fence) e conteggio degli [outlier](3%20Definizioni.md#outlier); salva il report e la figura con [istogramma](3%20Definizioni.md#istogramma) (troncato alle fences, così gli outlier non schiacciano il grosso della distribuzione in una sola barra) e [box-plot](3%20Definizioni.md#box-plot) sull’intervallo completo.

## **Fase 2 — Inferenza TruFor (trufor.py):**
[trufor.py](../scripts/trufor.py) esegue il `test.py` di [TruFor](6%20TruFor.md) come sottoprocesso sulle immagini filtrate, in batch da 50 immagini con 180 secondi di pausa tra un batch e l’altro per lasciar raffreddare la macchina (regolabili con `--batch-size` e `--wait`). Ogni immagine produce un file `.npz` con la mappa delle anomalie, la mappa di confidenza ed il punteggio globale. L’esecuzione è riavviabile: ogni `.npz` viene salvato appena pronto, ed una nuova esecuzione salta le immagini già analizzate riprendendo da dove si era interrotta. Le immagini oltre i 6 megapixel vengono ridotte prima dell’inferenza (patch `--max-pixels`) e le mappe riportate poi alla risoluzione originale.

Il punteggio globale viene unito alla tabella come `trufor_score`. Seguono tre analisi:
1. **Valutazione del detector**: le previsioni a soglia 0.5 (`--threshold`) vengono incrociate con la [ground truth](3%20Definizioni.md#ground-truth) etichettata a mano (0 = reale, 1 = alterata) nella [matrice di confusione](3%20Definizioni.md#matrice-di-confusione), da cui [accuratezza](3%20Definizioni.md#accuratezza), [precisione](3%20Definizioni.md#precisione), [sensibilità](3%20Definizioni.md#sensibilita) ed [F1](3%20Definizioni.md#f1-score) ([vedi approfondimento](5%20Approfondimento%20teorico.md#valutazione-dei-modelli-di-classificazione)); il report elenca anche i falsi positivi ed i falsi negativi, per l’ispezione manuale.
2. **[Correlazione](5%20Approfondimento%20teorico.md#correlazione)**: [Pearson](3%20Definizioni.md#indice-di-correlazione-pearson-coefficiente-di-correlazione-lineare) e [Spearman](3%20Definizioni.md#indice-di-correlazione-r-per-ranghi-di-spearman), con i p-value, tra `trufor_score` e `crowdtangle_score`.
3. **[Regressione OLS](5%20Approfondimento%20teorico.md#regressione-ols-ordinary-least-squares)**: variabile indipendente X = `trufor_score`, variabile dipendente Y = `crowdtangle_score`.

## **Fase 3 — Feature visive (features.py):**
[features.py](../scripts/features.py) estrae da ogni immagine le [19 feature visive](5%20Approfondimento%20teorico.md#feature-visive) e le unisce alla tabella. Genera poi il grafico a dispersione correlazione-correlazione: ogni feature è un punto con X = [correlazione](3%20Definizioni.md#correlazione) con l’engagement ed Y = correlazione con il punteggio di manipolazione, più una retta di [regressione](3%20Definizioni.md#regressione-lineare) tra i punti e la retta di riferimento y = x. Il metodo di correlazione si sceglie con `--method` (`pearson`, il default, o `spearman`).

## **Fase 4 — Maschere di localizzazione (mask.py):**
[mask.py](../scripts/mask.py) trasforma ogni `.npz` nella maschera binaria della regione alterata: riporta le mappe alle dimensioni dell’immagine ed applica [binarizzazione a doppia soglia, inversione di polarità e pulizia](5%20Approfondimento%20teorico.md#maschere-di-localizzazione) (soglie regolabili con `--threshold` e `--conf-threshold`). Salva un PNG per immagine con lo sfondo nero e la regione alterata nei colori originali: il formato è senza perdita e, essendo l’immagine quasi tutta nera, si comprime molto bene. Una maschera completamente nera è un risultato valido: “nessuna manipolazione confidente qui”.

Se esiste la cartella `masked_truth/` (maschere di riferimento disegnate a mano), calcola la [matrice di confusione](3%20Definizioni.md#matrice-di-confusione) a livello di pixel e la [IoU](3%20Definizioni.md#intersection-over-union-iou), globale e per singola immagine (elencata dalla peggiore, per individuare subito le maschere da ispezionare).

## **Fase 5 — Descrizione (describe.py):**
[describe.py](../scripts/describe.py) carica [BLIP](3%20Definizioni.md#blip-bootstrapping-language-image-pre-training) (captioning, [sezione 7](5%20Approfondimento%20teorico.md#blip-bootstrapping-language-image-pre-training)) ed EasyOCR ([OCR](3%20Definizioni.md#ocr-optical-character-recognition) italiano ed inglese, [sezione 8](5%20Approfondimento%20teorico.md#ocr-optical-character-recognition)) sulla scheda video disponibile (`mps`/`cuda`, altrimenti CPU) e descrive ogni immagine mascherata:
1. le maschere quasi nere (2% o meno di pixel visibili) ricevono l’etichetta fissa `"empty"` senza interpellare i modelli, che su un’immagine vuota allucinerebbero;
2. le altre vengono ritagliate al riquadro non nero: l’OCR legge il testo stampato (con i filtri di qualità) e BLIP genera la didascalia con la [beam search](3%20Definizioni.md#beam-search);
3. la didascalia viene ripulita (solo lettere a-z, accenti convertiti, parole duplicate eliminate) ed unita alle parole OCR nella colonna `anomaly_txt`.

Lavora in batch da 25 immagini con un checkpoint (`description_temp.csv`): un’esecuzione interrotta riprende saltando le immagini già descritte, e la tabella viene aggiornata una sola volta a fine corsa. Il report finale riassume la quota di `empty`, le lunghezze e le parole più frequenti, per un controllo di qualità a colpo d’occhio. Flag utili: `--do-sample` (campionamento al posto della beam search) e `--no-ocr` (solo didascalia).

## **Fase 6 — Classificazione dell’engagement (anomaly_classification.py):**
[anomaly_classification.py](../scripts/anomaly_classification.py) risponde alla domanda finale ([sezione 11](5%20Approfondimento%20teorico.md#classificazione-dellengagement)): vettorizza `anomaly_txt` con il [TF-IDF](3%20Definizioni.md#tf-idf-term-frequency---inverse-document-frequency) ([sezione 9](5%20Approfondimento%20teorico.md#tf-idf-term-frequency---inverse-document-frequency)) e ne costruisce anche la variante compressa con la [PCA](3%20Definizioni.md#pca-principal-component-analysis) ([sezione 10](5%20Approfondimento%20teorico.md#pca-principal-component-analysis)). Scarta i post nella dead zone (|punteggio| ≤ 0.5, regolabile con `--dead-zone`), binarizza l’engagement a 0 e, su entrambe le varianti di feature (più `trufor_score`), addestra la regressione logistica in cross-validation ed il [GLM binomiale](3%20Definizioni.md#glm-generalized-linear-model-binomiale) inferenziale. Il report unico riporta metriche, odds ratio e tabella dei coefficienti.

## **Risultati:**
Tutti gli output finiscono in `scripts/temp/`. I valori ottenuti:

| Analisi | Report | Valori chiave |
| :------ | :----- | :------------ |
| Detector (100 immagini etichettate) | [trufor_confusion_matrix.txt](../results/trufor_confusion_matrix.txt) | [Accuratezza](3%20Definizioni.md#accuratezza) 61%, [precisione](3%20Definizioni.md#precisione) 80.6%, [sensibilità](3%20Definizioni.md#sensibilita) 47.5%, [F1](3%20Definizioni.md#f1-score) 0.60. |
| Maschere (100 maschere di riferimento) | [mask_confusion_matrix.txt](../results/mask_confusion_matrix.txt) | Precisione 78.1%, sensibilità 39.7%, F1 0.53, [IoU](3%20Definizioni.md#intersection-over-union-iou) 0.36. |
| Correlazione (1217 immagini) | [correlation_trufor_crowdtangle.txt](../results/correlation_trufor_crowdtangle.txt) | [Pearson](3%20Definizioni.md#indice-di-correlazione-pearson-coefficiente-di-correlazione-lineare) r = 0.050 (p = 0.08), [Spearman](3%20Definizioni.md#indice-di-correlazione-r-per-ranghi-di-spearman) 0.039 (p = 0.17). |
| Regressione OLS | [regression_trufor_crowdtangle.txt](../results/regression_trufor_crowdtangle.txt) | R² = 0.003: il punteggio TruFor non spiega l’engagement. |
| Descrizioni | [description_report.txt](../results/description_report.txt) | 1211 immagini descritte, di cui 306 `empty`; mediana di 13 parole. |
| Classificazione (1201 post dopo la dead zone) | [anomaly_classification.txt](../results/anomaly_classification.txt) | TF-IDF grezzo: accuratezza 59.3% (baseline di maggioranza 51.3%), [AUC](3%20Definizioni.md#auc-area-under-the-curve) 0.639; TF-IDF + PCA (60 componenti): 58.6%, AUC 0.628. |

L’interpretazione complessiva — l’intensità della manipolazione non predice l’engagement, il suo contenuto porta invece un segnale debole ma reale — è discussa nei [risultati sul dataset delle elezioni](6%20TruFor.md#risultati-sul-dataset-delle-elezioni-europee-2024).

## **Limitazioni tecniche:**
I test hanno evidenziato dei limiti pratici — memoria RAM e spazio su disco proporzionali ai pixel dell’immagine, invisibilità dell’inpainting a diffusione coerente — che derivano tutti dall’architettura di TruFor: sono descritti nelle [limitazioni tecniche riscontrate](6%20TruFor.md#limitazioni-tecniche-riscontrate).

## **Conclusioni:**
La direzione per evolvere l’infrastruttura — un approccio a cascata che affianchi a TruFor un classificatore per le immagini interamente sintetiche — è delineata nei [possibili miglioramenti](6%20TruFor.md#possibili-miglioramenti).
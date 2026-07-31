# **Pipeline di esecuzione**
Questa relazione descrive, passo passo, l’esecuzione del progetto: dalla clonazione iniziale all’analisi finale dei risultati. La pipeline è composta da sei fasi automatiche eseguite in sequenza; la teoria delle operazioni impiegate è raccolta nell’[approfondimento teorico](5%20Approfondimento%20teorico.md), il modello forense è [TruFor](6%20TruFor.md) ed il [dataset](4%20Dataset.md) riguarda le elezioni europee 2024.

## **Configurazione dell’ambiente:**
I test sono stati svolti su un MacBook Air M5, con macOS Tahoe 26.4.1. La procedura di installazione (descritta anche nel [README](../README.md)) è la seguente:
1. **Clonazione del repository** del progetto. Nella cartella `TruFor/` è inclusa una copia del [repository ufficiale](https://github.com/grip-unina/TruFor/tree/main) di TruFor (GRIP-UNINA), trattata come dipendenza in sola lettura (la pipeline non vi scrive mai) ma con alcune piccole patch già applicate: il caricamento dei pesi con `weights_only=False` (richiesto dalle versioni recenti di PyTorch), il nuovo flag `--max-pixels` (ridimensiona le immagini troppo grandi prima dell’inferenza, vedi [Limitazioni tecniche](#limitazioni-tecniche)) ed il supporto alla scheda video Apple (`mps`) in alternativa a NVIDIA (`cuda`).
2. **Download dei [pesi](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip)** del modello (esclusi dal repository), da posizionare in `TruFor/TruFor_train_test/weights/trufor_ph3/best.pth.tar`.
3. **Creazione dell’ambiente virtuale** Python 3.14 (venv), utile per isolare e gestire le dipendenze software, elencate in [requirements.txt](../requirements.txt).
4. **Dataset** (anch’esso escluso dal repository): il file `crowdtangle.jsonl` e la cartella `crowdtangle_media/` vanno posizionati in `dataset/`.

## **Architettura:**
Le sei fasi corrispondono ad altrettanti script Python in [scripts/](../scripts/), da eseguire in ordine; l’ultima (la classificazione) è avviata da [`classification_home.py`](../scripts/classification_home.py), che a sua volta esegue i **quattro script indipendenti** `classification_*.py`. [config.py](../scripts/config.py) è l’unica fonte di verità: raccoglie tutti i percorsi ed i parametri regolabili (si modifica quello, non gli script); [utils.py](../scripts/utils.py) contiene le funzioni condivise (CSV, `.npz`, pulizia del testo, etc.).

L’esecuzione ruota attorno ad una tabella, `scripts/temp/dataset_filtered.csv`: quasi ogni fase vi aggiunge le proprie colonne (unite per `filename`), mentre ogni report leggibile viene scritto accanto come file `.txt`/`.png` separato, così dati e report restano distinti.

| Colonne | Aggiunte da | Significato |
| :------ | :---------- | :---------- |
| `id` | [`filter.py`](../scripts/filter.py) | Identificatore del post. |
| `crowdtangle_score` | [`filter.py`](../scripts/filter.py) | Punteggio di engagement. |
| `filename` | [`filter.py`](../scripts/filter.py) | Nome del file (hash dell’URL): la chiave di unione di ogni fase. |
| `trufor_score` | [`trufor.py`](../scripts/trufor.py) | Punteggio globale di manipolazione in [0, 1]. |
| 19 colonne (`aspect_ratio`, `brightness`, …) | [`features.py`](../scripts/features.py) | Le feature visive, una colonna ciascuna. |
| `anomaly_txt` | [`describe.py`](../scripts/describe.py) | Descrizione testuale della regione alterata (didascalia + parole lette). |

TruFor riceve la cartella di input (`temp/images`) e quella di output (`temp/npz`) entrambe sotto `scripts/temp/`: tutti i file intermedi vivono lì, il dataset originale non viene mai toccato.

## **Esecuzione:**
```sh
source .venv/bin/activate
cd scripts

python filter.py
python trufor.py [--generate-truth]
python features.py
python mask.py
python describe.py
python classification_home.py
```

`--generate-truth` serve solo alla prima esecuzione: genera il template della [ground truth](3%20Definizioni.md#ground-truth) e mette in pausa lo script finché l’utente non lo ha etichettato a mano; le esecuzioni successive riusano il file compilato (se presente).

In alternativa a `classification_home.py` è possibile eseguire individualmente gli script di classificazione (ognuno è indipendente):
```sh
python classification_logistic_regression.py
python classification_glm.py
python classification_lime_shap.py
python classification_visual_text.py
```

## **Fase 1 — Filtraggio (filter.py):**
[filter.py](../scripts/filter.py) legge il dataset grezzo e conserva solo i post analizzabili secondo i [criteri di filtraggio](4%20Dataset.md#filtraggio-filterpy) (punteggio presente, soli media fotografici, file rintracciabile su disco tramite l’[hash dell’URL](4%20Dataset.md#nomenclatura-dei-file-multimediali), nessuna collisione tra hash). Produce `dataset_filtered.csv` (una riga per immagine, ordinata per nome file) ed i symlink alle immagini valide in `temp/images/`, la cartella di input di TruFor.

Analizza poi la [distribuzione](5%20Approfondimento%20teorico.md#distribuzione) del punteggio [CrowdTangle](3%20Definizioni.md#crowdtangle): [mediana](3%20Definizioni.md#mediana) e [media](3%20Definizioni.md#media-aritmetica), [asimmetria](3%20Definizioni.md#asimmetria-statistica-skewness), [Tukey fences](3%20Definizioni.md#intervalli-tra-quartili-iqr---tukey-fence) e conteggio degli [outlier](3%20Definizioni.md#outlier); salva il report e la figura con [istogramma](3%20Definizioni.md#istogramma) (troncato alle fences, così gli outlier non schiacciano il grosso della distribuzione in una sola barra) e [box-plot](3%20Definizioni.md#box-plot) sull’intervallo completo.

## **Fase 2 — Inferenza TruFor (trufor.py):**
[trufor.py](../scripts/trufor.py) esegue il `test.py` di [TruFor](6%20TruFor.md) come sottoprocesso sulle immagini filtrate, in batch da 50 immagini con 180 secondi di pausa tra un batch e l’altro per lasciar raffreddare la macchina (regolabili con `--batch-size` e `--wait`). Ogni immagine produce un file `.npz` con la mappa delle anomalie, la mappa di confidenza ed il punteggio globale. L’esecuzione è riavviabile: ogni `.npz` viene salvato appena pronto, ed una nuova esecuzione salta le immagini già analizzate riprendendo da dove si era interrotta. Le immagini oltre i 6 megapixel vengono ridotte prima dell’inferenza (patch `--max-pixels`) e le mappe riportate poi alla risoluzione originale.

Il punteggio globale viene unito alla tabella come `trufor_score`. Seguono quattro analisi:
1. **Valutazione del detector**: le previsioni a soglia 0.25 (`--threshold`) vengono incrociate con la [ground truth](3%20Definizioni.md#ground-truth) etichettata a mano (0 = reale, 1 = alterata) nella [matrice di confusione](3%20Definizioni.md#matrice-di-confusione), da cui [accuratezza](3%20Definizioni.md#accuratezza), [precisione](3%20Definizioni.md#precisione), [sensibilità](3%20Definizioni.md#sensibilita) ed [F1](3%20Definizioni.md#f1-score) ([vedi approfondimento](5%20Approfondimento%20teorico.md#valutazione-dei-modelli-di-classificazione)). Il report affianca a queste tre letture di contesto: l’[AUC](3%20Definizioni.md#auc-area-under-the-curve), che misura la capacità di separare le classi indipendentemente dalla soglia scelta; la baseline di maggioranza, con cui va confrontata l’accuratezza; ed una scansione delle soglie da 0.20 a 0.50, che mostra come le metriche si spostino al variare del taglio. Elenca infine i falsi positivi ed i falsi negativi, per l’ispezione manuale.
2. **[Correlazione](5%20Approfondimento%20teorico.md#correlazione)**: [Pearson](3%20Definizioni.md#indice-di-correlazione-pearson-coefficiente-di-correlazione-lineare) e [Spearman](3%20Definizioni.md#indice-di-correlazione-r-per-ranghi-di-spearman), con i p-value, tra `trufor_score` e `crowdtangle_score`.
3. **[Regressione OLS](5%20Approfondimento%20teorico.md#regressione-ols-ordinary-least-squares)**: variabile indipendente X = `trufor_score`, variabile dipendente Y = `crowdtangle_score`.
4. **[GLM binomiale](3%20Definizioni.md#glm-generalized-linear-model-binomiale) (logit link)**: la controparte della regressione OLS per un bersaglio binario — scarta la dead zone (|punteggio| ≤ 0.5, `--dead-zone`), binarizza l’engagement (1 = sopra le attese, 0 = sotto) e ne modella la probabilità dal solo `trufor_score`, con pseudo-R² di McFadden e p-value.

## **Fase 3 — Feature visive (features.py):**
[features.py](../scripts/features.py) estrae da ogni immagine le [19 feature visive](5%20Approfondimento%20teorico.md#feature-visive) e le unisce alla tabella. Genera poi il grafico a dispersione correlazione-correlazione: ogni feature è un punto con X = [correlazione](3%20Definizioni.md#correlazione) con l’engagement ed Y = correlazione con il punteggio di manipolazione, più una retta di [regressione](3%20Definizioni.md#regressione-lineare) tra i punti e la retta di riferimento y = x. Il metodo di correlazione si sceglie con `--method` (`pearson`, il default, o `spearman`).

## **Fase 4 — Maschere di localizzazione (mask.py):**
[mask.py](../scripts/mask.py) trasforma ogni `.npz` nella maschera binaria della regione alterata: riporta le mappe alle dimensioni dell’immagine ed applica [binarizzazione a doppia soglia, inversione di polarità e pulizia](5%20Approfondimento%20teorico.md#maschere-di-localizzazione) (soglie regolabili con `--threshold` e `--conf-threshold`). Salva un PNG per immagine con lo sfondo nero e la regione alterata nei colori originali: il formato è senza perdita e, essendo l’immagine quasi tutta nera, si comprime molto bene. Una maschera completamente nera è un risultato valido: “nessuna manipolazione confidente qui”.

Se esiste la cartella `masked_truth/` (maschere di riferimento disegnate a mano) calcola, come confronto facoltativo che la pipeline non richiede, la [matrice di confusione](3%20Definizioni.md#matrice-di-confusione) a livello di pixel e la [IoU](3%20Definizioni.md#intersection-over-union-iou), globale e per singola immagine (elencata dalla peggiore, per individuare subito le maschere da ispezionare). Accanto ad ogni metrica riporta quella della baseline “tieni ogni pixel”: cresce con la quota di ground truth dipinta come manipolata e dice quindi quanto è difficile battere la risposta banale su queste maschere di riferimento.

## **Fase 5 — Descrizione (describe.py):**
[describe.py](../scripts/describe.py) carica [BLIP](3%20Definizioni.md#blip-bootstrapping-language-image-pre-training) (captioning, [sezione 7](5%20Approfondimento%20teorico.md#blip-bootstrapping-language-image-pre-training)) ed EasyOCR ([OCR](3%20Definizioni.md#ocr-optical-character-recognition) italiano ed inglese, [sezione 8](5%20Approfondimento%20teorico.md#ocr-optical-character-recognition)) sulla scheda video disponibile (`mps`/`cuda`, altrimenti CPU) e descrive ogni immagine mascherata:
1. le maschere quasi nere (2% o meno di pixel visibili) ricevono l’etichetta fissa `"nomask"` senza interpellare i modelli, che su un’immagine vuota allucinerebbero;
2. le altre vengono ritagliate al riquadro non nero: l’OCR legge il testo stampato (con i filtri di qualità) e BLIP genera la didascalia con la [beam search](3%20Definizioni.md#beam-search);
3. la didascalia viene ripulita (solo lettere a-z, accenti convertiti, parole duplicate eliminate) ed unita alle parole OCR nella colonna `anomaly_txt`.

Lavora in batch da 30 immagini con un checkpoint (`description_temp.csv`): un’esecuzione interrotta riprende saltando le immagini già descritte, e la tabella viene aggiornata una sola volta a fine corsa. Il report finale riassume la quota di `nomask`, le lunghezze e le parole più frequenti, per un controllo di qualità a colpo d’occhio. Flag utili: `--do-sample` (campionamento al posto della beam search) e `--no-ocr` (solo didascalia).

## **Fase 6 — Classificazione dell’engagement (classification_home.py):**
[classification_home.py](../scripts/classification_home.py) è l’entry point dell’ultima fase e risponde alla domanda finale ([sezione 11](5%20Approfondimento%20teorico.md#classificazione-dellengagement)). Prepara le feature testuali condivise — vettorizza `anomaly_txt` con il [TF-IDF](3%20Definizioni.md#tf-idf-term-frequency---inverse-document-frequency) ([sezione 9](5%20Approfondimento%20teorico.md#tf-idf-term-frequency---inverse-document-frequency)), scartando le parole troppo rare o troppo comuni, le code poco informative della curva di frequenza (`--min-df` e `--max-df`), e ne costruisce la variante compressa con la [PCA](3%20Definizioni.md#pca-principal-component-analysis) ([sezione 10](5%20Approfondimento%20teorico.md#pca-principal-component-analysis)) — scarta i post nella dead zone (|punteggio| ≤ 0.5, regolabile con `--dead-zone`) e binarizza l’engagement a 0.  
Esegue poi, uno dopo l’altro e ciascuno come sottoprocesso **indipendente** (un errore in uno non blocca gli altri), i quattro script di classificazione che scrivono ognuno il proprio report:
1. [classification_logistic_regression.py](../scripts/classification_logistic_regression.py): su entrambe le varianti di feature (più `trufor_score`) addestra la regressione logistica in cross-validation, con metriche out-of-sample ed odds ratio;
2. [classification_glm.py](../scripts/classification_glm.py): il [GLM binomiale](3%20Definizioni.md#glm-generalized-linear-model-binomiale) inferenziale sulle stesse feature ([sezione 12](5%20Approfondimento%20teorico.md#glm-generalized-linear-model-binomiale)), con la tabella dei coefficienti e p-value;
3. [classification_lime_shap.py](../scripts/classification_lime_shap.py): l’interpretazione [post-hoc](3%20Definizioni.md#post-hoc) a livello di parola ([sezione 13](5%20Approfondimento%20teorico.md#interpretazione-post-hoc-shap-e-lime));
4. [classification_visual_text.py](../scripts/classification_visual_text.py): il confronto tra blocchi di predittori, cioè quanto aggiungono all’engagement le [feature visive](5%20Approfondimento%20teorico.md#feature-visive) rispetto al testo.

Sulla matrice TF-IDF grezza (ampia e sparsa) il GLM semplice potrebbe non convergere — separazione quasi completa: i coefficienti divergono e p-value e standard error diventano privi di senso — perciò `classification_glm.py` interviene con una regolarizzazione elastic-net (`--alpha` per l’intensità della penalità, `--l1-wt` per il tipo: 0 = ridge, 1 = lasso, valori intermedi = elastic net), che restringe i coefficienti verso lo zero rendendoli stabili e leggibili, al prezzo dei p-value. Sulle feature compatte della PCA, invece, il GLM converge più facilemente e la tabella inferenziale resta pienamente valida.

Il flag `--grid-search` di `classification_glm.py` esegue una ricerca sistematica delle due strategie contro la mancata convergenza — il condizionamento del TF-IDF (`min_df` e `max_df`) e la regolarizzazione (`alpha` e `L1_wt`) — provando ogni combinazione e valutandola con il pseudo-R² di McFadden in cross-validation (fuori campione, così vocabolari di dimensioni diverse restano confrontabili). La tabella ordinata (`classification_glm_grid_search.txt`) indica la combinazione migliore e segnala quali conservano la piena inferenza, cioè dove il GLM semplice converge ed i p-value sono disponibili.

Poiché sul TF-IDF grezzo il GLM non fornisce p-value per singola parola, `classification_lime_shap.py` spiega il modello logistico dall’esterno, post-hoc ([sezione 13](5%20Approfondimento%20teorico.md#interpretazione-post-hoc-shap-e-lime)): la `permutation importance` e `SHAP` producono un ranking globale delle parole che alzano o abbassano l’engagement, mentre `LIME` spiega alcuni singoli post. Aggira così il problema di convergenza e restituisce l’interpretazione a livello di parola.

Le [feature visive](5%20Approfondimento%20teorico.md#feature-visive) estratte nella fase 3 restavano finora fuori da ogni modello: `classification_visual_text.py` le mette alla prova con una catena di quattro **GLM binomiali annidati** (ognuno aggiunge colonne al precedente) — `trufor_score` da solo, più il testo, più le feature visive, ed infine tutti insieme. Restare nella stessa famiglia di modelli e sugli stessi post è ciò che rende i pseudo-R² direttamente confrontabili tra blocchi. Le 19 feature vengono standardizzate (media 0, deviazione standard 1) perché vivono su scale molto diverse — `sharpness` vale migliaia, `brightness` sta fra 0 ed 1 — ed i loro coefficienti non sarebbero altrimenti paragonabili tra loro; dopo la standardizzazione ognuno si legge come “l’effetto di una deviazione standard di quella feature”. Il testo entra come componenti PCA e non come TF-IDF grezzo: è la variante su cui il GLM semplice converge più facilmente, e quindi che fornisce i p-value necessari al confronto.

I due blocchi misurano cose diverse: le feature visive descrivono l’**immagine intera** (quanto è ben fatta: contrasto, simmetria, regola dei terzi), mentre `anomaly_txt` descrive la **sola regione alterata** (che cosa è stato messo nell’immagine). Il confronto separa dunque il “come è fatta l’immagine” dal “che cosa contiene l’alterazione”.

Poiché il pseudo-R² in-sample cresce meccanicamente ad ogni colonna aggiunta — anche quando le colonne sono rumore — ogni blocco riporta anche il pseudo-R² in cross-validation (fuori campione) ed un test del rapporto di verosimiglianza (likelihood-ratio) contro il blocco in cui è annidato: la differenza fra le devianze dei due modelli viene confrontata con una distribuzione [chi-quadrato](3%20Definizioni.md#chi-quadrato) con tanti gradi di libertà quante sono le colonne aggiunte, ottenendo il p-value dell’ipotesi “queste colonne non aggiungono nulla”. Le due misure possono divergere, ed è esattamente ciò che accade (vedi [Risultati](#risultati)).

## **Risultati:**
Tutti gli output finiscono in `scripts/temp/`. I valori ottenuti:

| Analisi | Report | Valori chiave |
| :------ | :----- | :------------ |
| Detector (100 immagini etichettate) | [trufor_confusion_matrix.txt](../results/trufor_confusion_matrix.txt) | [AUC](3%20Definizioni.md#auc-area-under-the-curve) 0.754, indipendente dalla soglia. Alla soglia predefinita 0.25: [accuratezza](3%20Definizioni.md#accuratezza) 75% (baseline di maggioranza 62%), [precisione](3%20Definizioni.md#precisione) 78.5%, [sensibilità](3%20Definizioni.md#sensibilita) 82.3%, [F1](3%20Definizioni.md#f1-score) 0.80. |
| Maschere (43 maschere di riferimento) | [mask_confusion_matrix.txt](../results/mask_confusion_matrix.txt) | Precisione 87.8%, sensibilità 48.0%, F1 0.62, [IoU](3%20Definizioni.md#intersection-over-union-iou) 0.45. La precisione molto sopra la sensibilità dice che le regioni predette cadono quasi tutte dentro quelle disegnate a mano, ma sono più strette: differenza di granularità, non errore di localizzazione. Su questa ground truth (66.5% di pixel manipolati) la baseline “tieni ogni pixel” arriva però a F1 0.80 e IoU 0.66: sulle metriche di copertura le maschere non la battono. |
| Correlazione (1217 immagini) | [trufor_crowdtangle_correlation.txt](../results/trufor_crowdtangle_correlation.txt) | [Pearson](3%20Definizioni.md#indice-di-correlazione-pearson-coefficiente-di-correlazione-lineare) r = 0.051 (p = 0.08), [Spearman](3%20Definizioni.md#indice-di-correlazione-r-per-ranghi-di-spearman) 0.041 (p = 0.15). |
| Regressione OLS | [trufor_crowdtangle_regression.txt](../results/trufor_crowdtangle_regression.txt) | R² = 0.003: il punteggio TruFor non spiega l’engagement. |
| GLM binomiale (1201 post dopo la dead zone) | [trufor_crowdtangle_glm.txt](../results/trufor_crowdtangle_glm.txt) | Pseudo-R² di McFadden 0.0009, coefficiente `trufor_score` non significativo (p = 0.22): nemmeno sul bersaglio binario l’intensità predice l’engagement. |
| Descrizioni | [description_report.txt](../results/description_report.txt) | 1211 immagini descritte, di cui 313 con maschera vuota (`nomask`); mediana di 13 parole. |
| Classificazione (1201 post dopo la dead zone) | [classification_logistic_regression.txt](../results/classification_logistic_regression.txt) | TF-IDF grezzo: accuratezza 59.9% (baseline di maggioranza 51.3%), [AUC](3%20Definizioni.md#auc-area-under-the-curve) 0.642; TF-IDF + PCA (61 componenti): 59.1%, AUC 0.632. |
| GLM inferenziale della classificazione (1201 post) | [classification_glm.txt](../results/classification_glm.txt) | In-sample (non cross-validated): sul TF-IDF grezzo fallback ridge senza p-value, pseudo-R² di McFadden 0.136; su TF-IDF + PCA (61 componenti) il GLM converge con p-value validi, pseudo-R² 0.131 (varie componenti significative; `trufor_score` p = 0.06). |
| Interpretazione [post-hoc](3%20Definizioni.md#post-hoc) (1201 post) | [classification_explanations.txt](../results/classification_explanations.txt) | Parole che spingono l’engagement in alto: suit, woman, air, partito, black; in basso: ore, scrivi, microphone, silhouette, giugno. |
| GLM annidato testo + feature visive (1201 post) | [classification_visual_text.txt](../results/classification_visual_text.txt) | Pseudo-R² in-sample: `trufor_score` 0.0009 → + testo 0.131 → + visive 0.063 → tutti insieme 0.156. In cross-validation, però, il modello completo (0.032) non batte il solo testo (0.039). Feature visive che restano significative dopo la [correzione per test multipli](5%20Approfondimento%20teorico.md#correzione-per-test-multipli-bh-fdr) sulle 19 verifiche (per deviazione standard): `aspect_ratio` −0.22 (q = 0.0215) e `warmth` +0.22 (q = 0.0440). |

L’interpretazione complessiva: l’intensità della manipolazione non predice l’engagement, mentre il suo contenuto porta un segnale debole ma reale.

Il confronto tra blocchi aggiunge una sfumatura. Le feature visive dell’immagine intera portano anch’esse un segnale reale — il test del rapporto di verosimiglianza contro il solo `trufor_score` è nettissimo (104.1 su 19 gradi di libertà, p < 0.0001) — e da sole raggiungono quasi il testo (0.034 contro 0.039 in cross-validation). Unirle al testo, però, migliora solo il fit in-sample: fuori campione il modello completo peggiora (0.032). Il test le dichiara significative anche in questa posizione (41.2 su 19 gradi di libertà, p = 0.0022), ma le due misure rispondono a domande diverse — “è più di quanto darebbe il caso?” contro “funziona su post mai visti?” — ed il calo in cross-validation indica che le 19 colonne stanno modellando struttura specifica di questi 1201 post. La lettura onesta è che il “come è fatta l’immagine” ed il “che cosa contiene l’alterazione” portano informazione in buona parte sovrapposta, non cumulativa.

## **Limitazioni tecniche:**
I test hanno evidenziato dei limiti pratici — memoria RAM e spazio su disco proporzionali ai pixel dell’immagine, invisibilità dell’inpainting a diffusione coerente — che derivano tutti dall’architettura di TruFor: sono descritti nelle [limitazioni tecniche riscontrate](6%20TruFor.md#limitazioni-tecniche-riscontrate).

## **Conclusioni:**
La direzione per evolvere l’infrastruttura — un approccio a cascata che affianchi a TruFor un classificatore per le immagini interamente sintetiche — è delineata nei [possibili miglioramenti](6%20TruFor.md#possibili-miglioramenti).
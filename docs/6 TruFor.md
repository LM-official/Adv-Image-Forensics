# **Trufor, [paper](6%20Paper%20TruFor.pdf)**

## **Obiettivo dello studio:**
[TruFor](https://grip-unina.github.io/TruFor/) (Guillaro et al., CVPR 2023) è un [framework](3%20Definizioni.md#framework) forense progettato per rilevare e localizzare un’ampia gamma di manipolazioni nelle immagini. Risolve un problema comune nei rilevatori attuali: la mancanza di generalizzazione e robustezza. TruFor è in grado di identificare sia i fotoritocchi ([cheapfake](3%20Definizioni.md#cheapfake) come copia e incolla), sia le manipolazioni iper-realistiche generate dalle AI.

Il paper parte da una critica alla letteratura: la maggior parte dei metodi si concentra sulla localizzazione e tratta il rilevamento come un sottoprodotto (una statistica globale ricavata dalla mappa di localizzazione), producendo troppi falsi allarmi per un uso su larga scala, dove le immagini manipolate sono rare. TruFor ribalta l’ordine: prima stabilisce se l’immagine è manipolata, con un modulo di rilevamento esplicito, e solo sulle immagini segnalate approfondisce dove si trova la manipolazione.

## **L’innovazione tecnologica: RGB + Noiseprint++:**
TruFor non si limita ad analizzare i pixel, ma unisce due tipologie di indizi tramite una rete neurale basata su [Transformer](3%20Definizioni.md#transformer-deep-learning):
1. Indizi ad alto livello (RGB): L’immagine RGB visibile.  
2. Indizi a basso livello (Noiseprint++): Noiseprint++ è un’innovazione importante del paper. È un’impronta digitale del rumore addestrata in modo auto-supervisionato (self-supervised) solo su immagini reali. Cattura sia le imperfezioni della fotocamera, sia impara a riconoscere lo storico delle modifiche. Tutto ciò rende Noiseprint++ resistente ad operazioni come compressione o altre alterazioni causate dai social network.

L’addestramento di Noiseprint++ usa l’apprendimento contrastivo su piccole patch (64×64 pixel) estratte da circa 24700 immagini reali scattate da 1475 modelli di fotocamera: due patch devono produrre la stessa impronta solo se provengono dallo stesso modello di fotocamera, dalla stessa posizione nell’immagine e hanno subito la stessa storia di modifiche (editing history, simulata durante l’addestramento con 512 diverse combinazioni di ridimensionamenti, compressioni e cambi di illuminazione). Una manipolazione locale rompe questa regolarità, diventando un’anomalia rilevabile.

## **L’architettura e le fasi di addestramento:**
L’immagine RGB e la sua Noiseprint++ scorrono in due rami paralleli con un encoder condiviso; le informazioni delle due modalità si correggono a vicenda (fusione cross-modale) e alimentano due decoder gemelli, che generano la mappa delle anomalie e la mappa di confidenza. Un pooling pesato dalla confidenza riassume le mappe in poche statistiche globali, dalle quali un classificatore calcola il punteggio di integrità. L’addestramento avviene in tre fasi separate:
1. l’estrattore Noiseprint++ (contrastivo, solo immagini reali);  
2. l’encoder ed il decoder delle anomalie, addestrati come segmentazione binaria supervisionata su dataset di immagini vere e manipolate con [ground truth](3%20Definizioni.md#ground-truth) a livello di pixel;  
3. il decoder della confidenza ed il rilevatore finale: la confidenza impara a prevedere dove la mappa delle anomalie sbaglia (true class probability), così le zone di errore note vengono scartate dalla decisione globale.

## **Gli output del modello:**
TruFor non si limita a una risposta vero/falso sull’immagine analizzata, fornisce:
1. Mappa delle anomalie: Una mappa di calore che evidenzia le porzioni dell’immagine che sono state manipolate, cercando deviazioni dal pattern statistico regolare.  
2. Mappa di confidenza: La mappa delle anomalie potrebbe segnalare delle false manipolazioni (aree reali segnalate come false). La mappa di confidenza evidenzia le aree in cui le predizioni potrebbero essere errate, permettendo di ignorare i falsi positivi, riducendo i falsi allarme.  
3. Punteggio di integrità: È un punteggio globale che definisce nel complesso l’autenticità/falsità di un’immagine. È calcolato prendendo informazioni dalla mappa delle anomalie, ma fornendo peso solo alle zone considerate affidabili dalla mappa di confidenza.

## **Prestazioni e risultati:**
TruFor è stato testato su diversi dataset, mostrando di superare i metodi allo stato dell’arte sia nel rilevamento che nella localizzazione di falsi. Ha eccellenti capacità di generalizzare, evidenziando alterazioni localizzate create con [modelli a diffusione](3%20Definizioni.md#modelli-a-diffusione) o [reti GAN](3%20Definizioni.md#reti-generative-adversarial-network-gan). Grazie a Noiseprint++, il framework mantiene alte prestazioni anche in caso di immagini degradate, compresse e persino estratte da social network (che di default applicano compressioni e perdono dei metadati).

In numeri: [F1](3%20Definizioni.md#f1-score) medio di localizzazione a livello di pixel 0.785 contro lo 0.709 del migliore concorrente (CAT-Net v2), [AUC](3%20Definizioni.md#auc-area-under-the-curve) media di rilevamento a livello di immagine 0.857, e circa l’80% di [accuratezza](3%20Definizioni.md#accuratezza) anche senza calibrare la soglia di decisione — condizione in cui molti concorrenti scendono vicino al caso (50%).

## **Limitazioni degli autori:**
Gli stessi autori dichiarano due limiti del framework:
1. TruFor non rileva le immagini interamente generate: cerca regioni che deviano dal pattern regolare del resto dell’immagine, quindi ha bisogno di una maggioranza autentica come riferimento. È un limite rilevante per questo progetto — un post con una grafica interamente sintetica può passare inosservato.
2. L’addestramento richiede la supervisione completa a livello di pixel ed avviene in fasi separate; gli autori indicano come lavoro futuro un addestramento end-to-end con sole etichette a livello di immagine.

## **Limitazioni tecniche riscontrate nel mio utilizzo:**
Oltre ai limiti dichiarati dagli autori, i test di questo progetto (vedi [pipeline](7%20Pipeline.md)) hanno evidenziato dei limiti pratici derivanti dall’architettura di reti neurali del framework:
1. **Memoria RAM proporzionale ai pixel**: TruFor elabora l’immagine a piena risoluzione e la memoria richiesta cresce rapidamente con il numero di pixel. Le immagini a bassa risoluzione vengono processate efficacemente (e rapidamente), ma un’immagine da smartphone di qualità (~5000×3000 pixel) avrebbe richiesto circa 51 GiB di RAM. Mitigazioni adottate dalla pipeline: la patch `--max-pixels` riduce le immagini oltre i 6 megapixel prima dell’inferenza (le mappe vengono poi riportate alla risoluzione originale) ed i batch con pausa evitano la saturazione della macchina.
2. **Spazio su disco proporzionale ai pixel**: ogni `.npz` contiene le mappe per pixel in virgola mobile, quindi la sua dimensione dipende dalla risoluzione dell’immagine e non dal peso del file compresso di partenza: un JPEG da 58 KB ha prodotto un output di 5.8 MB, uno da 900 KB circa 25 MB.
3. **Inpainting a diffusione coerente**: un limite concettuale affine a quelli della sezione [Limitazioni degli autori](#limitazioni-degli-autori). Esempio, chiedendo ad uno strumento di inpainting basato sui [modelli a diffusione](3%20Definizioni.md#modelli-a-diffusione) (come Nano Banana) di aggiungere un cappello ad una persona:
   1. lo strumento crea una maschera, individuando l’area dove apparirà il cappello ed eliminando quei pixel; i pixel fuori dall’area selezionata restano inalterati;
   2. il buco viene colmato con un processo di diffusione: partendo da pieno rumore, cicli successivi di denoising formano il cappello;
   3. (ecco perché TruFor non rileva) nel riempire il buco lo strumento legge e rispetta i pixel originali adiacenti: tiene conto della luce e delle ombre ed analizza il rumore della foto (pelle del volto, sfondo, ecc.), sintetizzando un rumore artificiale coerente con il resto dell’immagine. L’impronta che TruFor cerca — l’incoerenza del rumore — viene così imitata, e la manipolazione può sfuggire.

## **Possibili miglioramenti:**
TruFor dimostra sensibilità alle variazioni del rumore fotografico ed alle classiche tecniche di fotoritocco, ma non è pensato per le immagini interamente sintetiche. Per massimizzare le performance dell’infrastruttura si raccomanda un approccio a cascata:
1. impiegare preliminarmente un classificatore, esterno a TruFor, per scartare le immagini interamente sintetiche;
2. usare, solo successivamente, TruFor per il compito di analizzare e localizzare le eventuali manomissioni locali.
# **Definizioni**

## **A/B testing (spit testing, bucket testing):**
È una pratica del digital marketing utilizzata per confrontare le performance di due diverse versioni di un contenuto. L’obiettivo è capire quale versione susciti maggiore interesse tra i visitatori. Il traffico viene diviso tra le due versioni, e poi vengono monitorate le azioni degli utenti.  
*Fonte: [Oracle](https://www.oracle.com/it/cx/marketing/what-is-ab-testing)*.

## **Accuratezza:**
Misura la proporzione di previsioni corrette sul totale (veri positivi \+ veri negativi) / totale. Se il dataset è sbilanciato un’alta accuratezza può risultare inutile.  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/confusion-matrix)*.

## **Algoritmo di Canny:**
È un operatore per il riconoscimento dei contorni con un approccio multi-stadio:
1. l’immagine viene sfocata con un filtro gaussiano per attenuare il rumore.
2. si calcola il gradiente di luminosità (intensità e direzione dei cambi di luminosità).
3. si tengono solo i massimi locali del gradiente (i punti più forti di ogni contorno).
4. una sogliatura con [isteresi](#isteresi) a doppia soglia accetta i punti sopra la soglia alta, scarta quelli sotto la bassa, e tiene quelli intermedi solo se contigui a punti già accettati. Bilancia il riconoscimento accurato dei contorni con la minimizzazione dei falsi positivi.

*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Algoritmo_di_Canny)*.

## **Analisi esplorativa (EDA):**
Analizzare set di dati e riassumerne le caratteristiche principali. Spesso si impiegano metodi di visualizzazione.  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/exploratory-data-analysis)*.

## **Asimmetria statistica (skewness):**
Un indice di asimmetria di una [distribuzione](#distribuzione-statistica) è un valore che cerca di fornire una misura della sua mancanza di [simmetria](#simmetria-statistica).  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Simmetria_\(statistica\))*.

## **AUC (Area Under the Curve):**
Rappresenta la probabilità che il modello, se fornito un esempio positivo e negativo, classifichi l'esempio positivo più alto dell'esempio negativo.  
*Fonte: [Google](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=it)*.

## **Beam search:**
È un algoritmo di ricerca basato su euristiche che esplora un grafo espandendo, ad ogni livello, solo un numero predefinito β (beam width, l'ampiezza del fascio) dei nodi più promettenti. È una specializzazione greedy della best-first search che riduce drasticamente i requisiti di memoria, al prezzo di non garantire né la completezza né l'ottimalità della soluzione.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Beam_search)*.

## **Bimodale:**
Rappresentazione grafica, in cui la maggior frequenza dei casi si addensa in corrispondenza di due misure del carattere in base al quale i dati sono stati raggruppati.  
*Fonte: [Treccani](https://www.treccani.it/vocabolario/bimodale/)*.

## **BLIP (Bootstrapping Language-Image Pre-training):**
È un VLP (Vision-Language Pretraining) [framework](#framework) progettato sia per attività di comprensione che di generazione.  
*Fonte: [Hugging Face](https://huggingface.co/docs/transformers/model_doc/blip)*.

## **Box-plot:**
Il box-plot è un grafico caratterizzato da tre elementi:
- Una linea o punto che indicano la posizione del valore centrale della distribuzione.  
- Un rettangolo (box) la cui altezza indica la variabilità dei valori prossimi alla media.  
- Due segmenti che partono dal rettangolo e i cui estremi sono determinati in base ai valori estremi della distribuzione.  

*Fonte: [Unina](https://www.docenti.unina.it/webdocenti-be/allegati/materiale-didattico/34471657)*.

## **Carattere statistico:**
Un insieme di caratteristiche rilevate su una o più [unità statistiche](#unita-statistica) appartenenti ad una [popolazione](#popolazione-statistica).  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Variabile_\(statistica\))*.

## **Cheapfake:**
Manipolazione multimediale che richiede un basso livello di sofisticazione tecnica, esempio senza l’uso di intelligenza artificiale, ma solo con software di modifica foto/video/audio.  
*Fonte: [Università degli studi di Torino](https://iris.unito.it/retrieve/c65c505b-470d-4c0f-a424-74e142e25f88/SELA-D-23-00161_R1.pdf)*.

## **Classificazione statistica:**
È lo strumento per organizzare le [unità statistiche](#unita-statistica) di una [popolazione](#popolazione-statistica) in gruppi distinti.  
*Fonte: [Istat](https://www.istat.it/classificazioni-e-strumenti/classificazioni)*.

## **CLIP (Contrastive Language-Image Pre-training):**
È una rete neurale progettata per apprendere concetti visivi direttamente dalla supervisione del linguaggio naturale. Impara associando grandi quantità di immagini ai relativi testi (didascalie) già disponibili su Internet.  
*Fonte: [OpenAI](https://openai.com/index/clip)*.

## **Colorfulness di Hasler-Susstrunk:**
La colorfulness è l’attributo della percezione visiva secondo cui il colore di un’area appare più o meno cromatico. La metrica di Hasler-Susstrunk la quantifica in un’immagine usando due canali di opposizione cromatica, più vicini alla percezione umana dei canali RGB: rosso-verde (R - G) e giallo-blu (media di R e G, meno B). Il punteggio combina quanto i colori variano su questi canali (deviazione standard) e quanto si allontanano in media dal grigio neutro: vale 0 per un’immagine in scala di grigi e cresce con colori più saturi e più vari.  
*Fonte: [Hasler & Süsstrunk, 2003](https://doi.org/10.1117/12.477378)*.

## **Compressione dati lossy:**
Un algoritmo di compressione dati che porta alla perdita di parte dell’informazione originale durante la fase di compressione/decompressione. In generale tendono a scartare informazioni poco rilevanti.  
Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Compressione_dati_lossy).

## **Contrasto RMS (Root Mean Square)**
Misura basata sulla deviazione standard delle intensità dei pixel di un'immagine.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Contrasto)*.

## **Correlazione:**
Relazione tra due variabili tale che a ciascun valore della prima corrisponda un valore della seconda, seguendo una certa regolarità. La correlazione non dipende da un rapporto di causa-effetto quanto dalla tendenza di una variabile a cambiare in funzione di un’altra.  
*Fonte: [WIkipedia](https://it.wikipedia.org/wiki/Correlazione_\(statistica\))*.

## **CrowdTangle:**
Era uno strumento di analisi pubblica di Meta per esplorare i contenuti pubblici sui social media. A partire dal 14 agosto 2024 non è più disponibile.  
*Fonte: [Meta](https://transparency.meta.com/researchtools/other-data-catalogue/crowdtangle)*.

## **Curtosi:**
La pesantezza delle [code di una distribuzione](#distribuzione-di-coda) in relazione al suo addensamento intorno al valore centrale.  
*Fonte: [Treccani](https://www.treccani.it/enciclopedia/curtosi_(Dizionario-di-Economia-e-Finanza))*.

## **Deepfake:**
Contenuto multimediale in cui l’aspetto e/o la voce di una persona vengono ricreati in modo realistico utilizzando l’intelligenza artificiale, nello specifico il [deep learning](#deep-learning).  
*Fonte: [Malwarebytes](https://www.malwarebytes.com/it/cybersecurity/basics/deepfakes)*.

## **Deep Learning:**
È un sottoinsieme del [Machine Learning (ML)](#machine-learning). Si basa su algoritmi di reti neurali artificiali modellati per funzionare ed apprendere in modo simile al cervello umano, sfruttando grandissime moli di dati. Le reti di deep learning sono composte da moltissimi livelli interconnessi di nodi, permettendogli di elaborare informazioni complesse.  
*Fonte: [Oracle](https://www.oracle.com/it/artificial-intelligence/machine-learning/what-is-deep-learning)*.

## **Distribuzione a legge di potenza:**
Ha come particolarità l'assenza di una scala caratteristica dei fenomeni.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Legge_di_potenza)*.

## **Distribuzione binomiale:**
È una distribuzione di probabilità discreta che descrive il numero di successi di un processo di Bernulli.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Distribuzione_binomiale)*.

## **Distribuzione di coda:**
La coda di una distribuzione è la parte che rappresenta i valori più bassi e più alti di un set di dati.  
*Fonte: [Learn Statistics Easily](https://it.statisticseasily.com/glossario/cos%27è-la-distribuzione-della-coda-spiegata-in-dettaglio)*.

## **Distribuzione statistica:**
È una rappresentazione del modo in cui le diverse [modalità](#modalita-statistica) di un [carattere](#carattere-statistico) si distribuiscono nelle [unità statistiche](#unita-statistica) che compongono il collettivo oggetto di studio.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Distribuzione_\(statistica\))*.

## **Entropia:**
Misura della quantità di informazione contenuta in un messaggio: l’informazione media guadagnata rivelando l’esito di un evento incerto. Più una sorgente è imprevedibile, più il suo esito è informativo; una sorgente perfettamente prevedibile ha entropia 0.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Entropia_\(teoria_dell%27informazione\))*.

## **Entropia di Shannon:**
[Entropia](#entropia) applicata ai livelli di grigio di un’immagine, misura quanta informazione contiene: bassa per aree piatte ed uniformi, alta per immagini ricche di variazioni. È calcolata come S = -sum(pk * log(pk)), dove pk è la frequenza dei pixel di valore k.  
*Fonte: [scikit-image](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.shannon_entropy)*.

## **Explainable AI (XAI):**
È l’insieme di processi e metodi che consentono agli umani di comprendere e considerare affidabili i risultati generati da algoritmi di machine learning.  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/explainable-ai)*.

## **F1-score:**
È la media armonica tra [precisione](#precisione) e [sensibilità](#sensibilita). Frequentemente avviene un bilanciamento, migliorando l’una peggiora l’altra. L’F1-Score (2 \* ((precisione \* sensibilità) / (precisione \+ sensibilità))) offre una metrica equilibrata ed è l’indicatore più affidabile, anche per dataset sbilanciati.  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/confusion-matrix)*.

## **Framework:**
Nella produzione di software, un framework è una struttura di supporto su cui un altro software può essere progettato. Il framework fornisce un punto di partenza indipendente per permettere agli sviluppatori di usare porzioni di codice comuni e risparmiare tempo, evitando la riscrittura del medesimo codice.  
*Fonte: [Università degli studi di Camerino](https://computerscience.unicam.it/marcantoni/tesi/POLCAT%20-%20UN%20FRAMEWORK%20PER%20INDAGINE%20FORENSE.pdf)*.

## **Frequenza assoluta:**
Dato un carattere la frequenza è il numero delle [unità statistiche](#unita-statistica) su cui una sua [modalità](#modalita-statistica) si presenta.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Frequenza_\(statistica\))*.

## **Frequenza cumulata (assoluta):**
Associata ad una [modalità](#modalita-statistica), è pari alla somma della sua [frequenza assoluta](#frequenza-assoluta) e di quelle delle [modalità](#modalita-statistica) che la precedono.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Frequenza_cumulata)*.

## **GLM (Generalized Linear Model) binomiale:**
Sono una generalizzazione dei [modelli lineari](#modelli-lineari) nell'ambito della [regressione lineare](#regressione-lineare). Nel modello lineare si ipotizza che la variabile endogena sia distribuita in modo normale, qui può essere distribuita come una qualsiasi variabile casuale della famiglia esponenziale, esempio [binomiale](3%20Definizioni.md#distribuzione-binomiale).  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Modello_lineare_generalizzato)*.

## **Gradiente di Sobel:**
L'operatore di Sobel viene usato nell'elaborazione delle immagini digitali, ad esempio per il riconoscimento dei contorni. Calcola un'approssimazione discreta del gradiente della luminosità: l'immagine viene convoluta con due filtri 3×3, uno per la derivata orizzontale ed uno per quella verticale. In ogni punto si ottiene il vettore gradiente (o la sua norma), che indica intensità e direzione della massima variazione di luminosità: le zone ad alto gradiente corrispondono ai probabili contorni, mentre un gradiente basso indica aree piatte ed uniformi.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Operatore_di_Sobel)*.

## **Ground truth:**
Dati verificati ed utilizzati per addestrare, convalidare e testare i modelli di intelligenza artificiale.  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/ground-truth)*.

## **Image Forgery Localization (IFL):**
Task di indagine forense digitale che ha l’obiettivo di segmentare le regioni alterate all’interno di un’immagine. Individua, isola e mappa quali aree dell’immagine sono oggetto di manipolazione.  
*Fonte: [Arxiv](https://arxiv.org/abs/2412.11214), Università di Cornell*.

## **Indice di correlazione Pearson (coefficiente di correlazione lineare):**
Date due variabili, esprime l’eventuale linearità tra esse.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Indice_di_correlazione_di_Pearson)*.

## **Indice di correlazione R per ranghi di Spearman:**
Non misura una relazione lineare, misura il grado di relazione tra due variabili ordinabili, se possibili continue. Esso permette di stabilire quanto bene una relazione tra due variabili può essere descritta usando una funzione monotona.  
Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Coefficiente_di_correlazione_per_ranghi_di_Spearman).

## **Intersection over Union (IoU):**
È una metrica di valutazione in computer vision per compiti come il rilevamento di oggetti, segmentazione e tracciamento. Serve a valutare quanto la previsione svolta da un modello sia accurata.  
*Fonte: [Università di Stanford](https://giou.stanford.edu)*.

## **Intervalli tra quartili (IQR) \- Tukey fence:**
È un metodo affidabile per segnare gli [outlier](#outlier). È l’intervallo tra il primo e quarto [quartile](#quartili); un valore è considerato anomalo se si allontana di più di 1,5 \* IQR verso l’alto e verso il basso.  
*Fonte: [Oracle](https://docs.oracle.com/cloud/help/it/pbcs_common/PFUSU/insights_metrics_IQR.htm#PFUSU-GUID-CF37CAEA-730B-4346-801E-64612719FF6B)*.

## **Isteresi:**
Fenomeno per cui il valore dipende non solo dai valori attuali, ma anche da quelli precedenti. Si conserva una “memoria” del passato, per cui l’effetto è in ritardo rispetto alla causa.   
*Fonte: [Treccani](https://www.treccani.it/vocabolario/isteresi)*.

## **Istogramma:**
È un diagramma che fornisce una rappresentazione di un insieme di dati statistici mediante un grafico a barre.  
*Fonte: [YouMath](https://www.youmath.it/lezioni/algebra-elementare/lezioni-di-algebra-e-aritmetica-per-scuole-medie/1711-istogrammi.html)*.

## **Matrice densa:**
Matrice i cui valori sono quasi tutti diversi da zero.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Matrice_sparsa)*.

## **Matrice di confusione:**
È un metodo di visualizzazione per i risultati di un algoritmo di classificazione. È una tabella che suddivide il numero di istanze di verità base di una classe specifica rispetto al numero totale di istanze di quella classe.  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/confusion-matrix)*.

## **Machine Learning:**
Il Machine Learning è un sottoinsieme dell’intelligenza artificiale (AI) incentrato sulla costruzione di sistemi ed algoritmi in grado di apprendere dai dati e migliorare con l'esperienza, senza essere stati esplicitamente programmati.  
*Fonte: [Oracle](https://www.oracle.com/it/artificial-intelligence/machine-learning/what-is-machine-learning)*.

## **Matrice sparsa:**
Matrice i cui valori sono quasi tutti zero.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Matrice_sparsa)*.

##  **Media (aritmetica):**
Dato un insieme di valori, la media si ottiene sommando tutti i valori e dividendo la somma per il numero di valori.  
*Fonte: [YouMath](https://www.youmath.it/lezioni/algebra-elementare/lezioni-di-algebra-e-aritmetica-per-scuole-medie/2064-media-moda-mediana.html#media-dati-statistici)*.

## **Mediana:**
Dato un insieme di dati disposti in ordine (crescente o decrescente) la mediana è il valore (o la coppia di valori in caso di numero pari di valori) che occupa la posizione centrale.  
*Fonte: [YouMath](https://www.youmath.it/lezioni/algebra-elementare/lezioni-di-algebra-e-aritmetica-per-scuole-medie/2064-media-moda-mediana.html#mediana-dati-statistici)*.

## **Metodo dei minimi quadrati (Ordinary Least Squares):**
È una tecnica di [regressione](#regressione-lineare) che permette di trovare una funzione, rappresentata da una curva di regressione, che si avvicini più possibile ad un insieme di dati. La funzione trovata è quella che minimizza la somma dei quadrati delle distanze tra i dati osservati e quelli della curva che rappresenta la funzione stessa.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Metodo_dei_minimi_quadrati)*.

## **Modalita' statistica:**
Una modalità di un carattere è uno dei suoi possibili possibili valori. Per carattere quantitativi si usa anche il termine valore.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Modalità_\(statistica\))*.

## **Modelli a diffusione:**
Sono una classe di modelli generativi basati su reti neurali, utilizzati principalmente per la generazione di immagini di alta qualità e per altri compiti di computer vision.  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/diffusion-models)*.

## **Modelli lineari:**
I modelli lineari eseguono la previsione per un obiettivo continuo in base alle relazioni lineari tra l'obiettivo e uno o più predittori.  
*Fonte: [IBM](https://www.ibm.com/docs/it/spss-statistics/30.0.0?topic=features-linear-models)*.

## **Modello di regressione:**
Studia la correlazione del variare di una variabile Y (dipendente) al variare di un’altra variabile X (indipendente). Per ogni valore di X esiste un sottoinsieme di valori di Y che seguono una distribuzione Gaussiana. È una tecnica adatta per lo sviluppo di un modello predittivo, data una X predice Y.  
*Fonte: [Università di Bari](https://www.uniba.it/it/docenti/bartolomeo-nicola/attivita-didattica/08_Regressione.pdf)*.

## **Multicollinearita':**
Indica quando le variabili indipendenti in un'equazione di [regressione lineare](#regressione-lineare) sono [correlate](#correlazione). La multicollinearità impedisce ai modelli predittivi di produrre previsioni [accurate](#accuratezza) aumentando la complessità del modello e l'[overfitting](#overfitting).  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/multicollinearity)*.

## **OCR (Optical Character Recognition):**
Individua le lettere presenti in, ad esempio in un'immagine, e le trasforma in stringhe testuali, in formato leggibile facilemente da altri software.  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/optical-character-recognition)*.

## **Outlier:**
Osservazione lontana dalla maggioranza dei dati. Valori anomali.  
*Fonte: [Treccani](https://www.treccani.it/enciclopedia/outlier_\(Dizionario-di-Economia-e-Finanza\))*.

## **Overfitting:**
Nel machine learning, l’overfitting si verifica quando un algoritmo si adatta troppo fedelmente ai suoi dati di addestramento, dando luogo ad un modello che non è in grado di effettuare previsioni accurate da dati diversi da quelli di addestramento.  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/overfitting#1580786320)*.

## **PCA (Principal Component Analysis):**
Riduce il numero di dimensioni in grandi set di dati a componenti principali, che conservano la maggior part delle informazioni originali.  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/principal-component-analysis)*.

## **Popolazione statistica:**
L’insieme degli elementi che sono oggetto di studio, ossia l’insieme delle [unità statistiche](#unita-statistica) sulle quali viene effettuata la rilevazione.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Popolazione_statistica)*.

## **Precisione:**
Misura l’affidabilità delle previsioni positive (veri positivi / (veri positivi \+ falsi positivi)). È vitale se i falsi allarmi sono molto rilevanti.  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/confusion-matrix)*.

## **Quartili:**
Valori in corrispondenza dei quali la distribuzione viene suddivisa in quattro parti uguali. I quartili sono:
1. Q1 (inferiore): valore al di sotto del quale cade il 25% dei casi.  
2. Q2 (mediano): valore al di sotto del quale cade il 50% dei casi.  
3. Q3 (superiore): valore al di sotto del quale cade il 75% dei casi.
Calcolo:
1. Ordinare in modo crescente i valori.  
2. Calcolare le [frequenze cumulate](#frequenza-cumulata-assoluta).  
3. Calcolare la posizione del quartile tramite le formule (N \= numeri di valori):  
   1. posQ1 \= (N \+ 1\) / 4  
   2. posQ2 \= ((N \+ 1\) / 4\) \* 2  
   3. posQ3 \= ((N \+ 1\) / 4\) \* 3  
   
*Fonte: [Sapienza](https://elearning.uniroma1.it/pluginfile.php/120245/mod_resource/content/0/Slide_Lezione_11_Indicatori%20di%20posizione.pdf)*.

## **Rango:**
Indica il posto occupato in un ordine. La correlazione per ranghi (coefficiente per ranghi) fornisce un indice numerico per stabilire quanto siano correlate due graduatorie.  
*Fonte: [Treccani](https://www.treccani.it/enciclopedia/rango_\(Enciclopedia-della-Matematica\))*.

## **Regola dei terzi:**
È una linea guida per la composizione che pone il soggetto su un terzo a sinistra o a destra di un’immagine, lasciando gli altri due terzi più aperti. Permette di realizzare foto coinvolgenti e ben composte.  
*Fonte: [Adobe](https://www.adobe.com/it/creativecloud/photography/discover/rule-of-thirds.html)*.

## **Regressione (lineare):**
La regressione formalizza e risolve il problema di una relazione funzionale tra variabili misurate sulla base di dati campionari estratti da un’ipotetica popolazione infinita.  
*Fonte: [WIkipedia](https://it.wikipedia.org/wiki/Regressione_lineare)*.

## **Regressione multipla:**
Metodo di stima di un modello in cui un certo numero di variabili è usato per spiegare un’altra variabile.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Regressione_multipla)*.

## **Reti Generative Adversarial Network (GAN):**
Una GAN è un modello di apprendimento automatico (basato sul [deep learning](#deep-learning)) progettato per generare dati realistici (esempio foto).  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/generative-adversarial-networks)*.

## **Rete neurale residuale (ResNet):**
È un’architettura di deep learning progettata in modo che i layer apprendano funzioni in relazione agli input dei layer stessi. È stata sviluppata per problemi di riconoscimento di immagini nella computer vision.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Rete_neurale_residuale)*.

## **Sensibilita':**
Misura la capacità del modello di individuare tutti i casi positivi effettivi (veri positivi / (veri positivi \+ falsi negativi)). È cruciale se è importante catturare tutti i positivi, avendo un basso tasso di mancate rilevazioni.  
*Fonte: [IBM](https://www.ibm.com/it-it/think/topics/confusion-matrix)*.

## **Sfocatura percettiva di Crete-Roffet:**
La metrica di Crete-Roffet stima la sfocatura senza bisogno di un’immagine di riferimento: sfoca ulteriormente l’immagine e misura quanto cambia la variazione tra pixel vicini. Un’immagine nitida ha bordi netti che l’ulteriore sfocatura ammorbidisce chiaramente (cambia molto); un’immagine già sfocata ha poco dettaglio da perdere (cambia poco). Il punteggio va da 0 (nitida) a 1 (molto sfocata).  
*Fonte: [Crété-Roffet et al., 2007](https://doi.org/10.1117/12.702790)*.

## **Simmetria statistica:**
Una [distribuzione](#distribuzione-statistica) di probabilità è simmetrica quando la sua funzione di probabilità è simmetrica rispetto ad un particolare valore x0.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Simmetria_\(statistica\))*.

## **SVD (SIngular Value Decomposition):**
È un metodo di fattorizzazione in algebra lineare che scompone una matrice in tre matrici, fornendo un modo per rappresentare i dati in termini di valori singolari.  
*Fonte: [Geeksforgeeks](https://www.geeksforgeeks.org/data-science/singular-value-decomposition-svd)*.

## **TF-IDF (Term Frequency \- Inverse Document Frequency):**
È un metodo statistico utilizzato nell’elaborazione del linguaggio naturale per valutare quanto sia importante una parola per un documento in relazione a una raccolta di più documenti.  
*Fonte: [Geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency)*.

## **Transformer (deep learning):**
Il Transformer è un’architettura di rete neurale artificiale basata sul meccanismo di attenzione multi-testa. Il testo viene convertito in token (rappresentazioni numeriche), trasformati poi in vettori matematici. In ogni livello di rete, ogni token viene contestualizzato rispetto agli altri token vicini elaborandoli in parallelo. Ciò amplifica il segnale dei token principali e riduce l’importanza dei token meno rilevanti.  
*Fonte: [Wikipedia](https://en.wikipedia.org/wiki/Transformer_\(deep_learning\))*.

## **Unita' statistica:**
Individuo o ente sul quale viene effettuata una rilevazione statistica.  
*Fonte: [Treccani](https://www.treccani.it/enciclopedia/unita-statistica_(Enciclopedia-della-Matematica))*.

## **Variabili latenti:**
Variabili non misurate direttamente, ma che spiegano la [correlazione](#correlazione) tra le variabili osservate.  
*Fonte: [Sapienza](https://elearning.uniroma1.it/pluginfile.php/928564/mod_resource/content/1/NOTE%20SUI%20SEM.pdf)*.

## **Varianza del Laplaciano:**
Il laplaciano è un operatore differenziale del secondo ordine, definito come la divergenza del gradiente di una funzione. Nell’elaborazione delle immagini la sua versione discreta (somma delle differenze rispetto ai pixel vicini) è un filtro dei bordi: risponde con valori forti dove l’intensità cambia bruscamente. La varianza della sua risposta sull’intera immagine è quindi una misura di nitidezza: alta in presenza di molti dettagli fini, bassa nelle immagini sfocate.  
*Fonte: [Wikipedia](https://it.wikipedia.org/wiki/Operatore_di_Laplace)*.
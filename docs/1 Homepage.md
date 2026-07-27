# **Analisi delle alterazioni delle immagini nell’advertising online**

## **Contesto:**
Le immagini pubblicitarie online sono sempre più spesso alterate, ad esempio con il fotoritocco tradizionale o con l’AI generativa. Da un lato queste tecniche offrono uno strumento creativo innovativo ma, dall’altro lato, agevolano la creazione di pubblicità ingannevoli iper-realistiche (specialmente in ambiti sensibili come la comunicazione politica, dove i contenuti alterati possono orientare l’opinione pubblica).  
La ricerca attuale si concentra sulla classificazione vero/falso dell’intera immagine, tralasciando quali componenti dell’annuncio siano state alterate: un’informazione essenziale quando la manipolazione non genera l’immagine da zero, ma si limita a modificare o aggiungere singoli oggetti.

## **Obiettivo:**
L’obiettivo del tirocinio è inferire quali parti di un’immagine pubblicitaria vengano generate o alterate, e mettere in relazione queste alterazioni con le prestazioni del post (engagement). Il caso di studio è l’advertising politico su Facebook ed Instagram durante le elezioni europee 2024, raccolto tramite [CrowdTangle](3%20Definizioni.md#crowdtangle). Invece di limitarsi alla rilevazione di [deepfake](3%20Definizioni.md#deepfake) (e [cheapfake](3%20Definizioni.md#cheapfake)), il progetto mira all’[Explainability (XAI)](3%20Definizioni.md#explainable-ai-xai) e alla [Image Forgery Localization (IFL)](3%20Definizioni.md#image-forgery-localization-ifl).

## **Strumenti e metodologie:**
Il progetto si articola in una [Exploratory Data Analysis (EDA)](3%20Definizioni.md#analisi-esplorativa-eda) con i seguenti passaggi chiave:
1. Raccolta e filtraggio dei dati: costruzione del dataset di annunci (post con immagine e relativo punteggio di engagement) e selezione dei post analizzabili; un sottoinsieme viene etichettato manualmente (reale/alterata) per valutare l’affidabilità del detector.  
2. Localizzazione delle alterazioni: applicazione di [framework](3%20Definizioni.md#framework) ed algoritmi avanzati in grado di generare mappe di localizzazione delle anomalie.  
3. Inferenza semantica e analisi strategica: partendo dalle mappe di localizzazione, si descrivono le singole componenti alterate e si verifica se — e come — il contenuto delle alterazioni si leghi alle prestazioni dell’annuncio. Si risponde a domande del tipo “Che cosa è stato alterato?” e “Questo tipo di alterazioni influenza l’engagement del post?”

## **Risultati (finora):**
Sulle circa 1200 immagini analizzate, l’intensità della manipolazione non predice l’engagement: il punteggio globale di [TruFor](6%20TruFor.md) è sostanzialmente scorrelato dal punteggio CrowdTangle ([Pearson](3%20Definizioni.md#indice-di-correlazione-pearson-coefficiente-di-correlazione-lineare) r ≈ 0.05). Il contenuto delle alterazioni porta invece un segnale debole ma reale: le descrizioni testuali delle regioni alterate (didascalia [BLIP](3%20Definizioni.md#blip-bootstrapping-language-image-pre-training) e parole lette via [OCR](3%20Definizioni.md#ocr-optical-character-recognition)), vettorizzate con [TF-IDF](3%20Definizioni.md#tf-idf-term-frequency---inverse-document-frequency), distinguono i post che superano le aspettative di engagement da quelli che le deludono con [AUC](3%20Definizioni.md#auc-area-under-the-curve) ≈ 0.64 (≈ 60% di [accuratezza](3%20Definizioni.md#accuratezza) contro il ≈ 51% della baseline di maggioranza). Il segnale predittivo sta quindi nel contenuto della manipolazione, non nella sua intensità.  
Sul sottoinsieme di 100 immagini etichettate a mano, TruFor come rilevatore risulta preciso ma conservativo: 61% di accuratezza, 80.6% di [precisione](3%20Definizioni.md#precisione) e 47.5% di [sensibilità](3%20Definizioni.md#sensibilita) (soglia 0.5) — quando segnala un’immagine come alterata ha ragione circa 4 volte su 5, ma alla soglia predefinita manca circa metà delle immagini manipolate.

# **Indice**
1. ## [**Homepage**](1%20Homepage.md)
2. ## [**Studio di fattibilità**](2%20Studio%20fattibilità.md)
3. ## [**Definizioni**](3%20Definizioni.md)
4. ## [**Dataset**](4%20Dataset.md) **(Elezioni europee 2024)**
5. ## [**Approfondimento teorico**](5%20Approfondimento%20teorico.md)
6. ## [**Paper TruFor**](6%20Paper%20TruFor.pdf) **(con [riassunto](6%20TruFor.md))**
7. ## [**Pipeline**](7%20Pipeline.md) **(dalla configurazione dell'ambiente all'analisi dei risultati)**
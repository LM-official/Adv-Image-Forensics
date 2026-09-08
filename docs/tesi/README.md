La tesi usa la classe `sapthesis` e si compila dal documento principale `tesi.tex`.

- `tesi.tex`: frontespizio, indice, ordine dei capitoli e bibliografia.
- `packages.tex`: pacchetti, lingua italiana, percorso delle immagini e metadati PDF.
- `capitoli/`: un file `.tex` per ogni capitolo.
- `immagini/`: figure da inserire con `\includegraphics{nome_file}`.
- `references.bib`: fonti bibliografiche.
- `sapthesis/`: classe, logo, stile bibliografico e documentazione originali. `tesi.tex` aggiunge questa cartella ai percorsi di ricerca prima di caricare la classe; `packages.tex` la include anche nei percorsi delle immagini.
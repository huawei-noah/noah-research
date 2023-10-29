This repository contains code for paper V. Malykh, K. Chernis, E. Artemova, I. Pionkovskaya. "SumTitles: a Summarization Corpus with Low Extractivity". COLING 2020.

The corpus SumTitles consists of three parts: Subtitles, Scripts, and Gold. 

To collect a **Subtitles** part of the corpus please download English monolingual subcorpus of [OpenSubtitles-v2018](http://opus.nlpl.eu/OpenSubtitles-v2018.php).
Then please unpack it to the default folder `./en/OpenSubtitles/xml/en` or another place by your choice.  Then you need to run 
`cd opensubtitles_corpus; python3 opensubtitles_processing.py` to process and align the subtitles. Use `--subtitles-path` command line option to specify the path.

To collect a **Scripts** part of the corpus you need to download movie scripts first.
To download the scripts please use `cd kvtoraman_scraper; sh scrap_films.sh`. Be carefull, the process requires several hours.
After that you need to make an alignment.
To make alignment for the scripts, please use command `sh ./parse.sh`.
Gold part is collected together with Scripts one.

Please use the following bibtex code for citation: 
```
@inproceedings{malykh-etal-2020-sumtitles,
    title = "{S}um{T}itles: a Summarization Dataset with Low Extractiveness",
    author = "Malykh, Valentin  and
      Chernis, Konstantin  and
      Artemova, Ekaterina  and
      Piontkovskaya, Irina",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.503",
    doi = "10.18653/v1/2020.coling-main.503",
    pages = "5718--5730",
    abstract = "The existing dialogue summarization corpora are significantly extractive. We introduce a methodology for dataset extractiveness evaluation and present a new low-extractive corpus of movie dialogues for abstractive text summarization along with baseline evaluation. The corpus contains 153k dialogues and consists of three parts: 1) automatically aligned subtitles, 2) automatically aligned scenes from scripts, and 3) manually aligned scenes from scripts. We also present an alignment algorithm which we use to construct the corpus.",
}
```
【This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.】
# Data lineage summary

Resources are in folder `../resources`.

### Out-of-domain data files

*Out-of-domain* (OOD) resources are public-domain corpus files. OOD are ensemble of sentences that are not of our knowledge-base/domain.

They come from following sources:
1. https://github.com/gunthercox/chatterbot-corpus which are sentences made to train bots for small talk
2. https://github.com/zeloru/small-english-smalltalk-corpus , ensemble of small talk conversations

Thos resources are copied into folders `../resources/chatterbox` and `../resources/convo` respectively.

## In-Domain data files 

*In-domain* (ID) resources are sample knowledge bases. Those texts are scraped from different FAQs:
- The Europcar FAQ, Average Size, from Car rental domain. Scrapped from all child pages of https://faq.europcar.com/
- The Fedora Project FAQ (Linux distribution community project), Small Sized. Scrapped from a single page here https://fedoraproject.org/wiki/FAQ#Getting_Started

Scrapping code (and run) can be found in notebook scraping.ipynb
Results from scrapping are scores in `../resources/scraping`
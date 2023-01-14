# Data lineage summary

Resources are in folder `../resources`.

### Out-of-domain data files

*Out-of-domain* (OOD) resources are public-domain corpus files. OOD are ensemble of sentences that are not of our knowledge-base/domain.

They come from following sources:
1. https://github.com/gunthercox/chatterbot-corpus which are sentences made to train bots for small talk
2. https://github.com/zeloru/small-english-smalltalk-corpus , ensemble of small talk conversations

Thos resources are copied into folders `../resources/chatterbox` and `../resources/convo` respectively.

## In-Domain data files 

*In-domain* (ID) resources are sample knowledge bases. Those texts are focused to a specific domain.

The *in-domain* resources will be scraped from online FAQs in the future, but for the moment they are simple copy-paste of publicely available FAQs.
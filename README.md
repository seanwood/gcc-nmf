# GCC-NMF
GCC-NMF is a blind source separation algorithm that combines:

 - [GCC](http://ieeexplore.ieee.org/abstract/document/1162830/) spatial localization method
 - [NMF](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) unsupervised dictionary learning algorithm

GCC-NMF has been applied to stereo speech separation and enhancement in both offline and real-time settings, though it is a generic source separation algorithm and could be applicable to other types of signals.

This GitHub repository is home to open source demonstrations in the form of **iPython notebooks**:

- [Offline Speech Separation]() iPython Notebook
- [Offline Speech Enhancement]() iPython Notebook

## Offline Speech Separation and Enhancement

The notebooks in this section cover the initial presentation of GCC-NMF in the following publications:

- Sean UN Wood and Jean Rouat, [*Speech Separation with GCC-NMF*](http://www.isca-speech.org/archive/Interspeech_2016/pdfs/1449.PDF), **Interspeech 2016**.  
DOI: [10.21437/Interspeech.2016-1449](http://dx.doi.org/10.21437/Interspeech.2016-1449)

- Sean UN Wood, Jean Rouat, Stéphane Dupont, Gueorgui Pironkov, [*Speech Separation and Enhancement with GCC-NMF*](https://www.gel.usherbrooke.ca/rouat/publications/IEEE_ACMTrGCCNMFWoodRouat2017.pdf), **IEEE/ACM Transactions on Audio, Speech, and Language Processing**, vol. 25, no. 4, pp. 745–755, 2017.  
DOI: [10.1109/TASLP.2017.2656805](https://doi.org/10.1109/TASLP.2017.2656805)

### Offline Speech Separation Demo

In the [offline speech separation notebook](), we show how GCC-NMF can be used to separate multiple concurrent speakers in an offline fashion. The NMF dictionary is first learned directly from the mixture signal, and sources are subsequently separated by attributing each atom at each time to a single source based on the dictionary atoms' estimated time delay of arrival (TDOA). Source localization is achieved with GCC-PHAT.

<br>
[![png](README_files/speechSeparationNotebookThumbnail.png)]()
<br><br>

<!---
<div style="text-align:center"><img src ="README_files/speechSeparationNotebookThumbnail.png" /></div>
<img align="right" src="README_files/speechSeparationNotebookThumbnail.png">
![png](README_files/speechSeparationNotebookThumbnail.png)
-->

### Offline Speech Enhancement Demo

The [offline speech enhancement notebook]() demonstrates how GCC-NMF can can be used for offline speech enhancement, where instead of multiple speakers, we have a single speaker plus noise. In this case, individual atoms are attributed either to the speaker or to noise at each point in time base on the the atom TDOAs as above. The target speaker is again localized with GCC-PHAT.

<br>
[![png](README_files/speechEnhancementNotebookThumbnail.png)]()
<br><br>
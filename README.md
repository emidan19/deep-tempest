# Deep-tempest:  Using Deep Learning to Eavesdrop on HDMI from its Unintended Electromagnetic Emanations

<img src="deep-tempest.png"/>

## Summary

In this project we have extended the original [**gr-tempest**](https://github.com/git-artes/gr-tempest) (a.k.a. [Van Eck Phreaking](https://en.wikipedia.org/wiki/Van_Eck_phreaking) or simply TEMPEST; i.e. spying on a video display from its unintended electromagnetic emanations) by using deep learning to improve the quality of the spied images. See an illustrative diagram above. Some examples of the resulting inference of our system and the original unmodified version of gr-tempest below.

<img src="examples.png"/>

The following external webpages provide a nice summary of the work:
* NewScientist: [AI can reveal what’s on your screen via signals leaking from cables](https://www.newscientist.com/article/2439853-ai-can-reveal-whats-on-your-screen-via-signals-leaking-from-cables/)
* RTL-SDR.com: [DEEP-TEMPEST: EAVESDROPPING ON HDMI VIA SDR AND DEEP LEARNING](https://www.rtl-sdr.com/deep-tempest-eavesdropping-on-hdmi-via-sdr-and-deep-learning/)
* PC World: [Hackers can wirelessly watch your screen via HDMI radiation](https://www.pcworld.com/article/2413156/hackers-can-wirelessly-watch-your-screen-via-hdmi-radiation.html)
* Techspot: [AI can see what's on your screen by reading HDMI electromagnetic radiation](https://www.techspot.com/news/104015-ai-can-see-what-screen-reading-hdmi-electromagnetic.html)
* Futura: [Hallucinant : ce système permet d’afficher et espionner ce qu’il y a sur l’écran d’un ordinateur déconnecté](https://www.futura-sciences.com/tech/actualites/technologie-hallucinant-ce-systeme-permet-afficher-espionner-ce-quil-y-ecran-ordinateur-deconnecte-114883/)
* hackster.io: [Deep-TEMPEST Reveals All](https://www.hackster.io/news/deep-tempest-reveals-all-c8cb4f0ebd08)
* Hacker News: [Deep-Tempest: Using Deep Learning to Eavesdrop on HDMI](https://news.ycombinator.com/item?id=41116682)

## Video demo

We are particularly interested in recovering the text present in the display, and we improve the Character Error Rate from 90% in the unmodified gr-tempest, to less than 30% using our module. Watch a video of the full system in operation:

[<img src="https://img.youtube.com/vi/ig3NWg_Yzag/maxresdefault.jpg" width="50%"/> ](https://www.youtube.com/watch?v=ig3NWg_Yzag)

## How does it works? (and how to cite our work or data)

You can find a detailed technical explanation of how deep-tempest works in [**our article**](https://arxiv.org/abs/2407.09717). If you found our work or data useful for your research, please consider citing it as follows:

````
@misc{fernández2024deeptempestusingdeeplearning,
      title={Deep-TEMPEST: Using Deep Learning to Eavesdrop on HDMI from its Unintended Electromagnetic Emanations}, 
      author={Santiago Fernández and Emilio Martínez and Gabriel Varela and Pablo Musé and Federico Larroca},
      year={2024},
      eprint={2407.09717},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2407.09717},
      note={Submitted}
}
````

## Data

In addition to the source code, we are also open sourcing the whole dataset we used. Follow [this dropbox link](https://www.dropbox.com/scl/fi/7r2o8nbws45q30j5lkxjb/deeptempest_dataset.zip?rlkey=w7jvw275hu8tsyflgdkql7l1c&st=e8rdldz0&dl=0) to download a ZIP file (~7GB). After unzipping, you will find synthetic and real captured images used for experiments, training, and evaluation during the work. These images consists of 1600x900 resolution with the SDR's center frequency at the third pixel-rate harmonic (324 MHz).

The structure of the directories containing the data is **different** for **synthetic data** compared to **captured data**:

### Synthetic data

* *ground-truth* (directory with reference/monitor view images)
    - image1.png
    - ...
    - imageN.png

* *simulations* (directory with synthetic degradation/capture images)
    - image1_synthetic.png
    - ...
    - imageN_synthetic.png

### Real data

- image1.png (*image1 ground-truth*)
- ...
- imageN.png (*imageN ground-truth*)

* *Image 1* (directory with captures of *image1.png*)
    - capture1_image1.png
    - ...
    - captureM_image1.png

* ...

* *Image N* (directory with captures of *image1.png*)
    - capture1_imageN.png
    - ...
    - captureM_imageN.png

## Code and Requirements

Clone the repository:

```shell
git clone https://github.com/emidan19/deep-tempest.git
```

Both [gr-tempest](./gr-tempest/) and [end-to-end](./end-to-end/) folders contains a guide on how to execute the corresponding files for image capturing, inference and train the deep learning architecture based on DRUNet from [KAIR image restoration repository](https://github.com/cszn/KAIR/tree/master).

The code is written in Python version 3.10, using Anaconda environments. To replicate the working environment, create a new one with the libraries listed in [*requirements.txt*](./requirements.txt):

```shell
conda create --name deeptempest --file requirements.txt
```

Activate it with:
```shell
conda activate deeptempest
```

Regarding installations with GNU Radio, **it is necessary to use the [gr-tempest](./gr-tempest/) version in this repository** *(which contains a modified version of the original gr-tempest)*. After this, run the following *grc* files flowgraphs to activate the *hierblocks*:
- [binary_serializer.grc](./gr-tempest/examples/binary_serializer.grc)
- [FFT_autocorrelate.grc](./gr-tempest/examples/FFT_autocorrelate.grc)
- [FFT_crosscorrelate.grc](./gr-tempest/examples/FFT_crosscorrelate.grc)
- [Keep_1_in_N_frames.grc](./gr-tempest/examples/Keep_1_in_N_frames.grc)

Finally run the flowgraph [deep-tempest_example.grc](./gr-tempest/examples/deep-tempest_example.grc) to capture the monitor images and be able to recover them with better quality using the *Save Capture* block.

## Credits

IIE Instituto de Ingeniería Eléctrica, 
Facultad de Ingeniería, 
Universidad de la República, 
Montevideo, Uruguay, 
http://iie.fing.edu.uy/investigacion/grupos/artes/

Please refer to the LICENSE file for contact information and further credits.

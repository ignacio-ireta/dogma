# Krizhevsky, Sutskever & Hinton — AlexNet / ImageNet Classification with Deep Convolutional Neural Networks (2012)

#paper #milestone #read

**Authors:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
**Institution:** University of Toronto  
**Published:** NeurIPS 2012  
**Paper page:** https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html  
**Slides / workshop PDF:** https://www.image-net.org/static_files/files/supervision.pdf

---

## The Question

LeCun's CNN idea was right in 1989. So why didn't deep vision take over until 2012?

Because the missing ingredients were not conceptual. They were practical: enough labeled data, enough GPU compute, and a training recipe that did not collapse under depth.

AlexNet is the moment those ingredients finally clicked.

---

## Paper Reference

The paper trained a deep convolutional network on **1.2–1.3 million ImageNet images across 1000 classes**. On the NeurIPS paper's benchmark split, the model reported **18.9% top-5 error** and **39.7% top-1 error**. In the **ILSVRC-2012 competition**, the SuperVision system won with **15.3% top-5 test error**, crushing the runner-up's **26.2%**.

That gap is the sound of a field waking up.

---

## Key Ideas

### 1. Scale the CNN up, properly
The network had **8 learned layers**: **5 convolutional + 3 fully connected**, around **60 million parameters**, and about **650,000 neurons**. This was not a toy digit recognizer anymore. It was a serious high-capacity vision model.

### 2. ReLU instead of saturating activations
AlexNet used **non-saturating ReLUs** instead of tanh/sigmoid. That sounds ordinary now, but at the time it was a speed weapon. ReLUs trained dramatically faster and made deeper convolutional nets much more practical.

### 3. Train on GPUs or go home
The model was split across **two GTX 580 GPUs**. This was one of the decisive moments where deep learning stopped being just a theory question and became a systems engineering question. Hardware was part of the algorithm now.

### 4. Regularize aggressively
ImageNet was large, but the network was also huge. AlexNet used:
- **data augmentation** (random crops, flips, RGB/intensity shifts)
- **dropout** in the fully connected layers
- **weight decay**

This is why [[../10-Backpropagation/Hinton Dropout 2012|Dropout]] matters so much: it was not an isolated trick, it was one of the pillars that kept AlexNet from memorizing the dataset.

### 5. Pooling and normalization as training stabilizers
The architecture used **overlapping max-pooling** and **local response normalization (LRN)**. LRN did not survive into the modern canon, but in 2012 it was part of the workable recipe.

---

## Why This Paper Hit So Hard

The important thing is not just that AlexNet won. It is **how much** it won by.

Deep learning had existed for decades, but many people still treated it like an interesting niche. AlexNet did not edge out the competition. It embarrassed the old regime in public.

That changed:
- what computer vision researchers believed was possible,
- what industry believed was worth funding,
- and what hardware companies suddenly mattered to AI.

2012 is the year the curve bends.

---

## Lineage

- **Built on:** [[LeCun Backpropagation Zip Codes 1989]] — the original CNN logic: local receptive fields, weight sharing, pooling
- **Built on:** [[../10-Backpropagation/Rumelhart Hinton Williams Backpropagation 1986]] — still the core training algorithm underneath the glamour
- **Built on:** [[../10-Backpropagation/Hinton Dropout 2012]] — regularization that made large nets survivable
- **Built on:** Hinton's 2006 deep learning revival and the general return of multilayer networks
- **Leads to:** [[He ResNets 2015]] — once AlexNet proved depth worked, the next question became: how deep can we go?
- **Leads to:** VGG, GoogLeNet, ResNet, and then the entire modern computer vision stack
- **Crosses into:** [[../12-Frontiers/Devlin BERT 2018]] and foundation-model thinking more broadly — pretrain a large model on a massive benchmark, then transfer the learned representations elsewhere

---

## Why It Matters

### For the lineage
AlexNet is the bridge between **"CNNs are elegant"** and **"CNNs run the world."**

LeCun proved convolution was a good inductive bias. AlexNet proved that with enough data, compute, and training tricks, that inductive bias could dominate a major real-world benchmark.

### The bigger pattern
AlexNet also foreshadows the LLM era uncannily well:
- more data
- more compute
- a scalable architecture
- a training recipe that finally works
- a shocking benchmark jump that resets the whole field

GPT-3 is, in spirit, AlexNet all over again.

---

## Limitations / What Aged Poorly

- **LRN mostly disappeared**; batch normalization and better initialization won later
- **Two-GPU model parallelism was a workaround**, not a clean abstraction
- **Fully connected tails were heavy**; later architectures got more parameter-efficient
- **CNN dominance was not permanent**; transformers eventually invaded vision too

But none of that changes the historical fact: this paper reopened the future.

---

## Status

- [x] Key facts reviewed
- [x] Note created
- [ ] Read the full architecture section carefully
- [ ] Add VGG / GoogLeNet as the next CNN branch

---

*LeCun lit the candle. AlexNet made the whole room notice.*

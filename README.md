# Semantic-Colorization-GAN

This is a supplementary material for the paper **SCGAN: Saliency Map-guided Colorization with Generative Adversarial Network**, IEEE Transactions on Circuits and Systems for Video Technology (TCSVT'20).

IEEE Xplore: https://ieeexplore.ieee.org/abstract/document/9257445/?casa_token=SCtE33FH3SAAAAAA:NaOfOWFjqItL2yizMZVNYVswXSv7Djl0vezWVTzpWajY8CbBoq9piVlU3z9GQjv8ZFCPCUXPoZQ

Arxiv: https://arxiv.org/abs/2011.11377

## 1 Training

We release the training code in **train** folder.

The codes require following libs:

- Python=3.6
- PyTorch>=1.0.0
- torchvision>=0.2.1
- Cuda>=8.0
- opencv-python>=4.4.0.46

If you want to train on multispectral data, please refer to the **train on multispectral images** folder.

The pre-trained global feature network can be found in: https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_my_cityu_edu_hk/ErOcFJc0pilMvCkE53Essi0Bjj89h90l0Y9kEYv390kPEw?e=exNoad

The saliency maps are computed by PFAN: https://github.com/CaitinZhao/cvpr2019_Pyramid-Feature-Attention-Network-for-Saliency-detection

## 2 Evaluation

Please refer to **evaluation** folder.

## 3 Testing Examples

### 3.1 Colorization Results

We show the representative image of our system.

![Represent](./img/representative_image.jpg)
![Represent](./img/representative_image2.jpg)

We provide a lot of results randomly selected from **ImageNet** and **MIT Place 365** validation datasets. These images contain multiple scenes and colors.

![Results](./img/results.png)

### 3.2 Comparison Results

The comparison results with other fully-automatic algorithms are:

![Comparison1](./img/fully_automatic.png)

The comparison results with other example-based algorithms are:

![Comparison2](./img/example_based.png)

### 3.3 Examples of Semantic Confusion and Object Intervention Problems

We give some examples to illustrate the semantic confusion and object intervention problems intuitively. The SCGAN intergrates the low-level and high-level semantic information and understands how to genrate a reasonable colorization. These settings / architectures help the main colorization network to **minimize the semantic confusion and object intervention problems**.

There is some examples of semantic confusion problem.

![Semantic Confusion](./img/semantic_confusion.png)

There is some examples of object intervention problem.

![Object Intervention](./img/object_intervention.png)

To further prove this point, we give more examples about generated attention region and how the saliency map works.

![Attention Region](./img/sal.png)

### 3.4 How our Model Learns at each Epoch

In order to prove our system has the strong fitting ability, we plot the evolution of results of multiple epochs of pre-training term and refinement term. We can see the CNN learns the high-level information at second term.

![Evolution](./img/evolution_by_epoch.png)

## 4 Legacy Image Colorization

### 4.1 Portrait Photographs

We choose several **famous legacy portrait photographs** in our experiments. The photographs chosen are with different race, gender, age, and scene. We also select a photo of Andy Lau, which represents the contemporary photographs.

![People Image](./img/people.png)

### 4.2 Landscape Photographs

We choose many **landscape photographs** by Ansel Adams because the quality is so good. While these photographs are taken from US National Archives (Public Domain).

![Landscape Image](./img/landscape.png)

### 4.3 Famous Lagacy Photographs

In this section, we select some **famous photographs** (especially before 1950). And we give a color version of them.

![Famous Image2](./img/legacy.png)

### 4.4 Other Works

There are many fantastic **legacy photography works**. Our colorization system still predicts visually high-quality reasonable colorized images.

![Famous Image1](./img/other_works.png)

## 5 Related Projects

**Automatic Colorization: [Project](https://tinyclouds.org/colorize/)
[Github](https://github.com/Armour/Automatic-Image-Colorization)**

**Learning Representations for Automatic Colorization: [Project](http://people.cs.uchicago.edu/~larsson/colorization/)
[Paper](https://arxiv.org/abs/1603.06668)
[Github](https://github.com/gustavla/autocolorize)**

**Colorful Image Colorization: [Project](http://richzhang.github.io/colorization/)
[Paper](https://arxiv.org/abs/1603.08511)
[Github](https://github.com/richzhang/colorization)**

**Let there be Color!: [Project](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/)
[Paper](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/data/colorization_sig2016.pdf)
[Github](https://github.com/satoshiiizuka/siggraph2016_colorization)**

**Deoldify: [Project](https://github.com/jantic/DeOldify/)
[Project2](https://deoldify.ai/)
[Github](https://github.com/jantic/DeOldify)**

**ColouriseSG: [Project](https://colourise.sg/)**

**Pix2Pix: [Project](https://phillipi.github.io/pix2pix/)
[Paper](https://arxiv.org/pdf/1611.07004.pdf)
[Github](https://github.com/phillipi/pix2pix)**

**CycleGAN: [Project](https://junyanz.github.io/CycleGAN/)
[Paper](https://arxiv.org/pdf/1703.10593.pdf)
[Github](https://github.com/junyanz/CycleGAN)**

## 6 Reference

If you think the paper is helpful for your research, please cite:
```bash
@article{zhao2020scgan,
  title={SCGAN: Saliency Map-guided Colorization with Generative Adversarial Network},
  author={Zhao, Yuzhi and Po, Lai-Man and Cheung, Kwok-Wai and Yu, Wing-Yin and Abbas Ur Rehman, Yasar},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={31},
  number={8},
  pages={3062--3077},
  year={2020}
}
```

## 7 Find our Latest Works About Image / Video Colorization

A similar work on mobile phone image enhancement is available in this [webpage](https://github.com/zhaoyuzhi/RAW2RGB-GAN)
```bash
@inproceedings{zhao2019saliency,
  title={Saliency map-aided generative adversarial network for raw to rgb mapping},
  author={Zhao, Yuzhi and Po, Lai-Man and Zhang, Tiantian and Liao, Zongbang and Shi, Xiang and others},
  booktitle={Proceedings of the International Conference on Computer Vision Workshop},
  pages={3449--3457},
  year={2019}
}
```

A SOTA fully-automatic video colorization work is available in this [webpage](https://github.com/zhaoyuzhi/VCGAN)
```bash
@article{zhao2022vcgan,
  title={VCGAN: Video Colorization with Hybrid Generative Adversarial Network},
  author={Zhao, Yuzhi and Po, Lai-Man and Yu, Wing-Yin and Rehman, Yasar Abbas Ur and Liu, Mengyang and Zhang, Yujia and Ou, Weifeng},
  journal={IEEE Transactions on Multimedia},
  doi={10.1109/TMM.2022.3154600},
  year={2022}
}
```

A legacy photo restoration work including scribble-based image colorization is available in this [webpage](https://github.com/zhaoyuzhi/Legacy-Photo-Editing-with-Learned-Noise-Prior)
```bash
@inproceedings{zhao2021legacy,
  title={Legacy Photo Editing with Learned Noise Prior},
  author={Zhao, Yuzhi and Po, Lai-Man and Lin, Tingyu and Wang, Xuehui and Liu, Kangcheng and Zhang, Yujia and Yu, Wing-Yin and Xian, Pengfei and Xiong, Jingjing},
  booktitle={Proceedings of the Winter Conference on Applications of Computer Vision},
  pages={2103--2112},
  year={2021}
}
```

A SOTA scribble-based video colorization work is available in this [webpage](https://github.com/zhaoyuzhi/SVCNet)
```bash
@article{zhao2022svcnet,
  title={SVCNet: Real-time Scribble-based Video Colorization with Pyramid Networks},
  author={Zhao, Yuzhi and Po, Lai-Man and Liu, Kangcheng and Wang, Xuehui and Yu, Wing-Yin and Xian, Pengfei},
  journal={arXiv preprint},
  year={2022}
}
```

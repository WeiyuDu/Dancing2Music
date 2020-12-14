## Dancing to Music - CIS 565 Final Project
### Han Yan, Weiyu Du

### Final Presentation
1) Improved model architecture for the Music Style Extractor: First of all, the paper used a hidden size of 30 for RNN, which was too weak to us. Instead, we used hidden size of 512, then added another linear layer to map it to 30. Moreover, we added two dropout layers in the classifier to help regularize the network. We also found that in the data processing stage, the audio file was truncated to 30 data points. We extended this to 59, the length of the shortest audio file in the dataset so we would not require sequence padding.

The paper did not provide details about how they trained the Music Style Extractor, so we built our own training pipeline. After 100 epochs, we found our accuracy to be **98.7%** ，which is a significant improvement from the original **73.5%** . We have reason to believe this will provides better music style feature, thus better dance generation quality.

2) We also trained a new model only on Ballet dances to verify, if, aside from the confusion about types of music, the model can capture typical ballet movements. Please see our visual results in the final presentation.

### Milestone 3 - Model Drilldown

#### Evaluation on Music Style Classification
  
Overall accuracy: **73.5%**
|  | Ballet | Zumba | Hiphop |
| ------ | ------ | ----- | ------ |
| Precision | 89.8% | 71.9% | 52.6% |
| Predicted as Ballet | N/A | 4.8% | 36.8% |
| Predicted as Zumba | 4.3% | N/A | 10.4% |
| Predicted as Hiphop | 5.8% | 23.2% | N/A |

#### Evaluation on Impact of Initial Pose

We take Gaussian distributions on the initial pose:
| sample 1 | sample 2 | sample 3 | sample 4 |
:------:|:------:|:------:|:------:
![](imgs/ms3_sample1.gif) | ![](imgs/ms3_sample2.gif) | ![](imgs/ms3_sample3.gif) | ![](imgs/ms3_sample4.gif)

#### Ablation Studies 

* Alignment of latent dance codes 
* Alignment between music and dance style
* Alignment of latent dance movements

### Conclusion
Music Style Extractor performs surprisingly poor on style classification task, which means 1) the model fails to understand what dance style it should generate 2) music feature is very poor. Combining with the lack of significant visual difference in the generation results of our ablation studies, we conclude music style understanding is the most evident bottleneck for our dancing to music task.

### Milestone 2 - Performance Analysis on CPU vs GPU with varying batch size

#### Training time: forward and backward network pass

| CPU training time           |  GPU training time |
:-------------------------:|:-------------------------:
![](imgs/cpu_train.png) | ![](imgs/gpu_train.png)

#### Training time: data loading

<img src="imgs/data_loading.png" width=400>

#### Inference time:

<img src="imgs/inference.png" width=400>

#### Model architecture: LSTM vs. GRU

<img src="imgs/model_arch.png" width=400>

### Milestone 1 - Code Refinement, Model Training and Visualization

#### Training

| ballet | hiphop |
:------:|:------:
![](imgs/ballet.gif) | ![](imgs/hiphop.gif)

#### Inference result

Generated dance using music in Sylvia - Act 3 solo piece

<img src="imgs/dance_sylvaia.gif" width=400>

For reference, here's a link to real person performance: https://www.youtube.com/watch?v=We7KAkWJow8

### Links to Presentations
Milestone 1: https://docs.google.com/presentation/d/11YTHrU7iGCIOVsd0SLnEJZcHaSJD--XU_pph3FcJ-fk/edit?usp=sharing
Milestone 2: https://docs.google.com/presentation/d/1MyYJxO-48K1sjDSLLNfdr59GFTHjvSDZCJRkToGwey4/edit?usp=sharing
Milestone 3: https://docs.google.com/presentation/d/1OBWptq5f9bYVLz_lLq-whvNVv35XpLIs4e05uK2CO7g/edit?usp=sharing

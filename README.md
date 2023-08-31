# Simultaneous Alignment and Surface Regression Using Hybrid 2D-3D Networks for 3D Coherent Layer Segmentation of Retinal OCT Images with Full and Sparse Annotations

Code for paper "Simultaneous Alignment and Surface Regression Using Hybrid 2D-3D Networks for 3D Coherent Layer Segmentation of Retinal OCT Images with Full and Sparse Annotations." （See fig below）

![image](https://github.com/ccarliu/Retinal-OCT-LayerSeg/assets/32379010/6be3426a-dff3-4cf3-9721-0c6bb0fe18ea)

Parts of codes are borrowed from https://github.com/Hui-Xie/DeepLearningSeg.

The preprocess code: https://github.com/YufanHe/oct_preprocess

## Datasets

1. public A2A OCT dataset: https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm
2. public JHU OCT dataset: http://iacl.ece.jhu.edu/index.php/Resources
3. public DME dataset: https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm

## Introduce -- Train and test process
Take the dme dataset as example:
1. Data preparing: Down the dataset from [here](https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm)
2. Environment preparing: The codes are implemented in ubuntu system. More requirements and virtual environment creation cound be seen in requirements.txt and run.sh. 
3. Training: Replace the datapath in the training command of run.sh, and run it. After training, the model will be stored in "./checkpoint". We provide pretrained models as described in our paper, together with the code.
4. Testing: Replace the datapath and checkpoint path in the testing command of run.sh, and run it, which will do the test and print the result.

## Contact

If you have any questions, please do not hesitate to contact liuhong@stu.xmu.edu.cn

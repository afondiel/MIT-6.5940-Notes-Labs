# 📘 Lecture 7: Manual Architecture Design (Classic & Mobile)

A concise, high-utility study guide for the core papers covered in MIT 6.5940 Lectures 7 and 8 to assist with your Lab 3 completion.


## 1. AlexNet (2012)
 
* Problem: Traditional vision methods scaled poorly on large-scale datasets like ImageNet.
* Solution: Deployed a deep 8-layer Convolutional Neural Network (CNN) trained on efficient GPUs.
* Findings/Results: Achieved a breakthrough 15.3% top-5 error rate.
* Challenge: Suffered from huge parameter overhead and primitive regularizations.


## 2. VGG Networks (2014)

* Problem: Unclear design principles for choosing hyper-parameters like kernel sizes.
* Solution: Standardized networks using homogenous blocks of small $3\times3$ convolutions.
* Findings/Results: Proved that smaller kernels with deeper stacks enhance feature abstraction.
* Challenge: Computationally heavy with massive fully connected layer memory footprints. 

## 3. Deep Residual Learning (ResNet) (2015)

* Problem: Deep networks suffer from vanishing/exploding gradients, degrading training accuracy.
* Solution: Introduced shortcut connections to perform identity mapping over layers.
* Findings/Results: Enabled training of ultra-deep networks up to 152 layers safely.
* Challenge: High memory requirements during inference for large feature maps.

## 4. SqueezeNet (2016)

* Problem: Large model sizes limit distributed training speed and edge deployment.
* Solution: Used "Fire modules" to squeeze channels using $1\times1$ convolutions.
* Findings/Results: Achieved AlexNet-level accuracy with a $50\times$ smaller model size.
* Challenge: Higher compute intensity per parameter compared to classic layouts.

## 5. ResNeXt (2017)
 
* Problem: Scaling accuracy via depth or width yields diminishing parameter efficiency.
* Solution: Introduced "cardinality" using parallel, split-transform-merge grouped convolutions.
* Findings/Results: Improved accuracy while keeping parameter counts strictly bounded.
* Challenge: Grouped convolutions suffered from poor hardware execution optimization.

## 6. MobileNetV1 (2017)

* Problem: Standard convolution operations are too heavy for low-power mobile devices.
* Solution: Decomposed standard convolutions into depthwise and pointwise convolutions.
* Findings/Results: Reduced total computational cost and parameters by nearly $9\times$.
* Challenge: Depthwise layers often suffer from low mathematical compute density.

## 7. MobileNetV2 (2018) (Crucial for Lab 3/Exams)
 
* Problem: Depthwise operations can accidentally destroy crucial features in low-dimensional spaces.
* Solution: Formulated Inverted Residuals paired with Linear Bottlenecks to preserve features.
* Findings/Results: Dramatically slashed memory usage via thin bottleneck interfaces.
* Challenge: Relies heavily on memory bandwidth for intermediate expansions.

## 8. ShuffleNet (2018)

* Problem: $1\times1$ dense pointwise convolutions dominate mobile model compute budgets.
* Solution: Employed pointwise group convolutions combined with a channel shuffle operation.
* Findings/Results: Restored cross-channel data flow while cutting overall computations.
* Challenge: High operational fragmentation degrades practical hardware throughput. 

## 9. RegNet (2020)


* Problem: Hand-tuning individual network configurations scales poorly across diverse constraints.
* Solution: Quantified and constrained structural trends to define network design spaces.
* Findings/Results: Found regular, predictable scaling rules outperforming complex manual designs.
* Challenge: Misses highly irregular but extremely efficient architecture outliers.

## 10. EfficientNet (2019)


* Problem: Arbitrarily scaling depth, width, or resolution yields sub-optimal accuracy gains.
* Solution: Invented Compound Scaling to expand all three dimensions uniformly.
* Findings/Results: Reached state-of-the-art accuracy using a fraction of the parameters.
* Challenge: Massive training memory requirements when scaling to large input resolutions.

## References
- https://github.com/afondiel/MIT-6.5940-Notes-Labs/blob/main/resources/references/L07-papers/L7-paper-references.md
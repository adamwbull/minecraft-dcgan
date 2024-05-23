
# Training Terms


### General

**Weights:** Learned neural network parameters. Both the generator and discriminator have weights they are optimizing against their own loss functions.

**Epoch:** A complete pass through the dataset.

### Hyperparameters Adjusted

**Learning Rate:** Affects how drastically backpropogation changes weights based on the gradient of the loss function.

*In the simplest terms, a gradient in this context refers to the vector of partial derivatives of the loss function with respect to each of the weights in the neural network. It is a vector measuring a rate of change, from which weights learn in a backward pass by correcting themselves in the opposite direction of a gradient produced in a forward pass.*

**Batch Size:** How many dataset samples the model looks at between weight updates.

*When batch_size is 1, a single sample is fed to the model. From this, losses are calculated and used to update model weights. With batch_size>1, we instead feed batch_size samples through the model, average their losses, and those averages update our weights.*

*Larger batch sizes help capture shared elements of samples at the potential cost of generalizing samples in the same batch. This generalization is stronger the larger the batch size, can be a benefit or hindrance depending on the dataset and objective of the model. A balance must be struck to create ideal outputs.*

### Assessment

**Discriminator Loss:** How well the discriminator distinguishes real from fake. 

*This should ideally be at or around 0.5-0.7, meaning real and fake are near indistinguishable. However, this should not be arrived at quickly since a loss around 0.5 can also mean the discriminator has no clue what's going on.*

**Generator Loss:** How well the generator fools the discriminator.

*What's important is making sure the generator loss improves over time, but not too quickly as that may indicate mode collapse or a faulty discriminator. It should also have "moments of struggle" where the discriminator gains an upper hand, generator loss goes up, but the generator recovers and falls down again.*

# Training Notes Explained

Dataset+X means that we trained on X structures with data augmentation.
Each `# Header` is a training attempt with a new model or larger dataset, where the header explains the context.

# Training GA-DCGAN Small on Large Dataset (Dataset+1288)

Implemented a global attention layer on its own to see if we can gain any benefits from using a high level attention mechanism while still keeping training times reasonable.

# Training LA-DCGAN and DCGAN Small, Large Dataset (Dataset+1288)

We are going to try training our LA-DCGAN on a larger dataset to explore if this creates more cohesive walls, floors, ande ceilings on structures as well as a standard DCGAN that is reminiscent of our first DCGAN, but aimed for 32 cubed outputs like our LA-DCGAN.

This LA-DCGAN model on the new dataset takes soooo long to train despite how promising the losses look on each epoch. it takes 5 days to do 100 epochs versus the simpler DCGAN doing 100 epochs in 3-5 hours. however the simpler DCGAN network is simply not up to snuff, and performs poorly and deconverges in epochs 70-150 depending on the hyperparams. outputs before deconvergence are all blurry representations of buildings, lacking fine detail to walls.

# Training LA-DCGAN Small (Dataset+976, limited to 32 cubed structures) + Attention

# Hypothesis

I have created a new, smaller neural network to focus on smaller structure creation while employing the attention mechanism. This was in part to speed up training while still exploring attention, as well as exploring whether our model would perform better with a target output size that captures a majority of the Minecraft builds we catalogued.

**Hyperparameter Targets**

Currently unable to train on higher batches without crashing.

- ll_lb (0.0001, 12)
- ml_lb (0.00015, 12)
- hl_lb (0.0002, 12)

## Summary Report (100 Epochs)

## Model Statistics (0-100 epochs best models)
### Stats for model state la2_ll_lb_terminated:
Best Loss D Closest To Half: 0.4971296191215515
Best Loss G: 0.0001435947051504627
Best Loss D: 6.755195727237151e-07
Loss D: 0.008772303350269794
Loss G: 9.574329376220703
Epoch: 0

### Stats for model state la2_ll_lb_best_loss_g:
Best Loss D Closest To Half: 0.4971296191215515
Best Loss G: 0.0001435947051504627
Best Loss D: 6.755195727237151e-07
Loss D: 1.2039103507995605
Loss G: 0.0001435947051504627
Epoch: 73

### Stats for model state la2_ll_lb_latest:
Best Loss D Closest To Half: 0.4971296191215515
Best Loss G: 0.0001435947051504627
Best Loss D: 6.755195727237151e-07
Loss D: 0.0259355790913105
Loss G: 5.093528747558594
Epoch: 75

# Training LA-DCGAN Large (Dataset+1244) + Attention

## Hypothesis

Time to investigate, using the same dataset and our best performing hyperparameter sets, the role attention layers could potentially play as described in the previous report.

Make sure to talk about how you saved computations in the query matrix lookup by limiting it to the center block.

**Hyperparameter Targets**

- hl_lb (0.0002, 12)
- ll_lb (0.0001, 12)
- ll_mb (0.0001, 18)

## Summary Report (100 Epochs)

Coming soon.

Note: ll_mb's measures beyond _latest were wiped by a system restart pre epoch 75. I have corrected the training code to prevent this for future restarts.

# Training DCGAN (Dataset+1244)

## Hypothesis

We will improve on the previous dataset experimentation by exploring our most successful learning rates on a larger dataset, along with a medium ground in between them.

We will try each of these learning rates with batch sizes that are around the most successful batch size of 16:

- 12 is a balance between 16 and 8 in the previous training cycle. 8 deconverged quickly, but perhaps the extra 4 samples will aid in generalization and provide a happy medium.

- 18 is higher than our baseline of 16 last time, but we want to push for larger batches in this cycle to explore the effects of generalization. If 18 performs better, it will become the new baseline over 16.

- 24 is our attempt for maximum generalization possible on the hardware we are training on. We tried 32 in the last training cycle, but were unable to train as the batch could not be held in memory.


**Learning Rates**

ll: 0.0001

ml: 0.00015

hl: 0.0002


**Batch Sizes**

lb: 12

mb: 18

hb: 24

## Summary Report (200 Epochs)

I have trained the three best models below without attention up to 200 epochs.

TODO: Investigate generated samples from the best loss target models within these.

## Summary Report (100 Epochs)

This training cycle proved successful in identifying further promising hyperparameters that had no problem training relatively stable over 100 epochs and produced outputs closest to resembling buildings:
- hl_lb (0.0002, 12)
- ll_lb (0.0001, 12)
- ll_mb (0.0001, 18)

Their outputs, however, are still quite fuzzy. A larger dataset and additional training on the current model may suffice to sharpen outputs, but I am drawn to attention networks and their close relation to how humans build that seems to be a key element missing from this generator's outputs. It seems to be outputting varied structures that at first glance look real, but it has a tough time with the smaller and sub-sectional details that bring a house together. For human Minecraft builders, we often have a mental image of the overall build and switch between two states: designing and outlining this build on a more global scale, or walking around and focusing on fine details, building in a smaller area, and thinking of how these areas relate to each other. Think of a builder who first sets up the cornerposts of a log cabin, then walks around filling in the wall sections one by one, taking into account how wall sections, the rooms they form, and more relate to each other. There is regional context here that isn't inherently rewarded in the current DCGAN setup with its global perspective.

While our model is well on the way to creating unique and interesting structures from this global perspective of a builder outlining their structure, a local attention layer implemented into the discriminator and generator networks could simulate the finer regional details our human builder might consider.

Undoubtedly, this process will be computationally intensive and lead to longer training times. But we will mitigate this by only employing one attention layer per model, and starting with a smaller local window size. This attention layer will identify a cube of space around every block, effectively restricting an attention operation to a a local neighborhood around each point. This in similar to how a builder considers the immediate regions around the one they are building as part of a larger structure,in the case of local attention applied to deeper convolutional layers, as we will have to do to keep computations at a reasonable level.

Theoretically with more computational power, local attention could be applied on a block-by-block basis. It may be worth resizing out final input and output spaces to explore this, but currently the 64^3 local attention cubes that would need to be computed for local attention applied to the largest tensors possible in our network is unfeasable. We will instead apply local attention to the feature maps of size 16^3, capturing regional relationships at less compute cost.

Visit [ladcgan.py](/src/ladcgan.py) to see the implementation.

### Overall Training Report for hl_hb:
Mean Discriminator Loss: 0.2129
Mean Generator Loss: 11.0869
Total Epochs: 101
Total Training Time: 1:54:43

### BEST_LOSS_D Last Saved Stats:
Epoch: 86
Discriminator Loss: 0.0000
Generator Loss: 15.9159

### Overall Training Report for hl_lb:
Mean Discriminator Loss: 0.7342
Mean Generator Loss: 6.9536
Total Epochs: 101
Total Training Time: 4:58:48

### BEST_LOSS_D Last Saved Stats:
Epoch: 58
Discriminator Loss: 0.0000
Generator Loss: 28.9427

### Overall Training Report for hl_mb:
Mean Discriminator Loss: 0.9119
Mean Generator Loss: 7.6984
Total Epochs: 101
Total Training Time: 8:44:06

### BEST_LOSS_D Last Saved Stats:
Epoch: 90
Discriminator Loss: 0.0000
Generator Loss: 25.7943

### Overall Training Report for ll_hb:
Mean Discriminator Loss: 0.4799
Mean Generator Loss: 6.7321
Total Epochs: 101
Total Training Time: 1:52:08

### BEST_LOSS_G Last Saved Stats:
Epoch: 68
Discriminator Loss: 0.0040
Generator Loss: 8.9308

### Overall Training Report for ll_lb:
Mean Discriminator Loss: 0.2419
Mean Generator Loss: 6.5616
Total Epochs: 101
Total Training Time: 13:22:05

### BEST_LOSS_D Last Saved Stats:
Epoch: 96
Discriminator Loss: 0.0152
Generator Loss: 4.9764

### BEST_LOSS_D_CLOSEST_TO_HALF Last Saved Stats:
Epoch: 86
Discriminator Loss: 0.0053
Generator Loss: 6.8833

### Overall Training Report for ll_mb:
Mean Discriminator Loss: 0.7405
Mean Generator Loss: 6.7522
Total Epochs: 101
Total Training Time: 11:13:31

Really good, diverse, and unique outputs. Still needs more learning.

### BEST_LOSS_D_CLOSEST_TO_HALF Last Saved Stats:
Epoch: 53
Discriminator Loss: 0.4453
Generator Loss: 1.0693

### Overall Training Report for ml_hb:
Mean Discriminator Loss: 0.4498
Mean Generator Loss: 7.1068
Total Epochs: 101
Total Training Time: 11:08:35

Experiencing mode collapse.

### Overall Training Report for ml_lb:
Mean Discriminator Loss: 0.4610
Mean Generator Loss: 7.4828
Total Epochs: 101
Total Training Time: 4:58:31

### BEST_LOSS_D Last Saved Stats:
Epoch: 99
Discriminator Loss: 0.0000
Generator Loss: 18.7827

### BEST_LOSS_D_CLOSEST_TO_HALF Last Saved Stats:
Epoch: 64
Discriminator Loss: 0.0346
Generator Loss: 6.5730

### Overall Training Report for ml_mb:
Mean Discriminator Loss: 0.9671
Mean Generator Loss: 6.3317
Total Epochs: 101
Total Training Time: 1:49:28

### BEST_LOSS_G Last Saved Stats:
Epoch: 81
Discriminator Loss: 1.8288
Generator Loss: 0.6433

# Training DCGAN (Dataset+872)

## Hypothesis

We aim to explore some standard learning rates employed in DCGANs, as well as some batch sizes that cover small batch sizes, but also our estimated possible maximum memory usage during training with our current hardware. Three batches were chosen with equal deviations between them leading up to this maximum as a starting point to find the ideal batch size.

**Learning Rates**

ll: 0.0001

ml: 0.0002

hl: 0.0003


**Batch Sizes**

lb: 8

mb: 16

hb: 32


## Summary Report 

We found that the low and medium rates performed admirably for the dataset they were trained on, providing decent discriminator average losses and pixelated, but nearly recognizable building outputs. We also had an error in the preprocessing data augmentation step that rotated our builds on the wrong axis, but despite this our model did well at generated buildings on every axis, despite the fact that this learned information is wrong compared to Minecraft buildings actually made by humans. With this preprocessing step fixed to rotate buildings on the correct axis, we should find even better results in our next training cycle.

### Report for medium_learn_medium_batch_best:
Mean Discriminator Loss: 0.3841

Median Discriminator Loss: 0.0210

Mode Discriminator Loss: 0.0000

Mean Generator Loss: 7.8494

Median Generator Loss: 6.9644

Mode Generator Loss: 7.4762

Total Epochs: 382

Total Training Time: 4:47:52

### Report for low_learn_medium_batch_best:
Mean Discriminator Loss: 0.3113

Median Discriminator Loss: 0.0255

Mode Discriminator Loss: 0.0000

Mean Generator Loss: 7.0641

Median Generator Loss: 6.0597

Mode Generator Loss: 6.0597

Total Epochs: 168

Total Training Time: 2:45:55

### Report for medium_learn_medium_batch_terminated:
Mean Discriminator Loss: 0.2664

Median Discriminator Loss: 0.0029

Mode Discriminator Loss: 0.0000

Mean Generator Loss: 22.4062

Median Generator Loss: 9.3507

Mode Generator Loss: 7.4762

Total Epochs: 730

Total Training Time: 9:23:28

### Report for high_learn_medium_batch:
Mean Discriminator Loss: 0.1344

Median Discriminator Loss: 0.0000

Mode Discriminator Loss: 0.0000

Mean Generator Loss: 38.7056

Median Generator Loss: 55.4074

Mode Generator Loss: 55.5773

Total Epochs: 101

Total Training Time: 1:14:35

### Report for low_learn_medium_batch_terminated:
Mean Discriminator Loss: 0.2501

Median Discriminator Loss: 0.0123

Mode Discriminator Loss: 0.0000

Mean Generator Loss: 8.4434

Median Generator Loss: 7.1449

Mode Generator Loss: 6.0597

Total Epochs: 214

Total Training Time: 3:19:37

### Report for low_learn_small_batch:
Mean Discriminator Loss: 0.0822

Median Discriminator Loss: 0.0000

Mode Discriminator Loss: 0.0000

Mean Generator Loss: 37.6540

Median Generator Loss: 51.4560

Mode Generator Loss: 10.7978

Total Epochs: 288
Total Training Time: 3:30:52
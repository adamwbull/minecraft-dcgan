# BuildLogger

## Introduction

BuildLogger connects with WorldEdit to allow production of 3D integer matrices representing Minecraft builds, which are saved in a SQLite database with user-prescribed labels. 

These matrices will eventually be used to train prompt-based AI that builds what you ask it to!

![how plugin works](/img/main.png)

## Commands

### /dataset stats

View the number of entries in the server's database, and the top 10 most common labels. 

What else could we add here that could be useful?

![output of /dataset stats](/img/stats.png)

### /dataset add (labels)

Use a WorldEdit Wand *(accessible with **//wand**)* or **//pos1** and **//pos2** to define a region. 

![output of /dataset add](/img/add.gif)

Capture whatever you want within the region! It could be a house, a tree, a bench, a piece of architecture on a larger build like a fancy arch, or even an entire castle!

Next, run this command and add a list a minimum of 5 but as many labels as you wish. Put your most accurate descriptors first, and more obscure ones after. Here are some examples:

![Modern House](/img/modern.jpg)

`/dataset add house, modern, high ceilings, in-ground pool, three floors, large`

![Castle Arch](/img/castlearch.png)

`/dataset add outpost, guard gate, stone bridge, castle, two towers, large`

![Watering Well](/img/well.png)

`/dataset add water well, water hole, covered, bucket, stone, small`

When capturing larger builds, use more labels to better describe what encapsulates the space, or split up the build into smaller dataset entries.

Think about small scenery objects too! Objects such as trees, flower paths, lampposts, etc. are builds that could be described in 5 labels. 

### /dataset view (id)

View the dimensions and label data for an entry.

![output of /dataset view](/img/view.png)

### /dataset paste (id)

Paste a dataset entry at your current position. 

**Please be careful with this command as it can crash the server if you paste something too large!**

![output of /dataset paste](/img/paste.gif)

### /dataset help

View all other commands and explanations.

![output of /dataset help](/img/help.png)

## Preprocessing Features

### Remove unnecessary air blocks when saving regions.
For a given matrix represented by X width, Y height, and Z length, for every dimension **D** we find a **D<sub>min</sub>** and **D<sub>max</sub>** representing the first and last non-air block in the dimension. We create a new matrix with these mins and maxes as bounds.

![output of /dataset add with shrink](/img/shrink.png)

### Block Simplification (TODO)
There are **973 blocks** natively tracked as integers in our matrices. If we were to allow all possible blocks into training, it would significantly increase training time.

We may want to explore some way of extracting features from the integers representing blocks and saving numbers with new meaning that we extrapolate back to later.
Or, simplify large sections of blocks to a similar block, like considering all wood types to be oak.

No matter what, our database should store both a "pure" matrix with 973 blocks of representation and then potentially other simplified ones.

### ... Other methods (TODO)

Data will have more preprocessing done within Python when training the model, but we can also explore other options here.
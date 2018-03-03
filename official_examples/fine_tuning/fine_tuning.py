"""
In this example, we'll explore a common approach that is particularly useful in 
real-world applications: take a pre-trained Caffe network and fine-tune the parameters 
on your custom data.

The advantage of this approach is that, since pre-trained networks are learned on a 
large set of images, the intermediate layers capture the "semantics" of the general 
visual appearance. Think of it as a very powerful generic visual feature that you can 
treat as a black box. On top of that, only a relatively small amount of data is needed 
for good performance on the target task.
"""
# GSoC-22-Project
![](https://github.com/aylinaydincs/GSoC-22-Project/blob/main/Photos/GSOC.jpg)


This repository contains the project where I contributed to the TensorFlow Team during GSoC in the year 2022.
#### Google Summer of Code Proposal
[Proposal](https://github.com/aylinaydincs/GSoC-22-Project/blob/main/Proposal/proposal.pdf)


## ArcFace Loss
![](https://github.com/aylinaydincs/GSoC-22-Project/blob/main/Photos/architecture.jpg)

>Additive Angular Margin Loss (ArcFace) has a clear geometric interpretation due to the exact correspondence to the geodesic distance on the hypersphere, and consistently outperforms the state-of-the-art and can be easily implemented with negligible computational overhead.


### Standalone usage:

```javascript

    labels = tf.Variable([1, 0])
    embeddings = tf.Variable([[0.2, 0.3, 0.1], [0.4, 0.5, 0.5]])
    
    #create loss according to your data
    loss_fn = tfsim.losses.ArcFaceLoss(num_classes=2, embedding_size=3)
    loss = loss_fn(labels, embeddings)

```

### Usage with model.compile():

```javascript
    #extract the required variable
    num_classes = np.unique(y_train).size
    embedding_size = model.get_layer('metric_embedding').output.shape[1]
    
    #create loss according to your data
    loss = tfsim.losses.ArcFaceLoss(num_classes= num_classes, embedding_size=embedding_size, name="ArcFaceLoss")
    
    #compile your model with your loss
    model.compile(optimizer=tf.keras.optimizers.SGD(LR), loss=loss, distance=distance)
```

#### Original Paper:
[ArcFace: Additive Angular Margin Loss for Deep Face Recognition.](https://arxiv.org/abs/1801.07698v3)

#### Offical Implementation of Paper: 
[MXNet](https://github.com/deepinsight/insightface)

## Medium Stories of Project
[Face Recognition and ArcFace Loss for Beginners](https://medium.com/@aylin.aydin/face-recognition-and-arcface-loss-for-beginners-cdfddbf7e88)

## License
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

- Special thanks to Yusuf Sarıgoz [@monatis](https://github.com/monatis) for being perfect mentor.

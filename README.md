# Pytorch implementation of ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
# Under construction
ENet (efficient neural network) is a deep learning architecture created specifically for tasks requiring low latency operation. ENet is up to 18×faster, requires 75× less FLOPs, has 79× less parameters, and provides similar or better accuracy to existing models. The implementation is based on the original [implementation] by the authors in lua. This is just a starting commit with  a lot of TODOS. The idea is to start with ENEt and create all the semantic segmentation models(UNet, deeplabV3) in one place. Also in future create a docker environment and the ability to train on custom datasets and deploy on Kuberetes/KubeFlow. As this is a persoanl project so this will take some time with the implementation.


### Todos

 - Create the encoder-decoder in the ENET architecture.



   [implementation]: <https://github.com/e-lab/ENet-training>
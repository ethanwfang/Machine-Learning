## Independent Subnet Training

In IST, the original network is decomposed into a set of narrow subnetworks with the same depth. These subnetworks are then trained locally before parameters are exchanged to produce new subnets and the training cycle repeats. 

This only stores a portion of network parameters on each device. So this means that subnet training is local and independent, and communication volume is reduced. 

IST is then advantageous against issues such as distributed data, slow interconnects, or limited device memory. This makes IST a suitable approach for cases of mandatory distribution. 

Example: GPT-3 takes thousands of years of GPU time to train. You can speed this up by using a ton of high speed GPUs to lower the time to weeks or months. In such an extensive training scenario, different sites or compute units are typically conneceted with a high-speed network, and the hardware is often carefully trailored to the task of distributed learning.

The authors argue that common methods of distributing ML computations cannot be expected to handle such non-ideal environments gracefully. 

The authors argue that IST is most benefitial for training networks with fully-connected layers in casess of mandatory distribution, where training is highly-distributed and hardware is less-than-ideal. 
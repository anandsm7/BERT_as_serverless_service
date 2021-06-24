![baas](img/baas.png)

<h3 style="text-align:center; font-weight: bold">
 Serve PyTorch model as an API using AWS + serverless framework
</h3>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


**BAAS** - BERT As A Service, here we will be deploying a trained BERT model using serverless infrastructure.
The above BERT model is used to classify user daily transaction logs to the following classes.
> Food, Transport, Bills, Shopping. Credit


## Model building

To train the BERT model [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IAJrx15szXsGDjKx1qihrvzAWqp2exz5?usp=sharing)

Note: upload the processed.csv file, which the training data


## Serving Model using Lambda + serverless framework

### Architecture
![api](img/new_API.png)


## API results
![response](img/resp.png)

## Tutorial

For complete tutorial please read my medium blog

[![tutorial](img/medium.svg)](https://medium.com/geekculture/deploying-pytorch-model-as-a-serverless-service-339b4b93e517)


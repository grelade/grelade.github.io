---
title: "Video captioning in bulk using Hugging Face + torchserve"
lang: en
layout: post
usemathjax: true
---

<a href="{% post_url 2023-11-17-captioning-in-bulk %}">![front](/assets/posts/2023-11-17/front.png)</a>

In the age of big data, grappling with demanding processing tasks is inevitable. When it comes to handling videos, the need for efficient, high-throughput solutions becomes paramount. This project delves into the setup of a multi-GPU pipeline optimized for bulk video captioning. Leveraging the *torchserve* server and the captioning model from *Hugging Face*, this project aims to enhance performance. [Code provided](https://github.com/grelade/vidcaption-ml).



---

## ML in 2023

In the era of foundational ML models, it is extremely easy to setup your own ML pipeline. With the help of a model repository like [*Hugging Face*](http://huggingface.co), the pipeline is up-and-running in a manner of minutes. These developments are perfect for a proof-of-concept or exploratory work. But what when the task is straightforward but laborious? Not feasible on a single notebook/GPU... so maybe some parallelism could help? For sure, but doing that might be not that easy. In this post I describe my approach to the problem of ML processing of a bulk of data.

## Project scope

The idea is to create a simple ML-based captioning tool which takes in videos and outputs descriptive, frame-by-frame captions. The plan is to make it as efficient as possible in order to process a rather large set of videos. A major technical requirement in this project is a physical separation between the captioning server and the storage server. I plan to setup a HTTP communication channel to send the video-frames and receive back the captions. I identify two approaches:

* a simple solution using a single-gpu HTTP *flask* captioning server and simple POST requests on the storage/client side.
* a multi-gpu solution using *torchserve* and asynchronous requests.

I focus on the second approach as much more interesting. I pick a [BLIP2 + OPT](https://huggingface.co/Salesforce/blip2-opt-2.7b) multimodal img2txt model as the captioning tool; it works quite well and requires only 8GB of the GPU VRAM. Via POST HTTP request, the captioning server should receive a batch of video frames saved as a `np.ndarray`  and send back the captions to the storage server. Decision to use *numpy* arrays instead of frame images is backed up by two considerations: a) optimize the extraction process, and b) enable more flexibility in sending prepackaged frame batches (or prebatches). I use the *torchserve* server to run the model concurrently on multiple GPUs.

## Captioning server

First I configure and start the captioning server. [*Torchserve*](https://pytorch.org/serve/) is a tool for serving scalable *pytorch* models in production. To this end, we install packages on top of a working *pytorch* environment:
```
pip install torchserve torch-model-archiver
```

To run a model, *torchserve* needs two elements:
- *\*.mar* **model archive file** containing all model artifacts/parameters/configuration files
- **configuration file** *config.properties*

#### Model archive file

`mar` file can be created using the CLI tool [*torch-model-archiver*](https://github.com/pytorch/serve/blob/master/model-archiver/README.md) in two ways, either
* by specifying the *pytorch* *\*.pth* model file and a predefined *torchserve* model handler (a task-aware wrapper class of the model, [find a list of predefined handlers here](https://pytorch.org/serve/default_handlers.html)), or
* by defining a custom *torchserve* handler *ts_handler.py*. 

I focus on creating a custom handler as we need something less standard, predefined CV handlers like `image_classifier` expect single image files as the inference input. I build on the [custom handler tutorial](https://pytorch.org/serve/custom_service.html#custom-handlers) to arrive at the following *ts_handler.py* file:

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from ts.torch_handler.base_handler import BaseHandler
from abc import ABC
import logging
import pickle
import numpy as np

logger = logging.getLogger(__name__)

class BLIP2Handler(BaseHandler, ABC):
    def __init__(self):
        super(BLIP2Handler, self).__init__()
        self._batch_size = 0
        self.initialized = False
        self.model = None
        self.inference_mode = None
        
    def initialize(self, context):
        logger.info(f"Manifest: {context.manifest}")
        logger.info(f"properties: {context.system_properties}")
        
        self._batch_size = context.system_properties["batch_size"]

        if torch.cuda.is_available():
            device = "cuda:" + str(context.system_properties.get("gpu_id")) 
        else:
            device = "cpu"

        self.device = torch.device(device)
        
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", 
                                                        device = self.device)
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                                   torch_dtype=torch.float16)
        self.model.to(self.device)
        
        logger.debug("BLIP2 model loaded successfully")
        self.initialized = True
        
    def preprocess(self, inputs):
        """Very basic preprocessing code - only tokenizes.
        Extend with your own preprocessing steps as needed.
        """
        frames_list = []
        for input_ in inputs:
            input_ = pickle.loads(input_['body'])
            if len(input_.shape)==3:
                self.inference_mode = 'single'
                frames_list.append(input_)
            elif len(input_.shape)==4:
                self.inference_mode = 'batch'
                frames_list.extend(list(input_))
        frames = np.array(frames_list)
        
        logging.info(f"Received data: {len(frames)} of type {type(frames[0])}")
        inputs = self.processor(frames, return_tensors="pt").to(self.device, torch.float16)  

        return inputs

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        if self.inference_mode == 'batch':
            generated_texts = [generated_texts]
        return generated_texts
   
_service = BLIP2Handler()

def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)

        return data
    except Exception as e:
        raise e
```

Given a basic knowledge of ML, most of the code is quite self-explanatory. Internally, *torchserve* calls the `handle` function. I briefly discuss some elements of the service handler:

`BLIP2Handler` class inherits from the `ts.torch_handler.base_handler.BaseHandler` and overrides three key functions:
* `initialize` - initializes the model (`self.preprocess` and `self.model`)
  - local instances running on single GPUs have access to global variables through `context`
  - `self._batch_size` defines the batch size 
* `preprocess` - data preprocessing 
  - data is passed to preprocessing in batches of size `self._batch_size`
  - depending on the inference mode each datum is unpickled and stacked to a *numpy* array
* `inference` - inference module

Once the *ts_handler.py* is defined, the model archive file *model_store/blip2.mar* is generated by the *server-create.sh* script:

```bash
torch-model-archiver --model-name "blip2" --version 1.0 --handler "ts_handler.py"
mkdir -p model_store & mv blip2.mar model_store
```

#### *torchserve* configuration file

Second element of the deployment is the configuration file *config.properties*. Some of its properties are described [here](https://pytorch.org/serve/configuration.html):

```applescript
model_store=model_store

max_request_size=500000000
inference_address=http://0.0.0.0:12345
management_address=http://0.0.0.0:12346
metrics_address=http://0.0.0.0:12347
grpc_inference_port=7070
grpc_management_port=7071

enable_metrics_api=false
disable_system_metrics=true
number_of_gpu=4

load_models=blip2.mar
models={\
  "blip2": {\
    "1.0": {\
        "defaultVersion": true,\
        "marName": "blip2.mar",\
        "minWorkers": 4,\
        "maxWorkers": 4,\
        "batchSize": 1,\
        "maxBatchDelay": 0,\
        "responseTimeout": 120\
    }\
  }\
}
```

What does the config do? Key variables are discussed:

- `model_store` - directory where the models are stored
- `max_request_size` - maximum HTTP request size 
- `load_models` - which models to load 
- `number_of_gpu`, `models -> minWorkers`, `models -> maxWorkers` - three variables controlling the number of GPUs to use. 
- `models -> batchSize` - single model batch size (the `BLIP2Handler._batch_size` parameter)
- `models -> maxBatchDelay` - how long (in ms) each model waits for the batch 
- `enable_metrics_api,disable_system_metrics` - turning off metrics to increase performance

A peculiar choice of parameters `batchSize=1` and `maxBatchDelay=0` and a huge request size `max_request_size=500MB` is explained in the optimization section.

#### Start/stop the server

To start *torchserve* server, I use the *start.sh* script:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
torchserve --start --ts-config config.properties
```

The *CUDA_VISIBLE_DEVICES* env variable specifies a subset of GPUs used by *torchserve*. To stop the server simply run *stop.sh*:
```bash
torchserve --stop
```

#### Server optimization

To optimize our pipeline, it is important to understand the workings of *torchserve*. Its default mode of operation is suitable for a server handling a lot of small, unrelated queries like multiple users categorizing images. Each request is taken in and the server either sends a full batch of size `batchSize` or an incomplete batch if the `maxBatchDelay` is reached. A single query is typically an image or some text.

Our mode of operation is quite different. We have a single client (the storage server) with numerous, large queries. Therefore, instead of creating a lot of requests with single frames, we prebatch frames on the client side and send them together. We ensure such relatively large requests are received fully by setting the `max_request_size` parameter to 500MB. On the server side, *torchserve* takes in the prebatched data as a single query and in turn calls the inference model handler once. Then, to run the captioning model ASAP, I set the `batch_size` parameter to one and `maxBatchDelay` to zero to ensure the server does not wait for the filling up of the batches.

## Storage server

The storage server is connected via ssh to the captioning server. I provide few clients with different types of video loading and request forming:

* *cli-single.py* - uses sync video loading and async single-frame requests,
* *cli-str-single.py* - uses async stream-like video loading and async single-frame requests,
* *cli-str-prebatch.py* - uses async stream-like video loading and async prebatched requests.



I provide a simple benchmark comparing the clients, the prebatching approach is clearly the best approach.

| client script               | torchserve config file   | capt. speed (frm/s) | video res  | client params |
|:-------------------------------|:-------------------------|:-----------------:|:----------|:--------------|
| cli-single.py     | config-single.properties | 34.6   | 480x270   |               |
| cli-str-single.py    | config-single.properties | 64.5  | 480x270   | |
| **cli-str-prebatch.py** | config.properties        | **140.4**  | 480x270   | prebatch_size = 32 |
| cli-single.py     | config-single.properties | 30.4  | 640x360   | |
| cli-str-single.py    | config-single.properties | 62.9  | 640x360   | |
| **cli-str-prebatch.py** | config.properties        | **125.1**  | 640x360   | prebatch_size = 32 |
| cli-single.py     | config-single.properties | 12.7  | 1280x720  | |
| cli-str-single.py    | config-single.properties | 40.7   | 1280x720  | |
| **cli-str-prebatch.py** | config.properties        | **76.6**  | 1280x720  | prebatch_size = 16 |
| cli-single.py     | config-single.properties | 6.4   | 1920x1080 | |
| cli-str-single.py    | config-single.properties | 29.1  | 1920x1080 | |
| **cli-str-prebatch.py** | config.properties        | **43.2**  | 1920x1080 |  |

Non-standard client parameters in the last column are given. Changing the prebatch size of the client *cli-str-prebatch.py* proved crucial in establishing the optimal throughput. All tests were completed locally i.e. without network latency. A single [rescaled video](https://file-examples.com/index.php/sample-video-files/sample-mp4-files/) with 901 frames was captioned.

## Conclusions and extensions

* Out-of-the-box parallel inference is quite easy to achieve using the *torchserve* server. 
* With few modifications like client-side prebatching, *torchserve* remains useful for non-standard bulk single-user inference with high throughputs.
* It is possible to improve the pipeline considerably by exporting to ONNX and enabling TensorRT technology. 

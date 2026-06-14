# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       28 |        2 |        0 |        0 |     92.86% |   258-260 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       18 |       28 |        9 |     72.12% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 159-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       61 |       16 |       16 |        1 |     77.92% |67, 73-74, 90-93, 106-109, 121-125 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       49 |        6 |       10 |        5 |     81.36% |49, 50-\>60, 52-\>60, 56-57, 59, 101-103 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      275 |       11 |       74 |       11 |     93.12% |106-\>exit, 108-\>exit, 117-\>exit, 207, 266, 282-\>288, 314, 355-362, 460-\>exit, 546, 677-\>690, 681, 761, 768-\>771 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      114 |       13 |       22 |        2 |     88.97% |89, 95-97, 171, 220-232 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |79-\>78, 100, 104 |
| src/async\_batch\_llm/base.py                                |      379 |       39 |      104 |       12 |     87.78% |92, 275-\>exit, 672-673, 702-703, 710-711, 715-725, 736-737, 810, 838, 851, 856-\>exit, 890, 894, 945-961, 1022, 1054-\>exit, 1058, 1060, 1077, 1092-\>exit, 1101-1105, 1122-1124 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       45 |        9 |       18 |        1 |     84.13% |45-47, 55-57, 105-106, 140-141 |
| src/async\_batch\_llm/classifiers/openai.py                  |       82 |       17 |       36 |        3 |     83.05% |59-70, 119-120, 128, 138, 204-205, 277 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       10 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |       90 |        1 |       44 |        2 |     97.76% |205, 242-\>exit |
| src/async\_batch\_llm/core/protocols.py                      |       15 |        4 |        0 |        0 |     73.33% |22, 31, 35, 39 |
| src/async\_batch\_llm/gateway.py                             |       67 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      110 |        8 |       24 |        5 |     90.30% |29-30, 49-\>47, 51, 66-68, 276-\>exit, 286-\>exit, 569-571 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        1 |        0 |        0 |     90.91% |        82 |
| src/async\_batch\_llm/models.py                              |      415 |       26 |      168 |       23 |     91.25% |32-34, 37-38, 125-\>128, 195-\>198, 203, 206, 225, 359-\>362, 364, 370-\>exit, 395, 474-\>477, 478, 481, 493-497, 532-533, 549-\>579, 560, 591, 625, 841-\>845, 843-\>845, 846-\>850, 848-849, 856, 870, 873-\>exit, 1228-\>1233 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       19 |        1 |        0 |        0 |     94.74% |        57 |
| src/async\_batch\_llm/observers/metrics.py                   |       72 |        1 |       26 |        5 |     93.88% |20, 46-\>exit, 57-\>exit, 67-\>exit, 173-\>182 |
| src/async\_batch\_llm/parallel.py                            |      256 |       28 |       64 |       10 |     88.12% |70-71, 74-75, 139, 146, 153, 179, 181, 183, 187, 278, 282, 293, 297, 301, 305, 309, 373-375, 445-\>448, 468, 473, 478, 549-\>556, 608-611, 645, 649-650 |
| src/async\_batch\_llm/parsing.py                             |       21 |        0 |        6 |        0 |    100.00% |           |
| src/async\_batch\_llm/single.py                              |       27 |        3 |        4 |        1 |     87.10% | 52-53, 67 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |       64 |        2 |       14 |        0 |     97.44% |   250-251 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |       55 |        0 |       20 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |       60 |        3 |       28 |        6 |     89.77% |51-\>57, 53-\>57, 61-\>65, 66-\>83, 108, 126-\>131, 142-143 |
| **TOTAL**                                                    | **2487** |  **211** |  **734** |   **97** | **90.00%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/geoff-davis/async-batch-llm/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/geoff-davis/async-batch-llm/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeoff-davis%2Fasync-batch-llm%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.
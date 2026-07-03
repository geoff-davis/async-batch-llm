# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       29 |        2 |        0 |        0 |     93.10% |   273-275 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       18 |       28 |        9 |     72.12% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 159-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       61 |        7 |       16 |        1 |     89.61% |70, 76-77, 94, 110, 124, 126 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       49 |        6 |       10 |        5 |     81.36% |49, 50-\>60, 52-\>60, 56-57, 59, 107-109 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      275 |       11 |       74 |       11 |     93.12% |106-\>exit, 108-\>exit, 117-\>exit, 207, 267, 286-\>292, 318, 359-366, 464-\>exit, 550, 682-\>695, 686, 766, 773-\>776 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      117 |       13 |       24 |        2 |     89.36% |93, 99-101, 175, 233-245 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |79-\>78, 100, 104 |
| src/async\_batch\_llm/base.py                                |      389 |       39 |      106 |       12 |     88.08% |103, 298-\>exit, 727-728, 757-758, 765-766, 770-780, 791-792, 865, 893, 906, 911-\>exit, 945, 949, 1000-1016, 1077, 1109-\>exit, 1113, 1115, 1132, 1147-\>exit, 1156-1160, 1177-1179 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       59 |        6 |       28 |        2 |     90.80% |75-76, 110-111, 207, 215 |
| src/async\_batch\_llm/classifiers/openai.py                  |       62 |        7 |       32 |        3 |     89.36% |89-90, 98, 108, 174-175, 247 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       19 |        0 |        8 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |       89 |        1 |       44 |        2 |     97.74% |231, 246-\>exit |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/gateway.py                             |       67 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      116 |        8 |       26 |        5 |     90.85% |30-31, 65-\>63, 67, 82-84, 292-\>exit, 302-\>exit, 585-587 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        1 |        0 |        0 |     90.91% |        98 |
| src/async\_batch\_llm/models.py                              |      519 |       31 |      212 |       26 |     91.93% |40-42, 45-46, 256-\>259, 336-\>339, 344, 347, 367, 512-\>515, 517, 523-\>exit, 548, 627-\>630, 631, 634, 647-651, 687-688, 704-\>734, 715, 746, 780, 1038-\>1087, 1041-\>1044, 1045-\>1081, 1061-1062, 1067, 1088-\>1092, 1090-1091, 1098, 1112, 1115-\>exit, 1313-1314, 1503-\>1508 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       19 |        1 |        0 |        0 |     94.74% |        57 |
| src/async\_batch\_llm/observers/metrics.py                   |       75 |        1 |       26 |        5 |     94.06% |33, 49-\>exit, 60-\>exit, 70-\>exit, 172-\>181 |
| src/async\_batch\_llm/parallel.py                            |      256 |       27 |       64 |        9 |     88.75% |70-71, 74-75, 139, 146, 179, 181, 183, 187, 279, 283, 294, 298, 302, 306, 310, 374-376, 446-\>449, 469, 474, 479, 550-\>557, 609-612, 646, 650-651 |
| src/async\_batch\_llm/parsing.py                             |       21 |        0 |        6 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |       82 |        0 |       22 |        0 |    100.00% |           |
| src/async\_batch\_llm/single.py                              |       27 |        3 |        4 |        1 |     87.10% | 52-53, 67 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |       95 |       12 |       20 |        1 |     88.70% |62-73, 130-\>exit, 329-330 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |       55 |        0 |       20 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |       63 |        3 |       30 |        5 |     91.40% |51-\>61, 68-\>72, 76-\>89, 114, 132-\>135, 151-152 |
| **TOTAL**                                                    | **2719** |  **199** |  **826** |  **100** | **91.17%** |           |


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
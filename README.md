# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       28 |        2 |        0 |        0 |     92.86% |   266-268 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       18 |       28 |        9 |     72.12% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 159-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       61 |        7 |       16 |        1 |     89.61% |70, 76-77, 94, 110, 124, 126 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       49 |        6 |       10 |        5 |     81.36% |49, 50-\>60, 52-\>60, 56-57, 59, 107-109 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      275 |       11 |       74 |       11 |     93.12% |106-\>exit, 108-\>exit, 117-\>exit, 207, 267, 286-\>292, 318, 359-366, 464-\>exit, 550, 682-\>695, 686, 766, 773-\>776 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      117 |       13 |       24 |        2 |     89.36% |93, 99-101, 175, 233-245 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |79-\>78, 100, 104 |
| src/async\_batch\_llm/base.py                                |      386 |       39 |      104 |       12 |     87.96% |100, 287-\>exit, 712-713, 742-743, 750-751, 755-765, 776-777, 850, 878, 891, 896-\>exit, 930, 934, 985-1001, 1062, 1094-\>exit, 1098, 1100, 1117, 1132-\>exit, 1141-1145, 1162-1164 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       59 |        6 |       28 |        2 |     90.80% |75-76, 110-111, 207, 215 |
| src/async\_batch\_llm/classifiers/openai.py                  |       62 |        7 |       32 |        3 |     89.36% |89-90, 98, 108, 174-175, 247 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       19 |        0 |        8 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |       86 |        1 |       42 |        2 |     97.66% |218, 233-\>exit |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/gateway.py                             |       67 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      116 |        8 |       26 |        5 |     90.85% |30-31, 65-\>63, 67, 82-84, 292-\>exit, 302-\>exit, 585-587 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        1 |        0 |        0 |     90.91% |        98 |
| src/async\_batch\_llm/models.py                              |      483 |       28 |      192 |       24 |     92.00% |40-42, 45-46, 233-\>236, 313-\>316, 321, 324, 344, 489-\>492, 494, 500-\>exit, 525, 604-\>607, 608, 611, 624-628, 664-665, 681-\>711, 692, 723, 757, 1007-\>1011, 1009-\>1011, 1012-\>1016, 1014-1015, 1022, 1036, 1039-\>exit, 1237-1238, 1427-\>1432 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       19 |        1 |        0 |        0 |     94.74% |        57 |
| src/async\_batch\_llm/observers/metrics.py                   |       75 |        1 |       26 |        5 |     94.06% |33, 49-\>exit, 60-\>exit, 70-\>exit, 172-\>181 |
| src/async\_batch\_llm/parallel.py                            |      256 |       28 |       64 |       10 |     88.12% |70-71, 74-75, 139, 146, 153, 179, 181, 183, 187, 279, 283, 294, 298, 302, 306, 310, 374-376, 446-\>449, 469, 474, 479, 550-\>557, 609-612, 646, 650-651 |
| src/async\_batch\_llm/parsing.py                             |       21 |        0 |        6 |        0 |    100.00% |           |
| src/async\_batch\_llm/single.py                              |       27 |        3 |        4 |        1 |     87.10% | 52-53, 67 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |       95 |       12 |       20 |        1 |     88.70% |62-73, 130-\>exit, 329-330 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |       55 |        0 |       20 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |       63 |        3 |       30 |        5 |     91.40% |51-\>61, 68-\>72, 76-\>89, 114, 132-\>135, 151-152 |
| **TOTAL**                                                    | **2594** |  **197** |  **780** |   **99** | **90.81%** |           |


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
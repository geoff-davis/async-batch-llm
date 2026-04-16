# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       24 |        2 |        0 |        0 |     91.67% |   207-209 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       31 |       28 |        7 |     53.85% |36-\>39, 90, 95-129, 146-\>167, 148, 150-153, 159-165 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       59 |       16 |       14 |        1 |     76.71% |57, 63-64, 80-83, 96-99, 111-115 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      109 |       12 |       18 |        1 |     89.76% |86-88, 149, 187-199 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |82-\>81, 105, 109 |
| src/async\_batch\_llm/base.py                                |      229 |       36 |       48 |        7 |     81.59% |90, 401-402, 416, 424-425, 429-433, 437-447, 470, 474, 519-535, 588-\>exit, 593, 610, 625-\>exit, 634-638, 655-657 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       44 |       12 |       16 |        2 |     76.67% |34-36, 44-46, 55-57, 75-76, 111-112 |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |       86 |        1 |       42 |        2 |     97.66% |164, 201-\>exit |
| src/async\_batch\_llm/core/protocols.py                      |       15 |        4 |        0 |        0 |     73.33% |22, 31, 35, 39 |
| src/async\_batch\_llm/llm\_strategies.py                     |       78 |       10 |       12 |        3 |     85.56% |28-29, 234-\>exit, 259-261, 326-328, 352-354 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        1 |        0 |        0 |     90.91% |        82 |
| src/async\_batch\_llm/models.py                              |      197 |       19 |       78 |       15 |     86.91% |24-26, 82-\>85, 157, 160, 179, 311-\>314, 316, 322-\>exit, 347, 408, 428, 431, 443-447, 476, 494, 516 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       18 |        1 |        0 |        0 |     94.44% |        51 |
| src/async\_batch\_llm/observers/metrics.py                   |       71 |        1 |       26 |        5 |     93.81% |16, 42-\>exit, 53-\>exit, 63-\>exit, 169-\>178 |
| src/async\_batch\_llm/parallel.py                            |      350 |       35 |       96 |       19 |     87.44% |93, 100, 107, 126, 128, 130, 205, 209, 220, 224, 228, 232, 236, 303-\>306, 346-348, 368, 392, 419-\>426, 457-458, 479-482, 511, 551, 574-578, 592-\>594, 622-\>631, 650, 705, 720-722, 811-\>823, 815, 892, 899-\>902, 940 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |       48 |        2 |       12 |        0 |     96.67% |   177-178 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |       47 |        2 |       20 |        4 |     91.04% |51-\>57, 53-\>57, 61-\>65, 66-\>83, 110-111 |
| **TOTAL**                                                    | **1546** |  **187** |  **422** |   **67** | **85.98%** |           |


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
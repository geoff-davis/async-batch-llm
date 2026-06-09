# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       25 |        2 |        0 |        0 |     92.00% |   240-242 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       31 |       28 |        7 |     53.85% |36-\>39, 87, 92-126, 143-\>164, 145, 147-150, 156-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       59 |       16 |       14 |        1 |     76.71% |57, 63-64, 80-83, 96-99, 111-115 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      112 |       12 |       20 |        1 |     90.15% |84-86, 160, 209-221 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |79-\>78, 100, 104 |
| src/async\_batch\_llm/base.py                                |      261 |       36 |       64 |        8 |     84.00% |90, 263-\>exit, 558-559, 573, 581-582, 586-590, 594-604, 627, 631, 678-694, 746-\>exit, 751, 768, 783-\>exit, 792-796, 813-815 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       46 |        9 |       18 |        1 |     84.38% |36-38, 46-48, 90-91, 125-126 |
| src/async\_batch\_llm/classifiers/openai.py                  |       83 |       17 |       36 |        3 |     83.19% |54-65, 111-112, 120, 130, 196-197, 269 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       10 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |       86 |        1 |       42 |        2 |     97.66% |164, 201-\>exit |
| src/async\_batch\_llm/core/protocols.py                      |       15 |        4 |        0 |        0 |     73.33% |22, 31, 35, 39 |
| src/async\_batch\_llm/llm\_strategies.py                     |       82 |        7 |       14 |        3 |     89.58% |28-29, 49-51, 236-\>exit, 245-\>exit, 482-484 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        1 |        0 |        0 |     90.91% |        82 |
| src/async\_batch\_llm/models.py                              |      388 |       24 |      160 |       22 |     91.24% |32-34, 37-38, 125-\>128, 195-\>198, 203, 206, 225, 359-\>362, 364, 370-\>exit, 395, 451, 478-\>481, 482, 485, 497-501, 530, 554, 770-\>774, 772-\>774, 775-\>779, 777-778, 785, 799, 802-\>exit, 1157-\>1162 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       18 |        1 |        0 |        0 |     94.44% |        51 |
| src/async\_batch\_llm/observers/metrics.py                   |       71 |        1 |       26 |        5 |     93.81% |16, 42-\>exit, 53-\>exit, 63-\>exit, 169-\>178 |
| src/async\_batch\_llm/parallel.py                            |      358 |       39 |      100 |       19 |     86.46% |97, 104, 111, 130, 132, 134, 209, 213, 224, 228, 232, 236, 240, 312-\>315, 355-357, 377, 401, 428-\>435, 466-467, 488-491, 525, 565, 591-599, 613-\>615, 646-\>655, 673, 728, 743-747, 840-\>852, 844, 923, 930-\>933, 971 |
| src/async\_batch\_llm/parsing.py                             |       21 |        0 |        6 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |       48 |        2 |       12 |        0 |     96.67% |   193-194 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |       54 |        2 |       24 |        5 |     91.03% |51-\>57, 53-\>57, 61-\>65, 66-\>83, 119-\>124, 135-136 |
| **TOTAL**                                                    | **1910** |  **207** |  **578** |   **78** | **87.58%** |           |


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
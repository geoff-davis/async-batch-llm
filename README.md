# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                              |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|-------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py             |       22 |        2 |        0 |        0 |     90.91% |   196-198 |
| src/async\_batch\_llm/base.py                     |      214 |       38 |       42 |        9 |     78.52% |90, 154, 159, 349-350, 364, 372-373, 377-381, 385-395, 418, 422, 467-483, 536->exit, 541, 558, 573->exit, 582-586, 603-605 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py       |       44 |       12 |       16 |        2 |     76.67% |34-36, 44-46, 55-57, 75-76, 111-112 |
| src/async\_batch\_llm/core/\_\_init\_\_.py        |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py              |       83 |        1 |       40 |        2 |     97.56% |156, 193->exit |
| src/async\_batch\_llm/core/protocols.py           |       14 |        4 |        0 |        0 |     71.43% |17, 26, 30, 34 |
| src/async\_batch\_llm/llm\_strategies.py          |      268 |       49 |       78 |       16 |     78.90% |26-29, 33-34, 339-341, 351->358, 353->358, 355-356, 516-518, 558, 584, 616-619, 628-660, 706-708, 712->718, 747, 759-768, 775->782, 777->782, 779-780, 809->exit, 830->exit, 900-902, 926-928 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py  |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py          |       11 |        1 |        0 |        0 |     90.91% |        82 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py           |       18 |        1 |        0 |        0 |     94.44% |        51 |
| src/async\_batch\_llm/observers/metrics.py        |       71 |        1 |       26 |        5 |     93.81% |16, 42->exit, 53->exit, 63->exit, 169->178 |
| src/async\_batch\_llm/parallel.py                 |      545 |       94 |      186 |       35 |     80.16% |87, 94, 101, 120, 122, 124, 192, 196->195, 299->302, 328, 333-334, 347-350, 364-367, 378-382, 393-395, 421-422, 431, 455, 482->489, 519-520, 541-544, 610, 646-657, 712->722, 714->722, 723-727, 740->752, 743-747, 783, 802->816, 817-821, 835->837, 865->873, 892, 947, 962-964, 1054->1066, 1058, 1135, 1142->1145, 1189->1194, 1237->1264, 1240, 1242-1245, 1255-1261, 1285, 1289-1318 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py  |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py        |       42 |        2 |       12 |        0 |     96.30% |   171-172 |
| src/async\_batch\_llm/strategies/rate\_limit.py   |       31 |        0 |        2 |        0 |    100.00% |           |
|                                         **TOTAL** | **1376** |  **205** |  **402** |   **69** | **82.79%** |           |


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
# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                       |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/batch\_llm/\_\_init\_\_.py             |       14 |        2 |        0 |        0 |     85.71% |   135-137 |
| src/batch\_llm/base.py                     |      214 |       38 |       42 |        9 |     78.52% |90, 154, 159, 349-350, 364, 372-373, 377-381, 385-395, 418, 422, 467-483, 536->exit, 541, 558, 573->exit, 582-586, 603-605 |
| src/batch\_llm/classifiers/\_\_init\_\_.py |        2 |        0 |        0 |        0 |    100.00% |           |
| src/batch\_llm/classifiers/gemini.py       |       44 |       12 |       16 |        2 |     76.67% |34-36, 44-46, 55-57, 75-76, 111-112 |
| src/batch\_llm/core/\_\_init\_\_.py        |        3 |        0 |        0 |        0 |    100.00% |           |
| src/batch\_llm/core/config.py              |       77 |       11 |       40 |       12 |     80.34% |22, 27, 32, 37, 56, 61, 66, 72, 117, 122, 144, 181->exit |
| src/batch\_llm/core/protocols.py           |       14 |        4 |        0 |        0 |     71.43% |17, 26, 30, 34 |
| src/batch\_llm/llm\_strategies.py          |      281 |      135 |       84 |       19 |     46.30% |26-29, 33-34, 215-221, 251, 272-288, 324, 335-344, 348-358, 442, 449, 458-460, 498-514, 544-546, 556, 585-626, 645-648, 657-689, 735-737, 741->747, 748, 776, 788-797, 801-811, 838->exit, 859->exit, 865-866, 891, 925-934, 941-966 |
| src/batch\_llm/middleware/\_\_init\_\_.py  |        2 |        0 |        0 |        0 |    100.00% |           |
| src/batch\_llm/middleware/base.py          |       11 |        1 |        0 |        0 |     90.91% |        82 |
| src/batch\_llm/observers/\_\_init\_\_.py   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/batch\_llm/observers/base.py           |       18 |        1 |        0 |        0 |     94.44% |        51 |
| src/batch\_llm/observers/metrics.py        |       71 |        1 |       26 |        5 |     93.81% |16, 42->exit, 53->exit, 63->exit, 169->178 |
| src/batch\_llm/parallel.py                 |      545 |       94 |      186 |       35 |     80.16% |87, 94, 101, 120, 122, 124, 192, 196->195, 299->302, 328, 333-334, 347-350, 362-365, 376-380, 391-393, 419-420, 429, 453, 480->487, 517-518, 539-542, 608, 644-655, 710->720, 712->720, 721-725, 738->750, 741-745, 781, 800->814, 815-819, 833->835, 863->871, 890, 945, 960-962, 1052->1064, 1056, 1133, 1140->1143, 1187->1192, 1235->1262, 1238, 1240-1243, 1253-1259, 1283, 1287-1316 |
| src/batch\_llm/strategies/\_\_init\_\_.py  |        3 |        0 |        0 |        0 |    100.00% |           |
| src/batch\_llm/strategies/errors.py        |       42 |        4 |       12 |        0 |     92.59% |42-43, 171-172 |
| src/batch\_llm/strategies/rate\_limit.py   |       31 |        0 |        2 |        0 |    100.00% |           |
|                                  **TOTAL** | **1375** |  **303** |  **408** |   **82** | **74.71%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/geoff-davis/batch-llm/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/geoff-davis/batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/geoff-davis/batch-llm/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/geoff-davis/batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeoff-davis%2Fbatch-llm%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/geoff-davis/batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.
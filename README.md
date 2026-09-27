# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       36 |        2 |        0 |        0 |     94.44% |   357-359 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/admission.py                |      411 |       17 |      116 |       14 |     94.12% |102, 106, 114, 131, 133, 135, 155, 171, 173, 261, 293-294, 378, 392, 397-398, 509-\>507, 584 |
| src/async\_batch\_llm/\_internal/artifact\_codec.py          |      150 |       13 |       36 |        6 |     89.78% |56, 60, 64, 93, 127, 293, 319, 327, 372, 388-391 |
| src/async\_batch\_llm/\_internal/capacity.py                 |      131 |       11 |       32 |        2 |     90.80% |43-44, 84-93, 159-160, 187-193 |
| src/async\_batch\_llm/\_internal/classifier\_resolver.py     |       38 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/cleanup.py                  |      261 |        8 |       94 |        7 |     94.65% |142-\>exit, 148, 175, 295-\>297, 301-\>exit, 382, 415-\>428, 469-473 |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       16 |       28 |        8 |     75.00% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 161-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       87 |        4 |       26 |        1 |     95.58% |79-80, 119-\>121, 144, 160 |
| src/async\_batch\_llm/\_internal/execution\_state.py         |       57 |        1 |        6 |        1 |     96.83% |       123 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       60 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/guardrails.py               |      137 |        9 |       44 |        5 |     91.16% |63, 73, 102, 141-\>exit, 159, 171-175 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      688 |       22 |      208 |       33 |     93.86% |222-\>exit, 224-\>exit, 228-\>exit, 237-\>exit, 245-\>exit, 273, 289, 358, 418, 464-\>exit, 473-\>475, 476-\>478, 492-495, 552, 578-\>586, 623-\>627, 641-\>exit, 685, 687, 841, 931, 1006-1013, 1052, 1166, 1256-1257, 1260-\>1265, 1325-\>1331, 1348-\>exit, 1410, 1426-\>1430, 1481-\>1485, 1490-\>1494, 1522-\>1527, 1533-\>1558, 1547-\>1558, 1558-\>1563, 1642-\>1649, 1698, 1700-\>1702 |
| src/async\_batch\_llm/\_internal/logical\_item.py            |       16 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      161 |        5 |       38 |        1 |     96.98% |120, 280, 359-365 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |      172 |        5 |       42 |        5 |     95.33% |91, 143-\>141, 218, 245, 251-\>254, 257, 330 |
| src/async\_batch\_llm/artifacts.py                           |      462 |       72 |      150 |       33 |     82.19% |121, 132-\>134, 142, 144-146, 180-181, 242-243, 251-252, 259, 263-264, 283, 290, 296, 300, 305, 316, 318-321, 323, 346, 351, 366, 371-376, 382, 388, 405-406, 410, 466-467, 470-471, 473, 509, 527-528, 575-576, 601, 654-659, 668, 679, 685-\>684, 711, 714, 750-751, 760, 774, 785, 791-\>803, 796-797, 825, 832-835, 846-847, 856-857, 863 |
| src/async\_batch\_llm/base.py                                |      906 |       47 |      244 |       23 |     93.22% |119, 334, 343, 420-\>exit, 709-\>exit, 712-\>exit, 763, 800-\>802, 855-\>860, 1137, 1139, 1472, 1514-1515, 1671, 1709, 1761-1766, 1791, 1806-1809, 1830-\>exit, 1867-1868, 1898, 1902, 1941-\>exit, 1952, 1975, 1977-1981, 2003-2011, 2105, 2154, 2173, 2177-\>2183, 2179-2182, 2213-2214, 2229-2231 |
| src/async\_batch\_llm/callable\_strategy.py                  |      130 |        4 |       44 |        6 |     94.25% |51-\>exit, 64-\>exit, 83, 119, 160, 219 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        5 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       92 |        9 |       48 |        5 |     90.00% |23-\>26, 37, 51-\>48, 56-57, 111-112, 148-149, 227, 235 |
| src/async\_batch\_llm/classifiers/openai.py                  |       76 |        7 |       40 |        1 |     93.10% |85-86, 104, 145-146, 180-181 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       32 |        3 |       12 |        1 |     90.91% | 71-72, 76 |
| src/async\_batch\_llm/classifiers/pydantic\_ai.py            |       16 |        2 |        6 |        0 |     90.91% |     16-17 |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |      184 |        3 |       92 |        3 |     97.83% |205, 440, 448 |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/factory.py                             |       62 |        0 |       28 |        2 |     97.78% |100-\>exit, 111-\>exit |
| src/async\_batch\_llm/gateway.py                             |       79 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      200 |       13 |       50 |        9 |     91.20% |36-37, 71-\>69, 73, 88-90, 282-283, 335-\>exit, 347-\>exit, 432, 450, 463-\>466, 634-\>636, 639, 772-774 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/models.py                              |      957 |      116 |      410 |       67 |     84.56% |52-54, 57-58, 108-124, 350-\>353, 422-\>exit, 430, 433, 470-\>473, 478, 481, 501, 668-\>671, 673, 679-\>exit, 704, 784-\>787, 788, 791, 834-838, 862, 873, 899-900, 916-\>950, 927, 929, 965, 981-\>999, 1020-\>1025, 1282-\>1331, 1285-\>1288, 1289-\>1325, 1305-1306, 1311, 1332-\>1336, 1334-1335, 1346, 1399-\>1409, 1402-\>1409, 1404-1405, 1425, 1428, 1431-\>1439, 1657-1658, 1744, 1767, 1777-1778, 1781, 1793-\>1802, 1797-1801, 1807-1815, 1822, 1895, 1904, 1908-1909, 1943, 1946, 1949-1952, 2029-\>2031, 2037-2038, 2075, 2077, 2111-\>2115, 2141, 2147-2160, 2162-2164, 2171-2179, 2181, 2196-2197, 2207, 2224-\>2226, 2227-\>2229, 2230-\>2232, 2244-2247, 2249-2253, 2261, 2263, 2276-\>2281 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       25 |        1 |        0 |        0 |     96.00% |        63 |
| src/async\_batch\_llm/observers/metrics.py                   |      139 |       12 |       70 |       14 |     85.65% |55, 72-\>79, 82-\>88, 89-\>exit, 114-\>108, 117-\>exit, 128, 134-135, 141, 142-\>147, 152, 155-160, 163, 169-\>exit |
| src/async\_batch\_llm/parallel.py                            |      415 |       27 |      110 |       10 |     92.95% |103-104, 107-108, 175, 215, 325, 336, 340, 348, 352, 356, 360, 364, 376, 483, 507-510, 548-549, 591-\>594, 657, 662, 757, 824, 881-\>886, 910-\>915, 962 |
| src/async\_batch\_llm/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |      102 |        1 |       28 |        1 |     98.46% |       225 |
| src/async\_batch\_llm/serialization.py                       |      249 |       41 |      102 |       23 |     81.20% |113, 116, 130, 139-140, 170, 222-223, 227, 252, 258, 268, 270, 350-351, 366, 376-377, 386, 388, 402, 473, 480, 487, 492, 510, 527, 563, 594-597, 649-650, 661-662, 668, 671-672, 684, 695 |
| src/async\_batch\_llm/single.py                              |       32 |        3 |        4 |        1 |     88.89% | 52-53, 67 |
| src/async\_batch\_llm/sqlite\_artifacts.py                   |      692 |       87 |      206 |       40 |     84.52% |86-87, 124-\>exit, 132, 137, 212-213, 215, 223-224, 265-266, 291, 301, 333-334, 366-371, 376, 380, 410, 419-420, 477-480, 500, 532, 551-554, 557-558, 560-\>567, 563-564, 575, 604, 612, 626, 632, 638-639, 661, 663-\>exit, 677-\>679, 680, 686-\>688, 697, 705-711, 748-749, 760-\>759, 775-\>774, 789-790, 802, 812-814, 830-\>845, 859, 879, 896-899, 903-905, 998-1000, 1016, 1083, 1090, 1093, 1097, 1108, 1139, 1235, 1301-1302, 1348-1349, 1357-1358, 1361-1362, 1366 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |      168 |       16 |       50 |        2 |     91.74% |63-64, 71-82, 202, 221, 427-428 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |      199 |        1 |       70 |        4 |     98.14% |65-\>exit, 197-\>199, 253-\>256, 330, 336-\>exit |
| src/async\_batch\_llm/token\_estimation.py                   |       35 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |      128 |        4 |       50 |        5 |     94.94% |85-\>95, 98-\>104, 100-\>104, 143-144, 151, 198-\>200, 214 |
| **TOTAL**                                                    | **7984** |  **582** | **2528** |  **333** | **90.65%** |           |


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
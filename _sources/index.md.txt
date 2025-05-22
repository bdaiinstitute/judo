.. image:: _static/images/logo-light.svg
   :width: 400
   :align: center
   :alt: judo
   :class: only-light

.. image:: _static/images/logo-dark.svg
   :width: 400
   :align: center
   :alt: judo
   :class: only-dark


# judo

<!-- prettier-ignore-start -->

.. toctree::
   :caption: API Reference
   :maxdepth: 10
   :hidden:
   :titlesonly:
   :glob:


   api/common/index
   api/learning/index
   api/planner/index
   api/visualizers/index


.. toctree::
   :caption: Documentation
   :hidden:
   :maxdepth: 1
   :titlesonly:

   docs


|build| |nbsp| |docs| |nbsp| |coverage|

.. |build| image:: https://github.com/bdaiinstitute/judo/actions/workflows/build.yml/badge.svg
   :alt: Build status icon
   :target: https://github.com/bdaiinstitute/judo
.. |docs| image:: https://github.com/bdaiinstitute/judo/actions/workflows/docs.yml/badge.svg
   :alt: Docs status icon
   :target: https://github.com/bdaiinstitute/judo
.. |coverage| image:: https://codecov.io/gh/bdaiinstitute/judo/graph/badge.svg?token=7A0OJ37JRF
   :alt: Test coverage status icon
   :target: https://github.com/bdaiinstitute/judo
.. |nbsp| unicode:: 0xA0
   :trim:


<!-- prettier-ignore-end -->

## Jacta: A Versatile Planner for Learning Dexterous and Whole-body Manipulation

[![Static Badge](https://img.shields.io/badge/ArXiv-8C48FC?style=for-the-badge)](https://arxiv.org/pdf/2408.01258)
[![Static Badge](https://img.shields.io/badge/Project_Page-8C48FC?style=for-the-badge)](https://judo.github.io/)
[![Static Badge](https://img.shields.io/badge/RAI_Institute-8C48FC?style=for-the-badge)](https://rai-inst.com/resources/papers/jacta-a-versatile-planner-for-learning-dexterous-and-whole-body-manipulation/)


Robotic manipulation is challenging and data-driven approaches typically require large amounts of data or expert demonstrations. Therefore, we introduce a motion planner for dexterous and whole-body manipulation tasks. The planner's demonstrations can directly be used by reinforcement learning. With this approach, we can efficiently learn policies for complex manipulation tasks, where traditional reinforcement learning alone only makes little progress.


![Jacta](_static/images/jacta_overview.jpg)

### Installation
Install cmake
```
sudo apt install cmake
```

Install mujoco_extensions
```
git clone https://github.com/bdaiinstitute/judo.git
cd judo
pip install src/mujoco_extensions -vv
```

Install judo
```
pip install -e .
```


### Getting started
```
python examples/planner/example_notebook.py
```
Open the visualizer in your browser by clicking on the link in the terminal.
```
http://localhost:8080/
```


### Citation
```
@inproceedings{brudigam2024jacta,
  author       = {Br{\"u}digam, Jan and Abbas, Ali-Adeeb and Sorokin, Maks and Fang, Kuan and Hung, Brandon and Guru, Maya and Sosnowski, Stefan and Wang, Jiuguang and Hirche, Sandra and Le Cleac'h, Simon},
  editor       = {Agrawal, Pulkit and Kroemer, Oliver and Burgard, Wolfram},
  title        = {Jacta: {A} Versatile Planner for Learning Dexterous and Whole-body Manipulation},
  booktitle    = {Conference on Robot Learning, 6-9 November 2024, Munich, Germany},
  series       = {Proceedings of Machine Learning Research},
  volume       = {270},
  pages        = {994--1020},
  publisher    = {{PMLR}},
  year         = {2024},
  url          = {https://proceedings.mlr.press/v270/bruedigam25a.html},
}
```

### Run tests locally
In the virtual environment:
```
pip install -e .[dev]
python -m pytest
```
you might have to
```
unset PYTHONPATH
```

.. Optimize documentation master file, created by
   sphinx-quickstart on Sat May 30 15:42:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tutorials
=========

Example 1: Curve fitting with the numpy interface
+++++++++++++++++++++++++++++++++++++++++++++++++

.. literalinclude:: ../examples/gauss_fit_example1_numpy.py
    :language: python

The result ...

.. image:: ../examples/gauss_fit1.png


Example 2: Curve fitting with the parameters interface
++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. literalinclude:: ../examples/gauss_fit_example1_parameters.py
    :language: python

.. image:: ../examples/gauss_fit2.png


Example 3: Curve fitting with uncorrelated known errors (Chi-squared)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. literalinclude:: ../examples/gauss_fit_example2.py
    :language: python


Example 4: Curve fitting with uncorrelated known errors (Chi-squared) with class based API
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. literalinclude:: ../examples/gauss_fit_example3.py
    :language: python


Example 5: Curve fitting with uncorrelated unknown errors (Bayesian)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Here we again use the class based API.

.. literalinclude:: ../examples/gauss_fit_example4.py
    :language: python

.. image:: ../examples/gauss_fit4.png
.. image:: ../examples/corner.png


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

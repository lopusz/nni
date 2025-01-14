
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "tutorials/nni_experiment.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_tutorials_nni_experiment.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorials_nni_experiment.py:


Start and Manage a New Experiment
=================================

.. GENERATED FROM PYTHON SOURCE LINES 7-9

Configure Search Space
----------------------

.. GENERATED FROM PYTHON SOURCE LINES 9-18

.. code-block:: default


    search_space = {
        "C": {"_type": "quniform", "_value": [0.1, 1, 0.1]},
        "kernel": {"_type": "choice", "_value": ["linear", "rbf", "poly", "sigmoid"]},
        "degree": {"_type": "choice", "_value": [1, 2, 3, 4]},
        "gamma": {"_type": "quniform", "_value": [0.01, 0.1, 0.01]},
        "coef0": {"_type": "quniform", "_value": [0.01, 0.1, 0.01]}
    }








.. GENERATED FROM PYTHON SOURCE LINES 19-21

Configure Experiment
--------------------

.. GENERATED FROM PYTHON SOURCE LINES 21-34

.. code-block:: default


    from nni.experiment import Experiment
    experiment = Experiment('local')
    experiment.config.experiment_name = 'Example'
    experiment.config.trial_concurrency = 2
    experiment.config.max_trial_number = 10
    experiment.config.search_space = search_space
    experiment.config.trial_command = 'python scripts/trial_sklearn.py'
    experiment.config.trial_code_directory = './'
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.training_service.use_active_gpu = True








.. GENERATED FROM PYTHON SOURCE LINES 35-37

Start Experiment
----------------

.. GENERATED FROM PYTHON SOURCE LINES 37-39

.. code-block:: default

    experiment.start(8080)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [2022-02-07 18:56:04] Creating experiment, Experiment ID: fl9vu67z
    [2022-02-07 18:56:04] Starting web server...
    [2022-02-07 18:56:05] Setting up...
    [2022-02-07 18:56:05] Web UI URLs: http://127.0.0.1:8080 http://10.190.173.211:8080 http://172.17.0.1:8080 http://192.168.49.1:8080




.. GENERATED FROM PYTHON SOURCE LINES 40-44

Experiment View & Control
-------------------------

View the status of experiment.

.. GENERATED FROM PYTHON SOURCE LINES 44-46

.. code-block:: default

    experiment.get_status()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    'RUNNING'



.. GENERATED FROM PYTHON SOURCE LINES 47-48

Wait until at least one trial finishes.

.. GENERATED FROM PYTHON SOURCE LINES 48-56

.. code-block:: default

    import time

    for _ in range(10):
        stats = experiment.get_job_statistics()
        if any(stat['trialJobStatus'] == 'SUCCEEDED' for stat in stats):
            break
        time.sleep(10)








.. GENERATED FROM PYTHON SOURCE LINES 57-58

Export the experiment data.

.. GENERATED FROM PYTHON SOURCE LINES 58-60

.. code-block:: default

    experiment.export_data()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    [TrialResult(parameter={'C': 0.9, 'kernel': 'rbf', 'degree': 4, 'gamma': 0.07, 'coef0': 0.03}, value=0.9733333333333334, trialJobId='dNOZt'), TrialResult(parameter={'C': 0.8, 'kernel': 'sigmoid', 'degree': 2, 'gamma': 0.01, 'coef0': 0.01}, value=0.9733333333333334, trialJobId='okYSD')]



.. GENERATED FROM PYTHON SOURCE LINES 61-62

Get metric of jobs

.. GENERATED FROM PYTHON SOURCE LINES 62-64

.. code-block:: default

    experiment.get_job_metrics()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    {'okYSD': [TrialMetricData(timestamp=1644227777089, trialJobId='okYSD', parameterId='1', type='FINAL', sequence=0, data=0.9733333333333334)], 'dNOZt': [TrialMetricData(timestamp=1644227777357, trialJobId='dNOZt', parameterId='0', type='FINAL', sequence=0, data=0.9733333333333334)]}



.. GENERATED FROM PYTHON SOURCE LINES 65-67

Stop Experiment
---------------

.. GENERATED FROM PYTHON SOURCE LINES 67-68

.. code-block:: default

    experiment.stop()




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [2022-02-07 18:56:25] Stopping experiment, please wait...
    [2022-02-07 18:56:28] Experiment stopped





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  24.662 seconds)


.. _sphx_glr_download_tutorials_nni_experiment.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: nni_experiment.py <nni_experiment.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: nni_experiment.ipynb <nni_experiment.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

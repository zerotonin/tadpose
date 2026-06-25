TadPose
=======

**Automated behavioural phenotyping of** *Xenopus laevis* **tadpoles from
24-well plate video.**

TadPose turns hour-long multi-well recordings of tadpoles into a
data-driven catalogue of behavioural prototypes, enabling unsupervised
quantification of seizure phenotypes in models of developmental and
epileptic encephalopathies (DEE).  It combines multi-landmark posture
dynamics with centre-of-mass kinematics, clusters them without
human-defined categories, and compares the resulting behavioural
repertoire across drug and genetic seizure models.

The pipeline uses `DeepLabCut <https://github.com/DeepLabCut/DeepLabCut>`_
for markerless pose estimation and
`reRandomStats <https://github.com/zerotonin/reRandomStats>`_ for
re-randomisation statistics.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   installation
   usage
   api

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

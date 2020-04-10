# bpqm
Code base for the Belief-Propagation with Quantum Messages (BPQM) Algorithm

MATLAB codes for the 2020 arXiv paper discussing the BPQM algorithm

Paper: https://arxiv.org/abs/2003.04356

Code: https://github.com/nrenga/bpqm

Copyright (C) 2020  Narayanan Rengaswamy

This project is licensed under the terms of the GNU Affero General Public License (AGPL) v3.0. See LICENSE.md for details.


**Scripts**:

*bpqm_5_bit_code_simulate.m*: This simulates the BPQM algorithm on the 5-bit code discussed in the paper and reproduces the simulation curves in Fig. 14.



**Functions**:

*cbp.m*: This is an implementation of the classical belief-propagation (BP) algorithm, i.e., the sum-product algorithm, for decoding binary linear codes.

*kron_multi.m*: This is a simple function that helps to construct the Kronecker product of more than 2 matrices at once, which extends the native "kron" function in MATLAB.



**Data**:

*datax_nbar.mat*: X-axis information to plot the Yuen-Kennedy-Lax (Limit) for the 5-bit code example.

*datay_perr_vs_nbar.mat*: Y-axis information that plots the YKL limit using https://arxiv.org/abs/1507.04737.
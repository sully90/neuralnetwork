# README #

This package contains the Artificial Neural Network developed and trained in Sullivan et al. 2017 <https://arxiv.org/abs/1707.01427>, written in c++ and used to constrain the halo baryon fraction. No external libraries are required, simply type 'make' to compile (Linux/Mac).

# Connection Weights #

The connections weights file is included in weights/NN50.weights. The code loads this file by default, but may require modification based on the users requirements.

# Running the code #

To run the code in its default state, you must execute the binary from the code directory. The mandatory command line arguments are:

-f: Filename containing data to make predictions for (see example file fb_00106_mw.predict)
-o: Filename to write results.

The prediction file should be formatted as shown in the example, where the columns are:

log_10(ftidal), log_10(xHII), log_10(T(gas)/Tvir), log_10(T/|U|)

The output file contains five columns, the four above plus the predicted halo baryon fraction in units of the cosmic mean (assuming a WMAP7 cosmology).

# Modifications to the code #

It is likely that users will need to modify certain elements of the code to better suit their work. While no guarantee of assistance is given, please feel free to create an issue on this repository or contact the authors directly.

# License #

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Users are free to modify and redistriubute the code as required.
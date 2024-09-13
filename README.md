#!******************************************************!
 !*                                                    *!
 !*                 PreMevE-MEO (2024)                 *!
 !*                                                    *!
 !*         Yinan FENG, Yue CHEN, & Youzuo LIN         *!
 !*                                                    *!
 !*     @Triad National Security, LLC, and/or the @    *!
 !*     @ U.S. Government retain ownership of all @    *!
 !*     @ rights in the Software and the copyright@    *!
 !*     @         subsisting therein              @    *!
 !*                                                    *!
 !*                  September, 2024                   *!
 !******************************************************!
 !* This software uses the following packages:         *!
 !*    matplotlib, numpy, pysci, torch                 *!
 !*    timm, python-utils, pythonnet, cuda             *!
 !*    pandas, scikit-learn, & tensorflow              *!
 !* See package_licenses.txt for copyrights/notices    *!
 !****************  PREMEVE-MEO*2024  ******************!


# PreMevE-MEO  - a PREdictive model for MEgaelectron-Volt Electrons driven by particle intputs from MEO satellites(Software). This is the codes used for the paper xxxxx

# O4773

# Copyright Notice
©2024. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
(End of Notice)

# This program is Open-Source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met: 

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 

    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution

    Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
(End of Notice)


# NOTICE OF DISCLAIMER
The Software are provided AS IS, WITHOUT WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR ANY OTHER WARRANTY, EXPRESS OR IMPLIED. TRIAD AND THE U.S. Test and Evaluation Agreement No. 22-04417 GOVERNMENT MAKE NO EXPRESS OR IMPLIED REPRESENTATION OR WARRANTY THAT THE SOFTWARE WILL NOT INFRINGE ANY PATENT, COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHT. IN NO EVENT WILL TRIAD OR THE U.S. GOVERNMENT BE LIABLE FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES RESULTING FROM EXERCISE OF THIS AGREEMENT OR THE USE OF SOFTWARE.

# NOTICE OF DISCLAIMER
The Software are provided AS IS, WITHOUT WARRANTY OF FUCTIONING at your own platform. This R&D code was crafted to run locally on LANL's Darwin HPC platform, and modfications are expected when this code being mitigated to other platforms/environments. 


# Getting started and use the code
-Clone this repository to your local machine using your tool of choice. The setup.sh script was used to set up environment on LANL Darwin.
-Insall needed packages, includking such as cuda, miniconda, and others, if they are missing.
-Using command "sbath jobs/train.sh' to submit model training jobs. Modfications of parameters in train.sh may be necessary for different models. This step can take a long time for training.
-Using command "sbath jobs/val.sh' to submit model validation and test jobs.
-Model outputs are save to the ckpt subfolder. Several existing models are saved in the same place. 
-Input and target data files are found inside the dataset subfolder.


### gr-tempest

**An implementation of TEMPEST en GNU Radio.** 

![Screenshot of simulation](https://iie.fing.edu.uy/investigacion/grupos/artes/wp-content/uploads/sites/13/2020/05/captura_tempest.png)

**Status:** The examples folder contains several examples that work with recordings and it's been tested on hardware (see videos below). Feedback is more than welcome!

**As of August 2022 several new blocks have been added, and are still under testing. If something's not working, checkout the `vanilla-3.8' branch for the previous, more tested, version**

**If you find the code useful, please consider starring the repository. This will help us get funding to support the project.**

TEMPEST (or [Van Eck Phreaking](https://en.wikipedia.org/wiki/Van_Eck_phreaking)) is a technique to eavesdrop video monitors by receiving the electromagnetic signal emitted by the VGA/HDMI cable and connectors (although other targets are possible, such as keyboards, for which the same term is generally used, see [Wikipedia/Tempest](https://en.wikipedia.org/wiki/Tempest_(codename))). 

This is basically a re-implementation of Martin Marinov's excelent TempestSDR in GNU Radio (see https://github.com/martinmarinov/TempestSDR). The reason is that I felt it may be easier to maintain and extend. Note however that the basic ideas were imitated, but the synchronization algorithms are different, and some functionalities (particularly in the GUI) are missing. 

For a technical explanation you may read [Marinov's thesis](https://github.com/martinmarinov/TempestSDR/raw/master/documentation/acs-dissertation.pdf) or [Pablo Menoni's thesis](https://iie.fing.edu.uy/publicaciones/2018/Men18/) (in spanish). You may also watch [my presentation at GRCon21](https://youtu.be/k_vsFspGpAA) (in English), which includes a technical overview and several demos.

**Notes and examples**

See the examples folder for working examples (and [this youtube playlist](https://www.youtube.com/playlist?list=PLgyC55ufTHCJ9NARNCnoL9QT7isSI9SeV) to see them in action). Recordings may be obtained from https://iie.fing.edu.uy/investigacion/grupos/artes/es/proyectos/espionaje-por-emisiones-electromagneticas/ (in spanish).

There are four examples: 
- *manual_simulated_tempest_example.grc*. This is a simulation of TEMPEST and the signal you are actually spying. It helps to understand the parameters involved and the resulting signal's problems. Both the channel's and the synchronization algorithms' parameters may be modified on the fly from the interface. The image source was mostly copied from [gr-paint's](https://github.com/drmpeg/gr-paint) and any image file readable by PIL may be used (and in any size, as the block reshapes it to the size indicated in the block). 
- *manual_tempest_example.grc*. Feed a recording or the USRP pointing to a VGA cable and you should see an image! However, in this example all parameters of the VGA signal are manually set (and sampling errors are corrected based on these parameters). Moreover, the fine sampling correction should be modified until a stable image is obtained (and horizontal and vertical alignment are also changeable). 
- *semi_automatic_tempest_example.grc*. Same as above, but the resolutions are obtained from a list of the most typical resolutions available. I've obtained the list from TempestSDR (although the command `xrandr --verbose` should produce the list too). 
- *automatic_tempest_example.grc*. Same as above, but the horizontal line is synchronized (thus, only manual vertical alignment is necessary). To achieve this, you should estimate the number of pixels that separate two vertical lines that always appear in the image (due to blanking). Note however that the algorithm may be unstable. In that case, fall back to *semi_automatic_tempest_example.grc*. 
- Semi-automatic and manual examples are also included to spy on HDMI signals. Due to the significant difference between VGA and HDMI, there is no automatic version yet. 

Limitations: 
- A vertical alignment block is under testing, although it still does not work. 
- The spied image is shown in a *Video SDL Sink*, which has its limitations (such as dynamically setting its dimensions). However, the block *Framing* somewhat solves this problem by choosing a user-defined rectangle of the incoming signal. Thus, exploring resolutions while executing the flowgraph is now possible. See the comments on the flowgraph. 
- The synchronization blocks may be heavy on the PC. This may be alleviated by properly configuring VOLK as explained below. Moreover, if CPU is still a problem with the *sampling synchronization* block, you may try reducing the variable `d_proba_of_updating` and/or `d_max_deviation` in *lib/sampling_synchronization_impl.cc* (and using only *manual_simulated_tempest_example.grc*).

**Requirements**: GNU Radio 3.8 or 3.7, either compiled from source or installed with a binary (see below if this this is your case for further requirements). If you have trouble compiling gr-tempest, I've prepared a docker container which meets the requirements (you have to compile and install gr-tempest in it though). Follow [this link](https://github.com/git-artes/docker-gnuradio). 

**Important**: If you have GNU Radio 3.7 installed, you have to execute `git checkout maint-3.7` right after `cd gr-tempest` below.

**Build instructions**

For a system wide installation:

    git clone https://github.com/git-artes/gr-tempest.git  
    cd gr-tempest
    mkdir build  
    cd build  
    cmake ../  
    make && sudo make install  

For a user space installation, or GNU Radio installed in a location different from the default location /usr/local:

    git clone https://github.com/git-artes/gr-tempest.git  
    cd gr-tempest 
    mkdir build  
    cd build  
    cmake -DCMAKE_INSTALL_PREFIX=<your_GNURadio_install_dir> ../
    make
    make install  

Please note that if you used PyBOMBS to install GNU Radio the DCMAKE_INSTALL_PREFIX should point to the PyBOMBS prefix. 

On Debian/Ubuntu based distributions, you may have to run:

    sudo ldconfig  

**Remarks**
- Instructions above assume you have downloaded and compiled GNU Radio (either with the build-gnuradio script or with PyBOMBS). If you've used a pre-compiled binary package (like in $ sudo apt-get install gnuradio), then you should also install gnuradio-dev (and naturally cmake and git, if they were not installed, plus libboost-all-dev, libcppunit-dev, liblog4cpp5-dev and swig).   
- Note that gr-tempest makes heavy use of VOLK. A profile should be run to make the best out of it. Please visit https://gnuradio.org/redmine/projects/gnuradio/wiki/Volk for instructions, or simply run:   

    volk_profile 

**FAQ**

*Q*: Cmake complains about unmet requirements. What's the problem?   
*A*: You should read the errors carefully (though we reckon they are sometimes mysterious). Most probably is a missing library. Candidates are Boost (in Ubuntu libboost-all-dev) or libcppunit (in Ubuntu libcppunit-dev).   

*Q*: Cmake complains about some Policy CMP0026 and LOCATION target property and who knows what else. Again, what's the problem?  
*A*: This is a problem with using Cmake with a version >= 3, which is installed in Ubuntu 16, for instance. The good news is that you may ignore all these warnings. 

*Q*: It is not compiling. What's the problem?  
*A*: Again, you should read carefully the errors. Again, it's most probably a missing library, for instance log4cpp (in Ubuntu liblog4cpp5-dev). If the problem is with the API of GNU Radio, you should update it. I've tested gr-tempest with GNU Radio 3.7.11. Finally, if it complains about the random number generator, you have to compile using the -std=c++11 flag. 

*Q*: I got the following error: "ModuleNotFoundError: No module named 'tempest'". What's wrong?

*A*: You probably didn't setup the PYTHONPATH correctly. It should include at least /usr/local/lib/python3/dist-packages. For a system-wide solution, you may edit /etc/environment and include the following line PYTHONPATH=$PYTHONPATH:"/usr/local/lib/python3/dist-packages"

*Q*: I got the following error: "AttributeError: 'module' object has no attribute 'sampling_synchronization'" (or some other block). Why?  
*A*: This problem may be generated by several factors. Did you "sudo ldconfig"? Another possibility is that you don't have swig installed (in this case, you must uninstall gr-tempest, delete CMakeCache.txt in the build directory, and re-install; that is, after installing swig).   

IIE Instituto de Ingeniería Eléctrica  
Facultad de Ingeniería  
Universidad de la República  
Montevideo, Uruguay  
http://iie.fing.edu.uy/investigacion/grupos/artes/  
  
Please refer to the LICENSE file for contact information and further credits.   


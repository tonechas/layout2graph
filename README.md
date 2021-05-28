# layout2graph
Code for converting a plant layout drawing into an undirected graph.

This software automatically checks whether a layout drawing (in `.dwg` format) meets certain functional specifications or not by comparing the obtained graph with a target graph. Such system comes in handy in an engineering teaching setting, as it can be utilised to help instructors grade students' designs.
## Getting started
First you have to define the following variables in `config.py`:
* `study_case` is the name of the folder that contains data (`motorhome-conversion-center` in the provided example). This folder must hace 4 subfolders, namely:
    * `drawings`, where the  `.dwg` files are located.
    * `graphs`, which stores the generated graphs and images.
    * `logs` for logging data.
    * `temp` for temporary files.
* `target_nodes` and `target_edges`: these define the specifications and constraints a design must comply with.
* `key` is a dictionary whose keys are integer numbers and the values are the names of the different spaces.

When the configuration is complete you simply have to run `layout2graph.py`.
## Caveat
If you get a weird `win32com` exception when trying to run the program, you should locate and remove the `gen_py` folder as recommended [here](https://stackoverflow.com/questions/9024668/added-typelib-via-makepy-for-autocad-now-win32com-not-working-for-autocad).
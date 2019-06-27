How to use:

```
roofline.R:
    Rscript --vanilla roofline.R <filename.ssv> <backend> <device> <mode> <bw> <outfile>
```

Parameters
```
filename.ssv:
    file to plot

backend, device:
    Printed on plot

mode:
    0: Plot as a percentage of a max bw (specified in <bw>)
    1: Plot the raw bw data (do not put anything for <bw>)
    2: Use a fixed max for the y axis (specified in <bw>)

outfile: 
    Optionally specify an output file name
```

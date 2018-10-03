library(ggplot2)

#Modes
# 0: Percent of max, 1: raw data 2: fixed max

args = commandArgs(trailingOnly=TRUE)

#Examples
#args = c("~/sgbench/results/v0.1/scatter/cuda/p100/sg_sparse_roofline_cuda_p100_SCATTER.ssv", "CUDA", "Tesla P100", 0, 539486.884, "testfile.png")
#args = c("~/sgbench/results/v0.1/scatter/cuda/p100/sg_sparse_roofline_cuda_p100_SCATTER.ssv", "CUDA", "Tesla P100", 1)
#args = c("~/sgbench/results/v0.1/scatter/cuda/p100/sg_sparse_roofline_cuda_p100_SCATTER.ssv", "CUDA", "Tesla P100", 2, 120000)
#args = c("-h")

if(length(args) < 4 || args[1] == "-h") {
  stop("Usage: Rscript --vanilla roofline.R roofline.ssv backend device mode <bw(mib/s)> <outputfile.png>")
} 

filename = args[1]

#split_filename = strsplit(filename, "_")

backend = args[2]
device = args[3]
mode = as.numeric(args[4])

max_bw = 1
fixed_max = 1
outfile = "NONE"

if (mode == 0){
  if(length(args) < 5) {
    stop("If you want to display bw as percent of max, you must supply the max as the 4th arg")
  } 
  max_bw = as.numeric(args[5])
  if(length(args) >= 6) {
    outfile = args[6]
  }
}else if (mode == 1) {
  if(length(args) >= 5) {
    outfile = args[5]
  }
} else if (mode == 2) {
  if(length(args) < 5) {
    stop("If you want to display a fixed heigh y axis, you must supply the max as the 4th arg")
  } 
  fixed_max = as.numeric(args[5])
  if(length(args) >= 6) {
    outfile = args[6]
  }
}

#teslapeak = 191079.160
#p100peak  = 539486.884
#titanpeak = 434231.942

data = read.table(filename)
colnames(data) = c('backend', 'kernel', 'op', 
                   'time', 'source_size', 'target_size', 
                   'idx_size', 'worksets', 'bytes_moved', 
                   'usable_bandwidth', 'omp_threads', 'vector_len','block_dim')

#Determine sparsity regardless of what kernel was run
data$density = pmax(data$target_size / data$idx_size, data$source_size/data$idx_size)
if (mode == 0) {
  data$bw_pct  = (data$usable_bandwidth / max_bw) * 100
}

#Determine the kernel name (to be used in plot title)
if (as.character(data$kernel[1]) == "SCATTER") {
  kernel_name = "Scatter"
} else if (as.character(data$kernel[1]) == "GATHER") {
  kernel_name = "Gather"
} else{
  kernel_name = "Scatter+Gather"
}

#How to aggregate data https://stackoverflow.com/questions/34523679/aggregate-multiple-columns-at-once
agg = aggregate(.~density+vector_len, data, max)
agg$density = as.factor(agg$density)
vec = unique(log2(data$vector_len))

#Set mode-specific axes
if (mode == 0) {
  #data$bw_pct  = (data$usable_bandwidth / max_bw) * 100
  y_name = "Usable Bandwidth (% of BabelStream)"
  y_breaks = seq(0,ceiling(max(data$bw_pct)))
  
  p = ggplot(agg, aes(x=log2(vector_len), y=bw_pct, col=density, group=density)) + expand_limits(y=0)
  
} else if (mode == 1) {
  y_name = "Usable Bandwidth (MiB/s)"
  y_breaks = seq(0,max(data$usable_bandwidth),round(max(data$usable_bandwidth)/10,-3))
  
  p = ggplot(agg, aes(x=log2(vector_len), y=usable_bandwidth, col=density, group=density)) + expand_limits(y=0)
  
} else if (mode == 2){
  y_name = "Usable Bandwidth (MiB/s)"
  y_breaks = seq(0,fixed_max, round(fixed_max,-3)/10)
  
  p = ggplot(agg, aes(x=log2(vector_len), y=usable_bandwidth, col=density, group=density)) +
    expand_limits(y=c(0,fixed_max) )
}


p = p + geom_point(size=3) + 
    geom_line() + 
    scale_x_continuous(name="Work per kernel", breaks=vec, labels=as.character(2^vec)) + 
    scale_y_continuous(name=y_name, breaks=y_breaks) + 
    labs(title=paste(kernel_name, "Rooflines"), subtitle=paste(device, ", ",backend, " Backend", sep=""), color="1/Density")  + 
    theme_bw()

if (outfile == "NONE"){
  outfile=paste(tools::file_path_sans_ext(basename(filename)), ".png", sep="")
}
ggsave(outfile, device=png(), plot=p, width=6, height=6, units="in")
print(paste("Wrote to:", outfile))

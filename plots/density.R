library(ggplot2)

args = commandArgs(trailingOnly=TRUE)


#setwd("/Users/airavata/sgbench/plots")

#Example
#args = c("../results/v0.2/gather/cuda/p100/sg_sparse_roofline_cuda_p100_GATHER_2.ssv", 
#         "../results/v0.2/gather/cuda/titan/sg_sparse_roofline_cuda_titan_GATHER_2.ssv",
#         "outfile.ssv")

if(length(args) < 2 || args[1] == "-h" || args[1] == "--help") {
  stop("Usage: Rscript --vanilla density.R <file1.ssv> [file2.ssv] ... <outputfile.png>")
} 

dev=c("p100","titan","k40")
bw=c(539486.884,434231.942,191079.160)
name=c("P100","Titan Xp", "K40c")
dev_bw=data.frame(dev,bw,name)

MiBtoMB = 1024*1024 / (1000*1000)

#Read in all files
data = read.table(args[1])
colnames(data) = c('backend', 'kernel', 'op', 
                   'time', 'source_size', 'target_size', 
                   'idx_size', 'worksets', 'bytes_moved', 
                   'usable_bandwidth', 'omp_threads', 
                   'vector_len','block_dim','shmem')
data$file = 1
device = (strsplit(args[1],"_"))[[1]][5]
data$dev = device
data$usable_bandwidth = data$usable_bandwidth * MiBtoMB
data$bw_pct = (data$usable_bandwidth / subset(dev_bw, dev==device)$bw) * 100
device_names = device

if(length(args) > 2){
  for(i in 2:(length(args)-1)){
    temp = read.table(args[i])
    colnames(temp) = c('backend', 'kernel', 'op', 
                       'time', 'source_size', 'target_size', 
                       'idx_size', 'worksets', 'bytes_moved', 
                       'usable_bandwidth', 'omp_threads', 
                       'vector_len','block_dim','shmem')
    temp$file= i
    temp$usable_bandwidth = temp$usable_bandwidth * MiBtoMB
    device = (strsplit(args[i],"_"))[[1]][5]
    device_names = c(device_names, device)
    temp$dev = device
    temp$bw_pct = (temp$usable_bandwidth / subset(dev_bw, dev==device)$bw) * 100
    data = rbind(data,temp)
  }
}

outfile = args[length(args)]



#teslapeak = 191079.160
#p100peak  = 539486.884
#titanpeak = 434231.942


#Determine sparsity regardless of what kernel was run
data$density = pmax(data$target_size / data$idx_size, data$source_size/data$idx_size)
data$ld=log2(data$density)
#Determine the kernel name (to be used in plot title)
kernel_name=(strsplit(args[1],"_"))[[1]][6]

#How to aggregate data https://stackoverflow.com/questions/34523679/aggregate-multiple-columns-at-once
data2 = data[c("vector_len", "block_dim", "file", "bw_pct", "ld","dev")]
agg = aggregate(.~ld+dev,subset(data2,file==1),max)
for(i in 2:(length(args)-1)){
  agg_tmp = aggregate(.~ld+dev,subset(data2,file==i),max)
  agg = rbind(agg,agg_tmp)
}

for(n in dev){
  agg[agg==n] = as.character(subset(dev_bw,dev==n)$name)
}


agg$file = as.factor(agg$file)
y_name = "Usable Bandwidth (% of BabelStream)"
y_breaks = seq(0,90,10)
p = ggplot(agg, aes(x=ld, y=bw_pct, col=dev, group=dev)) + expand_limits(y=0)

p = p + geom_point(size=3) + 
  geom_line() + 
  scale_x_continuous(name="Sparisty", breaks=agg$ld, labels=as.character(2^agg$ld)) + 
  scale_y_continuous(name=y_name, breaks=y_breaks) + 
  labs(title=paste("Impact of Access Sparsity"), subtitle=paste(kernel_name," kernel"), color="Device", name="NULL")  + 
  theme_bw() + scale_fill_discrete(labels=c("one", "two"))

ggsave(outfile, device=png(), plot=p, width=6, height=6, units="in")
print(paste("Wrote to:", outfile))

- [KNN-Mapreduce-From-Scratch](#knn-mapreduce-from-scratch)
    - [实验目标](#%E5%AE%9E%E9%AA%8C%E7%9B%AE%E6%A0%87)
    - [环境配置](#%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE)
    - [使用工具](#%E4%BD%BF%E7%94%A8%E5%B7%A5%E5%85%B7)
    - [使用数据集](#%E4%BD%BF%E7%94%A8%E6%95%B0%E6%8D%AE%E9%9B%86)
    - [实验思路](#%E5%AE%9E%E9%AA%8C%E6%80%9D%E8%B7%AF)
    - [实现代码](#%E5%AE%9E%E7%8E%B0%E4%BB%A3%E7%A0%81)
        - [辅助类实现](#%E8%BE%85%E5%8A%A9%E7%B1%BB%E5%AE%9E%E7%8E%B0)
            - [距离计算类](#%E8%B7%9D%E7%A6%BB%E8%AE%A1%E7%AE%97%E7%B1%BB)
            - [样本实例类](#%E6%A0%B7%E6%9C%AC%E5%AE%9E%E4%BE%8B%E7%B1%BB)
        - [Mapper实现](#mapper%E5%AE%9E%E7%8E%B0)
        - [Reducer实现](#reducer%E5%AE%9E%E7%8E%B0)
        - [KNN_MapReduce实现](#knnmapreduce%E5%AE%9E%E7%8E%B0)
    - [实验过程](#%E5%AE%9E%E9%AA%8C%E8%BF%87%E7%A8%8B)
        - [启动服务](#%E5%90%AF%E5%8A%A8%E6%9C%8D%E5%8A%A1)
        - [导入数据集到HDFS上](#%E5%AF%BC%E5%85%A5%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%B0hdfs%E4%B8%8A)
        - [运行mapreduce代码](#%E8%BF%90%E8%A1%8Cmapreduce%E4%BB%A3%E7%A0%81)
    - [实验结果及分析](#%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C%E5%8F%8A%E5%88%86%E6%9E%90)
    - [附录](#%E9%99%84%E5%BD%95)
        - [xshell上的文件导入导出](#xshell%E4%B8%8A%E7%9A%84%E6%96%87%E4%BB%B6%E5%AF%BC%E5%85%A5%E5%AF%BC%E5%87%BA)
        - [可能出现的问题](#%E5%8F%AF%E8%83%BD%E5%87%BA%E7%8E%B0%E7%9A%84%E9%97%AE%E9%A2%98)

# KNN-Mapreduce-From-Scratch

## 实验目标
 - 使用腾讯云主机安装docker
 - 使用docker安装hadoop并启动服务（单节点：只是为了运算方便）
 - 使用Mapreduce实现KNN算法

## 环境配置

主机配置

|内存大小|处理器类型|处理器个数|操作系统|
|:-:|:-:|:-:|:-:|
|2GB|Linux VM-13-51-ubuntu 4.4.0-130-generic|1|16.04版|

docker配置

 - Docker version 19.03.2, build 6a30dfc

## 使用工具

- docker
    - 安装hadoop 启动容器
- Xshell 5
    - 用于连接和访问云服务器，实验过程中均在该平台上运行命令

## 使用数据集

本实验使用如下数据分布的训练集和测试集进行算法的测试。

![](./images/1.png)


该数据集具有如下特点：

- 数据集中每行数据前17列是以逗号划分的特征列，最后一列为取值为0或者1的标签列；
- 数据集特征可分为如下两类：
    - 离散特征：第0-4列和第14列的取值只为0和1，第15列的取值只为0或者3
    - 连续特征：除了离散特征外的所有其他列均为连续特征列。


其中，训练集共有33600行数据，测试集则有14400行数据。

训练集中，0标签数据占比51.3%，1标签数据占比48.7%，总体差别比例不大。



## 实验思路

首先，mapreduce可应用于KNN的前提是：KNN算法中的步骤可拆分为不要求顺序执行的多个部分。理解了KNN算法原理后便可想到：多个测试样本求K近邻的过程是不要求顺序执行的，即该过程是可以并行处理的。因此，便可这样设计mapreduce的过程：

- Mapper:输入一个测试样本，得到该测试样本的K个近邻；
- Reducer:输入一个测试样本以及它的K个近邻，得到这K个近邻的标签的众数作为预测标签。


## 实现代码

### 辅助类实现

#### 距离计算类

由于KNN求K近邻需要计算两个样本直接的距离，这里便实现该类来完成该部分功能。

```java
class Distance {
    /*计算样本a与样本b之间的欧式距离*/
    public static double calcEuclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
}
```

这里为了测试方便，只实现了欧式距离的计算方式。

#### 样本实例类

由于在算法处理过程中需要将数据集的每行数据，也即是每个样本，依靠分隔符将其在各个特征上的取值和标签取值提取出来。因此便实现该类来完成数据的转换工作。

```java
class Instance {
    /*根据每一行数据划分数据的特征、标签*/
    private double[] attributeSet;//样例属性
    private double label;         //样例标签

    public Instance(String data_line) {
        //用逗号分隔数据
        String[] data_input = data_line.split(",");		 
        //前length-1项为属性
        attributeSet = new double[data_input.length - 1];
        for (int i = 0; i < attributeSet.length; i++) {
            attributeSet[i] = Double.parseDouble(data_input[i]);
        }
        //第length项为标签
        label = Double.parseDouble(data_input[data_input.length - 1]);
    }

    public double[] getAttributeSet() {
        return attributeSet;
    }

    public double getlabel() {
        return label;
    }
}
```

### Mapper实现

由于在Mapper中要计算一个测试样本和整个训练集所有样本的距离，并找出与其距离最近的K个近邻的标签。因此，首先需要实现如下两个函数：
- setup函数，该函数在所有Mapper结点开始工作之前被调用，因此便可在该函数中加载整个训练集到内存上；
- map函数：其是Mapper的核心函数，将计算测试样本与训练集所有样本的距离，并找出与其距离最近的K个近邻的标签。

```java
 public static class KNN_Mapper extends Mapper<LongWritable, Text, Text, Text> {
    public ArrayList<Instance> trainSet = new ArrayList<Instance>();
    /*******在这里修改 K 值*******/
    public int K = 1;
    /*******在这里修改 K 值*******/
    protected void setup(Context context) throws IOException, InterruptedException {
        //读取训练集
        FileSystem fileSystem = null;
        try {
            fileSystem = FileSystem.get(new URI("hdfs://192.168.142.128:9000/"), new Configuration());
        } catch (Exception e) {
        }
        FSDataInputStream trainSet_input = fileSystem.open(new Path("hdfs://192.168.142.128:9000/knn_train/train.txt"));
        BufferedReader trainSet_data = new BufferedReader(new InputStreamReader(trainSet_input)); 

        //逐行划分训练集中特征以及标签
        String str = trainSet_data.readLine();
        while (str != null) {
            trainSet.add(new Instance(str));
            str = trainSet_data.readLine();
        }
    }

    protected void map(LongWritable k1, Text v1, Context context) throws IOException, InterruptedException {
        
        ArrayList<Double> distance = new ArrayList<Double>(K);
        ArrayList<String> trainlabel = new ArrayList<String>(K);
        
        //初始化前 K 小距离和标签值
        for (int i = 0; i < K; i++)
        {
            distance.add(Double.MAX_VALUE);
            trainlabel.add("NAN");
        }
        
        //读取一个test样本
        Instance testInstance = new Instance(v1.toString());
        //计算test样本和各个train样本距离
        for (int i = 0; i < trainSet.size(); i++) {
            double dis = Distance.calcEuclideanDistance(trainSet.get(i).getAttributeSet(), testInstance.getAttributeSet());

            for (int j = 0; j < K; j++)//若距离比元素值小，则覆盖
            {
                if (dis < (Double) distance.get(j)) {
                    distance.set(j, dis);
                    trainlabel.set(j, trainSet.get(i).getlabel() + "");
                    break;
                }
            }
        }
        
        //mapper过程输出: 以测试集特征为 key 值，以 K 个近邻的标签值列表为 value 值
        for (int i = 0; i < K; i++)
        {
            context.write(new Text(v1.toString()), new Text(trainlabel.get(i) + ""));
        }
    }
}

```

### Reducer实现

Reducer的任务就是根据输入的测试样本以及它的K个近邻，使用多数投票的方式，以这K个近邻的标签的众数作为预测标签。

```java
public static class KNN_Reducer extends Reducer<Text, Text, Text, NullWritable> {
    
    protected void reduce(Text k2, Iterable<Text> v2s, Context context) throws IOException, InterruptedException {
        
        //提取输入的标签值
        ArrayList<String> KNeighborsLabel = new ArrayList<String>();
        for (Text v2 : v2s)
        {
            KNeighborsLabel.add(v2.toString());
        }
        
        //统计 K 近邻的标签
        String predictlabel = MajorityVoting(KNeighborsLabel);
        
        //reducer过程输出： 以测试集特征以及预测标为 key 值，以 空值 为 value 值
        String preresult = k2.toString() + "," + predictlabel;
        context.write(new Text(preresult), NullWritable.get());
    }

    public String MajorityVoting(ArrayList KNeighbors) {
        /*多数投票函数实现*/
        HashMap<String, Double> freqCounter = new HashMap<String, Double>();
        
        //遍历所有输入标签，统计出现次数
        for (int i = 0; i < KNeighbors.size(); i++)
        {
            if (freqCounter.containsKey(KNeighbors.get(i))) {
                double frequence = freqCounter.get(KNeighbors.get(i)) + 1;
                freqCounter.remove(KNeighbors.get(i));
                freqCounter.put((String) KNeighbors.get(i), frequence);
            } else {
                freqCounter.put((String) KNeighbors.get(i), new Double(1));
            }
        }
        
        //多数投票得到最终预测标签
        Iterator it = freqCounter.keySet().iterator();
        double maxi = Double.MIN_VALUE;
        String final_predict = null;
        while (it.hasNext())//取出现最多的标签
        {
            String key = (String) it.next();
            Double labelnum = freqCounter.get(key);
            if (labelnum > maxi) {
                maxi = labelnum;
                final_predict = key;
            }
        }
        return final_predict;
    }
}
```

### KNN_MapReduce实现

实现了Mapper和Reducer后，便要在KNN_MapReduce类中main函数进行mapreduce的设置，这样hadoop才可以根据我们配置的信息按照我们想要的方式进行算法的运行。

```java
public class KNN_MapReduce {
	/*KNN mapreduce实现*/

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 3) {
            System.err.println("Usage: KNN_MapReduce <trainSet_path> <testSet_path> <output_path>");
            System.exit(2);
        }
		
        //若存在output文件夹，则先删除output文件夹
		FileSystem fileSystem = FileSystem.get(conf);
        if (fileSystem.exists(new Path(otherArgs[2])))
        {
            fileSystem.delete(new Path(otherArgs[2]), true);
        }
		
        //设置基本信息
        Job job = new Job(conf, "KNN");
        job.setJarByClass(KNN_MapReduce.class);
        job.setInputFormatClass(TextInputFormat.class);

        //设置Mapper
        job.setMapperClass(KNN_Mapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setNumReduceTasks(1);
        job.setPartitionerClass(HashPartitioner.class);

        //设置Reducer
        job.setReducerClass(KNN_Reducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        //设置训练数据的输入路径
        FileInputFormat.addInputPath(job, new Path(otherArgs[1]));
        //设置任预测结果的输出路径
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[2]));
        //等待任务完成后退出程序
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}    
```

## 实验过程

### 寻找hadoop镜像 并启动容器

docker为hadoop的云化带来了极大便利,安装和应用也会更快更方便

```sh
hadoop@master:~$ docker search hadoop
```
```sh
sequenceiq/hadoop-docker         An easy way to try Hadoop                       617  [OK]
uhopper/hadoop                   Base Hadoop image with dynamic configuration…   99   [OK]
harisekhon/hadoop                Apache Hadoop (HDFS + Yarn, tags 2.2 - 2.8)     55   [OK]
bde2020/hadoop-namenode          Hadoop namenode of a hadoop cluster             26   [OK]
izone/hadoop                     Hadoop 2.8.5 Ecosystem fully distributed, Ju…   14   [OK]
bde2020/hadoop-datanode          Hadoop datanode of a hadoop cluster             12   [OK]
uhopper/hadoop-namenode          Hadoop namenode                                 9    [OK]
singularities/hadoop             Apache Hadoop                                   8    [OK]
uhopper/hadoop-datanode          Hadoop datanode                                 8    [OK]
```
我们当然选择星最多的.选择第一个:
```sh
hadoop@master:~$ docker pull sequenceiq/hadoop-docker
```
然后等待下载.下载完毕之后启动docker镜像，将端口映射出来：
```sh
hadoop@master:~$ docker run -it --name hadoop_03  -p  50070:50070 -p 9000:9000 -p  8088:8088 sequenceiq/hadoop-docker /etc/bootstrap.sh -bash
```
 - 50070：HDFS文件管理页面
 - 9000：是fileSystem默认的端口号 浏览器直接访问是访问不到的
 - 8088：运行mapreduce作业，其的运行状态会在8088显示

理论上应该还会开放一些别的端口，但是这里我们只映射这三个就足够运行knn算法。

### 启动服务

1.首先，可以看到容器启动成功后自动进入容器，运行后输入`jps`，可看到如下输出，可见hadoop服务启动正常。

```sh
hadoop@master:~$ jps
7008 Jps
131 NameNode
227 DataNode
663 NodeManager
565 ResourceManager
414 SecondaryNameNode
```
此时你一定很疑问,hadoop安装到哪里去了呢？输入`cd /usr/loacl`就能够看到hadoop安装在这
```sh
bash:~# cd /usr/loacl 
```
2.访问ip:50070端口可以访问到UI界面：

![](./images/4.png)

当然箭头指的两个文件夹现在你是显示不到，因为你还没有上传！

### 是不是很简单 几条命令我们就安装好了hadoop，并启动了服务！



### 导入数据集到HDFS上

接着通过xshell将训练集`train.txt`以及测试集文件`test.txt`导入到云服务器上`/home/ubuntu/`，随便你放在什么地方，我们要把数据集传到docker容器中去，并运行如下命令以将两个文件放置在HDFS上。

```sh
服务器:~# docker cp /home/ubuntu/train.txt 容器id:/user/local
服务器:~# docker cp /home/ubuntu/test.txt 容器id:/user/local
```
进去容器：
```sh
服务器:~# docker exec -ti 容器id /bin/bash
```
将文件上传到hdfs:
```sh
bash:~# bin/hdfs dfs -put train.txt /knn_train
bash:~# bin/hdfs dfs -put test.txt /knn_test
```
通过50070端口或者命令`bin/hdfs dfs -ls /`查看是否上传成功
```sh
bash:~# bin/hdfs dfs -ls /
```
成功的话就是这个界面啦

![](./images/4.png)


### 运行mapreduce代码

使用idea新建maven项目目录如下：

![](./images/3.png)

说明的一点是pom.xmld的需要的文件

```sh
  <dependency>
      <groupId>org.apache.hadoop</groupId>
      <artifactId>hadoop-common</artifactId>
      <version>2.7.0</version>
    </dependency>
    <dependency>
      <groupId>org.apache.hadoop</groupId>
      <artifactId>hadoop-hdfs</artifactId>
      <version>2.7.0</version>
    </dependency>
    <dependency>
      <groupId>org.apache.hadoop</groupId>
      <artifactId>hadoop-mapreduce-client-common</artifactId>
      <version>2.7.0</version>
    </dependency>
    <dependency>
      <groupId>org.apache.hadoop</groupId>
      <artifactId>hadoop-mapreduce-client-core</artifactId>
      <version>2.7.0</version>
    </dependency>
```

我们需要将整个工程打包成jar,在idea下的terminal窗口：
```$xslt
mav clean package
```
![](./images/10.png)

接着通过xshell将`TestKnn01-1.0-SNAPSHOP.jar`导入到云主机上，从云主机上再次导入到hadoop容器中去


### 运行 mapreduce

```$xslt
bin/hadoop jar TestKnn01-1.0-SNAPSHOT.jar TestKnn02/KNN_MapReduce /knn_train /knn_test /knn_output
```
 - 注意运行与wordcount的jar相似，只是jar包变成了knn的算法而已
 - 当前目录是hadoop的安装目录 /usr/loacl/hadoop/
 - 运行时包名/方法名 才能够访问的到
 - /knn_train ：输入目录
 - /knn_test：输入目录
 - /knn_output：输出目录

### 运行过程
![](./images/7.jpg)
![](./images/8.jpg)

![](./images/5.png)

![](./images/6.png)

![](./images/9.png)

### 导出运行结果文件 或者直接cat hdfs的目录即可查看运行结果

```$xslt
#/bin/hadoop dfs -cat /knn_output/part-r-00000 
或者：
#/bin/hadoop dfs -get /knn_output/*
#sz part-r-00000 //发送到本地电脑端
```



可以看到，KNN算法在测试集上的准确率和F1值均在80%以上，其中K=1时算法表现最佳，运行时间基本在40min左右。

## 附录

### xshell上的文件导入导出

- 文件导入：在命令行输入`rz`后会弹出对话框，选择文件后文件会被导入在当前路径下。
- 文件导出：在命令行输入`sz <file name>`后会弹出对话框，选择路径后文件会被导出到对应路径下。
- 云服务器文件导入容器内：docker cp /xx/xx 容器id:/xx/xx
- mvn打包：mvn clean package
- 不杀死容器方式退出容器：control + p + q

### 可能出现的问题

- hadoop端口映射问题，查看云主机是否开放了相关的端口
- 程序中的ip和9000端口要改成自己的
- 训练文件和测试文件一定要放在hdfs中
- running job 可能出现卡死 一般不会出现 检查相关配置文件


### 感谢linzch3 

参考链接：

https://github.com/linzch3/KNN-Mapreduce-From-Scratch

https://blog.csdn.net/weixin_42685589/article/details/81111406

https://blog.csdn.net/qq_39009237/article/details/86346762 


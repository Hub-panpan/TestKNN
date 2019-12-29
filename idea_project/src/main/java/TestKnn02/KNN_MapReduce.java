package TestKnn02;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.GenericOptionsParser;

public class KNN_MapReduce {


    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 3) {
            System.err.println("Usage: KNN_MapReduce <trainSet_path> <testSet_path> <output_path>");
            System.exit(2);
        }


        FileSystem fileSystem = FileSystem.get(conf);
        /*判断输出路径是不是存在 如果是存在的话就把文件夹删除*/
        if (fileSystem.exists(new Path(otherArgs[2])))
        {
            fileSystem.delete(new Path(otherArgs[2]), true);
        }
        /*结束*/

        //设置job的名字
        Job job = new Job(conf, "KNN");
        //设置job的主函数入口
        job.setJarByClass(KNN_MapReduce.class);
        //就是指定TextInputFormat来完成这项工作，这个类是hadoop默认的其实可以不写
        job.setInputFormatClass(TextInputFormat.class);
        

		// 指定mapper类，指定mapper的输出<k2,v2>类型
        job.setMapperClass(KNN_Mapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
       //  指定reduce类，指定reduce的输出<k3,v3>类型
        job.setNumReduceTasks(1);
        job.setPartitionerClass(HashPartitioner.class);


        job.setReducerClass(KNN_Reducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(otherArgs[1]));

        FileOutputFormat.setOutputPath(job, new Path(otherArgs[2]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);

    }

    public static class KNN_Mapper extends Mapper<LongWritable, Text, Text, Text> {
        public ArrayList<Instance> trainSet = new ArrayList<Instance>();

        public int K = 1;

        protected void setup(Context context) throws IOException, InterruptedException {

            FileSystem fileSystem = null;
            try {
                fileSystem = FileSystem.get(new URI("hdfs://123.206.23.3:9000/"), new Configuration());
            } catch (Exception e) {
            }
            FSDataInputStream trainSet_input = fileSystem.open(new Path("hdfs://123.206.23.3:9000/knn_train/train.txt"));
            BufferedReader trainSet_data = new BufferedReader(new InputStreamReader(trainSet_input));


            String str = trainSet_data.readLine();
            while (str != null) {
                trainSet.add(new Instance(str));
                str = trainSet_data.readLine();
            }
        }

        protected void map(LongWritable k1, Text v1, Context context) throws IOException, InterruptedException {
            //distance 
            ArrayList<Double> distance = new ArrayList<Double>(K);
            //trainlable
            ArrayList<String> trainlabel = new ArrayList<String>(K);


            for (int i = 0; i < K; i++)
            {
                distance.add(Double.MAX_VALUE);
                trainlabel.add("NAN");
            }


            Instance testInstance = new Instance(v1.toString());

            for (int i = 0; i < trainSet.size(); i++) {
                double dis = Distance.calcEuclideanDistance(trainSet.get(i).getAttributeSet(), testInstance.getAttributeSet());

                for (int j = 0; j < K; j++)
                {
                    if (dis < (Double) distance.get(j)) {
                        distance.set(j, dis);
                        trainlabel.set(j, trainSet.get(i).getlabel() + "");
                        break;
                    }
                }
            }


            for (int i = 0; i < K; i++)
            {
                context.write(new Text(v1.toString()), new Text(trainlabel.get(i) + ""));
            }
        }
    }

    public static class KNN_Reducer extends Reducer<Text, Text, Text, NullWritable> {

        protected void reduce(Text k2, Iterable<Text> v2s, Context context) throws IOException, InterruptedException {


            ArrayList<String> KNeighborsLabel = new ArrayList<String>();
            for (Text v2 : v2s)
            {
                KNeighborsLabel.add(v2.toString());
            }


            String predictlabel = MajorityVoting(KNeighborsLabel);


            String preresult = k2.toString() + "," + predictlabel;
            context.write(new Text(preresult), NullWritable.get());
        }

        public String MajorityVoting(ArrayList KNeighbors) {

            HashMap<String, Double> freqCounter = new HashMap<String, Double>();


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


            Iterator it = freqCounter.keySet().iterator();
            double maxi = Double.MIN_VALUE;
            String final_predict = null;
            while (it.hasNext())
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
}

class Distance {

    public static double calcEuclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
}

class Instance {

    private double[] attributeSet;
    private double label;

    public Instance(String data_line) {

        String[] data_input = data_line.split(",");

        attributeSet = new double[data_input.length - 1];
        for (int i = 0; i < attributeSet.length; i++) {
            attributeSet[i] = Double.parseDouble(data_input[i]);
        }

        label = Double.parseDouble(data_input[data_input.length - 1]);
    }

    public double[] getAttributeSet() {
        return attributeSet;
    }

    public double getlabel() {
        return label;
    }
}

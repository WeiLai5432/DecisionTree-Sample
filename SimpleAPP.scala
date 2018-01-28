import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors,Vector}


object SimpleApp{
	def main(args: Array[String]) {

		//新sc环境
		val conf = new SparkConf().setAppName("Simple Application")
		val sc = new SparkContext(conf)

		//读取数据，已对数据进行预处理，将属性字符串变成double类型数字
		val data = sc.textFile("data/train-double.txt")

		//对训练集进行预处理，区分标签和特征

		//二分类
		// val parsedData = data.map { line =>
		//  val parts = line.split(',');
		//  LabeledPoint(if(parts(6)=="unacc") 0.toDouble else 1.toDouble, 
		//  Vectors.dense(parts(0).toDouble,parts(1).toDouble,parts(2).toDouble,parts(3).toDouble,parts(4).toDouble,parts(5).toDouble ))
		// }

		//多分类
		val parsedData = data.map { line =>
		 val parts = line.split(',');
		 LabeledPoint(if(parts(6)=="unacc") 0.toDouble else if(parts(6)=="acc") 1.toDouble else if(parts(6)=="good") 2.toDouble else 3.toDouble,
		 Vectors.dense(parts(0).toDouble,parts(1).toDouble,parts(2).toDouble,parts(3).toDouble,parts(4).toDouble,parts(5).toDouble ))
		}

		//将训练集分成训练和测试
		val splits = parsedData.randomSplit(Array(0.7, 0.3))
		val (trainingData, testData) = (splits(0), splits(1))

		//设置决策树的参数
		val numClasses = 4  // 4 for multiclasses
		val categoricalFeaturesInfo = Map[Int, Int]()
		val impurity = "gini"
		val maxDepth = 6
		val maxBins = 32

		val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

		//训练模型
		val labelAndPreds = testData.map { point =>
		 val prediction = model.predict(point.features)
		 (point.label, prediction)
		}

		//输出树的结构
		println("Learned model:\n" + model.toDebugString)

		//在验证集上的结果
		val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
		val testAcc = (1-testErr)
		println("testAccuracy:" + testAcc)

		//读取测试集并进行实验
		val testTree = sc.textFile("data/test-double.txt")

		val tmp = testTree.map { line => val parts = line.split(',')
		 LabeledPoint(0,Vectors.dense(parts(0).toDouble,parts(1).toDouble,parts(2).toDouble,parts(3).toDouble,parts(4).toDouble,parts(5).toDouble ))
		}

		val Preds = tmp.map { point =>
		 val prediction = model.predict(point.features)
		 (point.label, prediction)
		 }

		// val preds = Preds.map { line => if(line._2 == 1) "good" else "bad"} 

		val preds = Preds.map { line => if(line._2 == 0) "unacc" else if(line._2 == 1) "acc" else if(line._2 == 2) "good" else if(line._2 == 3) "vgood"} 

		preds.saveAsTextFile("data/multipreds")
	}
}
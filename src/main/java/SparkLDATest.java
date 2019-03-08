import java.util.Map;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.clustering.LDAModel;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

/**
 * @author Victor Oliveira
 */
public class SparkLDATest {

	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf()
			.setMaster("local")
			.setAppName("SparkLDATest");

		JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);

		String path = "data/kotlin_sample.txt";

		JavaRDD<String> document = javaSparkContext.textFile(path);

		JavaRDD<Vector> parsedData = document.map(row -> {
			String[] words = row.trim().split(" ");

			double[] values = new double[words.length];

			for (int i = 0; i < words.length; i++) {
				values[i] = Double.parseDouble(words[i]);
			}

			return Vectors.dense(values);
		});


		// Index documents with unique IDs
		JavaPairRDD<Long, Vector> corpus =
			JavaPairRDD.fromJavaRDD(parsedData.zipWithIndex().map(Tuple2::swap));

		corpus.cache();

		LDAModel ldaModel = new LDA().run(corpus);

		// Output topics. Each is a distribution over words (matching word count vectors)
		System.out.println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize()
			+ " words):");

		Matrix topics = ldaModel.topicsMatrix();

		for (int topic = 0; topic < 3; topic++) {
			System.out.print("Topic " + topic + ":");
			for (int word = 0; word < ldaModel.vocabSize(); word++) {
				System.out.print(" " + topics.apply(word, topic));
			}
			System.out.println();
		}

		ldaModel.save(javaSparkContext.sc(),
			"liferay_analytics/tags/lda_output");

		javaSparkContext.stop();
	}
}

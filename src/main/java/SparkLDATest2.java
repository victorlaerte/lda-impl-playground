import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * @author Victor Oliveira
 */
public class SparkLDATest2 {

	public static void main(String[] args) {
		SparkSession spark = SparkSession
			.builder()
			.master("local")
			.appName("SparkLDATest2")
			.getOrCreate();

		// $example on$
		// Loads data.
		Dataset<Row> dataset = spark.read().format("libsvm")
			.load("data/kotlin_sample.txt");

		// Trains a LDA model.
		LDA lda = new LDA().setK(10).setMaxIter(10);
		LDAModel model = lda.fit(dataset);

		double ll = model.logLikelihood(dataset);
		double lp = model.logPerplexity(dataset);
		System.out.println("The lower bound on the log likelihood of the entire corpus: " + ll);
		System.out.println("The upper bound on perplexity: " + lp);

		// Describe topics.
		Dataset<Row> topics = model.describeTopics(3);
		System.out.println("The topics described by their top-weighted terms:");
		topics.show(false);

		// Shows the result.
		Dataset<Row> transformed = model.transform(dataset);
		transformed.show(false);
		// $example off$

		spark.stop();
	}
}

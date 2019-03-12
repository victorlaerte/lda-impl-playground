import java.util.ArrayList;
import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import scala.Function1;
import scala.collection.Iterable;
import scala.collection.immutable.List;

/**
 * @author Marcellus Tavares
 */
public class SparkLDATest3 {


	// https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3741049972324885/3783546674231782/4413065072037724/latest.html

	public static void main(String[] args) {
		SparkSession spark = SparkSession
			.builder()
			.master("local")
			.appName("SparkLDATest2")
			.getOrCreate();


		spark.udf().register("translate", new UDF2<Iterable<Integer>, Iterable<String>, String[]>() {

			@Override
			public String[] call(Iterable<Integer> integerIterable, Iterable<String> vocabulary) throws Exception {
				List<String> stringList = vocabulary.toList();
				java.util.List<String> terms = new ArrayList<>();

				integerIterable.foreach(new Function1<Integer, Void>() {
					@Override
					public Void apply(Integer v1) {
						terms.add(stringList.apply(v1));
						return null;
					}
				});

				return terms.toArray(new String[terms.size()]);
			}
		}, DataTypes.createArrayType(DataTypes.StringType));

		Dataset<Row> rows = spark
			.read()
			.textFile("data/kotlin_sample.txt")
			.withColumnRenamed("value", "corpus")
			.withColumn("id", functions.row_number().over(Window.orderBy(functions.lit(1))))
			.withColumn("tokens", functions.split(functions.col("corpus"), " "));

		CountVectorizerModel countVectorizerModel = new CountVectorizer()
			.setInputCol("tokens")
			.setOutputCol("features")
			.setVocabSize(10000)
			.setMinDF(5)
			.fit(rows);

		Dataset<Row> corpusDataset = countVectorizerModel
			.transform(rows)
			.select("id", "features");

		corpusDataset.show();

		LDA lda = new LDA()
			//.setOptimizer(new OnlineLDAOptimizer().setMiniBatchFraction(0.8))
			.setK(10)
			.setMaxIter(10);

		Dataset<Row> topicsDataset = lda
			.fit(corpusDataset)
			.describeTopics(10);

		topicsDataset
			.withColumn("terms", functions.callUDF("translate", functions.col("termIndices"), functions.lit(countVectorizerModel.vocabulary())))
			.show(false);

		spark.stop();
	}
}
